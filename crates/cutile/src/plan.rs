//! ③ select：按声明顺序取 tile，roofline 只给估值（README §9、§12.1）。
//!
//! P0 不在 tile 候选之间比较：档案里没有能推导占用率或分块收益的常数（`[simt]` 只有组宽度），
//! 与 tile 相关的量只有 `deq[GMAX]` 的栈占用，它不改 flops、不改 bytes——任何"这个 tile 更快"的
//! 预测都只能编数字。所以轴取值的顺序就是优先级，与降级链同一口径：TOML 顺序即优先级，
//! 不自动打分。plan 表、±1 档 3 候选微基准与失效指纹是 P3 交付物。

use crate::cap::Path;
use crate::decl::{OpDecl, TileCombo, expand};
use crate::prof::Profile;
use crate::sch::Masking;
use core::fmt;
use diagnostic::{Code, Error};
use op::model::Cost;

/// 一次 plan 的产物。
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// 变体名。
    pub variant: String,
    /// 调度。
    pub schedule: String,
    /// 声明的 tile 取值（`[axes]` 笛卡尔积的第一个组合）。
    pub tile: TileCombo,
    /// 预测耗时（秒）。
    pub predicted_s: f64,
    /// 计算量（FLOP）。
    pub flops: f64,
    /// 流量（字节）。
    pub bytes: u64,
    /// 掩盖账：`mma_share`、级数与未掩盖占比（判据与报告用的是同一组数）。
    pub masking: Masking,
}

fn scale_flops(flops: f64) -> String {
    if flops >= 1e9 {
        format!("{:.1} GFLOP", flops / 1e9)
    } else {
        format!("{:.1} MFLOP", flops / 1e6)
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} [{}] → {:.3} ms（{}，{:.1} MB，{}）",
            self.variant,
            self.schedule,
            self.tile,
            self.predicted_s * 1e3,
            scale_flops(self.flops),
            self.bytes as f64 / 1e6,
            self.masking
        )
    }
}

/// roofline 预测：`max(flops/峰值, bytes/带宽) + 未掩盖项`。
///
/// 未掩盖项取自 [`Masking`] 而不是另算一遍：`doctor`/`bench` 打印的那个百分比，与这里加进去的
/// 必须是同一个数（README §12.2 判据 1）。与 tile 无关——这是 §9 的 P0 明确不做候选比较的原因，
/// 不是遗漏。
pub fn predict(profile: &Profile, flops: f64, bytes: u64, masking: Masking) -> Option<f64> {
    let compute = profile
        .peak
        .flops_f32
        .as_ref()
        .filter(|constant| constant.value > 0.0)
        .map(|constant| flops / constant.value);
    let memory = profile
        .peak
        .bw
        .as_ref()
        .filter(|constant| constant.value > 0.0)
        .map(|constant| bytes as f64 / constant.value);
    let base = match (compute, memory) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return None,
    };
    Some(base * (1.0 + masking.unmasked))
}

/// ③ select 本体：取轴展开出的**第一个** tile，再用算子给出的代价做 roofline 估值。
///
/// 代价（`flops`/`bytes`）由算子层 `op::model::Model::cost` 给出——流量公式属于算子语义，
/// 机制层不认识任何算子。
pub fn select(op: &OpDecl, profile: &Profile, path: &Path, cost: Cost) -> diagnostic::Result<Plan> {
    let masking = Masking::of(path.mma_share, &path.schedule);
    Ok(Plan {
        variant: path.variant.clone(),
        schedule: path.schedule.as_str(),
        tile: first_tile(op)?,
        predicted_s: predicted_s(profile, cost, masking)?,
        flops: cost.flops,
        bytes: cost.bytes(),
        masking,
    })
}

/// 轴展开出的第一个 tile：TOML 顺序即优先级，不在这里打分。
fn first_tile(op: &OpDecl) -> diagnostic::Result<TileCombo> {
    expand(op)?.into_iter().next().ok_or_else(|| {
        Error::new(
            Code::Internal,
            "轴展开后没有任何 tile 候选（声明解析应当保证至少一个轴）",
        )
    })
}

/// roofline 估值；峰值是能力：档案没给出可求值的峰值就无法判定，而不是“预测为 0”。
fn predicted_s(profile: &Profile, cost: Cost, masking: Masking) -> diagnostic::Result<f64> {
    predict(profile, cost.flops, cost.bytes(), masking).ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!(
                "档案 `{}` 没有可推导的 peak 常数，无法做 roofline 估算",
                profile.meta.vendor
            ),
        )
    })
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
