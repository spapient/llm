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
mod tests {
    use super::*;
    use crate::prof::Profile;
    use diagnostic::data::Value;

    fn profile() -> Profile {
        Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案")
    }

    fn op() -> OpDecl {
        let source = &op::builtin_named("gemv_i4").expect("内置");
        OpDecl::parse(source.declaration(), source.template()).expect("声明")
    }

    fn path(mma_share: f64) -> Path {
        Path {
            variant: "pipe2".to_string(),
            index: 0,
            schedule: crate::sch::Schedule::parse("pipe<2>").expect("调度"),
            mma_share,
        }
    }

    /// 降级链的最后一个变体：`direct` 没有流水，掩盖账必然超出阈值。
    fn fallback(mma_share: f64) -> Path {
        Path {
            variant: "direct".to_string(),
            index: 1,
            schedule: crate::sch::Schedule::Direct,
            mma_share,
        }
    }

    /// 与算子无关的代价：`plan` 只消费这个数对，不认识任何流量公式。
    fn cost() -> Cost {
        Cost {
            flops: 2.0 * 4096.0 * 4096.0,
            read: 8 * 1024 * 1024,
            write: 0,
        }
    }

    #[test]
    fn roofline_is_the_bottleneck_plus_unmasked_part() {
        let profile = profile();
        let cost = cost();
        // `mma_share = 1` 时未掩盖项为 0：剩下的只有两个下界里的大者。
        let masked = Masking::of(1.0, &crate::sch::Schedule::Direct);
        let compute_only = predict(&profile, cost.flops, 0, masked).expect("可预测");
        let memory_only = predict(&profile, 0.0, cost.bytes(), masked).expect("可预测");
        let both = predict(&profile, cost.flops, cost.bytes(), masked).expect("可预测");
        assert!((both - compute_only.max(memory_only)).abs() < 1e-18);
    }

    #[test]
    fn select_takes_the_declared_tile() {
        let plan = select(&op(), &profile(), &path(0.62), cost()).expect("可选择");
        assert_eq!(plan.variant, "pipe2");
        assert_eq!(plan.tile.summary(), "GMAX=32");
        assert!(plan.predicted_s > 0.0);
        assert_eq!(plan.flops, cost().flops);
        assert_eq!(plan.bytes, cost().bytes());
        assert!(plan.masking.within_limit(), "首选变体应当达标");
    }

    #[test]
    fn the_plan_line_reports_the_judgement_it_used() {
        // 计划行里的百分比与阈值结论必须是判定的那两个数，而不是另算一份。
        let plan = select(&op(), &profile(), &path(0.62), cost()).expect("可选择");
        let line = plan.to_string();
        assert!(line.contains("未掩盖 19%（阈值 20%，达标）"), "{line}");
    }

    #[test]
    fn a_falling_back_variant_is_reported_not_rejected() {
        // 兜底变体的掩盖账超阈值是事实，不是错误：`select` 照旧出计划，行里写明它是降级兜底。
        let plan = select(&op(), &profile(), &fallback(0.35), cost()).expect("兜底也应有计划");
        assert_eq!(plan.variant, "direct");
        assert!(!plan.masking.within_limit(), "{}", plan.masking);
        assert!(
            plan.to_string()
                .contains("未掩盖 65%（阈值 20%，未达标：降级兜底）"),
            "{plan}"
        );
    }

    #[test]
    fn declaration_order_decides_the_tile_not_its_size() {
        // 轴取值顺序即优先级：把 64 写在前面就取 64，即使它更大。
        let source = &op::builtin_named("gemv_i4").expect("内置");
        let mut reordered = source.declaration().clone();
        reordered.set(
            "axes.GMAX",
            Value::array(vec![Value::int(64), Value::int(32)]),
        );
        assert_ne!(&reordered, source.declaration(), "声明里的轴写法变了");
        let op = OpDecl::parse(&reordered, source.template()).expect("声明");
        let plan = select(&op, &profile(), &path(0.62), cost()).expect("可选择");
        assert_eq!(plan.tile.summary(), "GMAX=64");
    }

    #[test]
    fn prediction_is_the_same_for_every_tile() {
        // 模型里没有可比较的量：两个候选的预测必须相同，否则说明有人给 predict 编了常数。
        let source = &op::builtin_named("gemv_i4").expect("内置");
        let mut swapped = source.declaration().clone();
        swapped.set(
            "axes.GMAX",
            Value::array(vec![Value::int(32), Value::int(64)]),
        );
        let wide = OpDecl::parse(&swapped, source.template()).expect("声明");
        let narrow = select(&op(), &profile(), &path(0.62), cost()).expect("可选择");
        let wide = select(&wide, &profile(), &path(0.62), cost()).expect("可选择");
        assert_eq!(narrow.predicted_s, wide.predicted_s);
        assert_eq!(narrow.bytes, wide.bytes);
    }

    #[test]
    fn profile_without_peaks_is_an_error() {
        let mut profile = profile();
        profile.peak = crate::prof::Peak::default();
        let error = select(&op(), &profile, &path(0.62), cost()).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("roofline"), "{error}");
    }
}
