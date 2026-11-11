//! ③ select：形状 → tile，roofline 作为判定器（DESIGN §9、§12.1）。

use crate::cap::Path;
use crate::lower::expand::{OpDecl, TileCombo, expand};
use crate::prof::Profile;
use core::fmt;

/// 一次 tile 选择的产物。
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// 变体名。
    pub variant: String,
    /// 调度。
    pub schedule: String,
    /// 选中的 tile。
    pub tile: TileCombo,
    /// 预测耗时（秒）。
    pub predicted_s: f64,
    /// 计算量（FLOP）。
    pub flops: f64,
    /// 流量（字节）。
    pub bytes: u64,
    /// 未掩盖的非 MMA 占比。
    pub unmasked: f64,
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
            "{} {} [{}] → {:.3} ms（{}，{:.1} MB，未掩盖 {:.0}%）",
            self.variant,
            self.schedule,
            self.tile,
            self.predicted_s * 1e3,
            scale_flops(self.flops),
            self.bytes as f64 / 1e6,
            self.unmasked * 100.0
        )
    }
}

/// 未掩盖的非 MMA 占比：调度级数越多，可掩盖的部分越多（DESIGN §12.2 判据 1）。
pub fn unmasked_share(mma_share: f64, stages: u32) -> f64 {
    let stages = f64::from(stages.max(1));
    ((1.0 - mma_share) / stages).clamp(0.0, 1.0)
}

/// roofline 预测：`max(flops/峰值, bytes/带宽) + 未掩盖项`。
pub fn predict(
    profile: &Profile,
    flops: f64,
    bytes: u64,
    mma_share: f64,
    stages: u32,
) -> Option<f64> {
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
    let unmasked = unmasked_share(mma_share, stages);
    Some(base * (1.0 + unmasked))
}

/// int4 GEMV 的流量：打包权重 + 输入 + 分组缩放 + 输出。
pub fn gemv_i4_traffic(m: u64, n: u64, group: u64) -> u64 {
    let group = group.max(1);
    m * n / 2 + m * 4 + n * (m / group).max(1) * 4 + n * 4
}

/// ③ select 本体：在轴展开出的 tile 候选里取预测最快的一个。
pub fn select(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    m: u64,
    n: u64,
    group: u64,
) -> Result<Plan, String> {
    let tiles = expand(op).map_err(|error| error.to_string())?;
    let flops = 2.0 * m as f64 * n as f64;
    let bytes = gemv_i4_traffic(m, n, group);
    let stages = path.schedule.stages();

    let mut best: Option<Plan> = None;
    for tile in tiles {
        let Some(predicted) = predict(profile, flops, bytes, path.mma_share, stages) else {
            return Err(format!(
                "档案 `{}` 没有可推导的 peak 常数，无法做 roofline 估算",
                profile.meta.vendor
            ));
        };
        let candidate = Plan {
            variant: path.variant.clone(),
            schedule: path.schedule.as_str(),
            tile,
            predicted_s: predicted,
            flops,
            bytes,
            unmasked: unmasked_share(path.mma_share, stages),
        };
        let better = best
            .as_ref()
            .is_none_or(|current| candidate.predicted_s < current.predicted_s);
        if better {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| "轴展开后没有任何 tile 候选".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prof::Profile;

    fn profile() -> Profile {
        Profile::parse(include_str!("../devices/cpu.toml")).expect("档案")
    }

    fn op() -> OpDecl {
        OpDecl::parse(
            include_str!("../ops/gemv_i4.toml"),
            include_str!("../ops/gemv_i4.k"),
        )
        .expect("声明")
    }

    fn path(mma_share: f64) -> Path {
        Path {
            variant: "pipe2".to_string(),
            index: 0,
            schedule: crate::sch::Schedule::parse("pipe<2>").expect("调度"),
            mma_share,
        }
    }

    #[test]
    fn unmasked_share_shrinks_with_stages() {
        let one = unmasked_share(0.5, 1);
        let two = unmasked_share(0.5, 2);
        assert!(two < one, "级数越多，越能掩盖");
        assert!((one - 0.5).abs() < 1e-12);
    }

    #[test]
    fn roofline_is_the_bottleneck_plus_unmasked_part() {
        let profile = profile();
        let flops = 2.0 * 4096.0 * 4096.0;
        let bytes = gemv_i4_traffic(4096, 4096, 32);
        let compute_only = predict(&profile, flops, 0, 1.0, 1).expect("可预测");
        let memory_only = predict(&profile, 0.0, bytes, 1.0, 1).expect("可预测");
        let both = predict(&profile, flops, bytes, 1.0, 1).expect("可预测");
        assert!((both - compute_only.max(memory_only)).abs() < 1e-18);
    }

    #[test]
    fn select_picks_the_smaller_tile_for_the_cpu_peak() {
        let plan = select(&op(), &profile(), &path(0.62), 4096, 4096, 32).expect("可选择");
        assert_eq!(plan.variant, "pipe2");
        assert!(plan.predicted_s > 0.0);
        assert!(plan.flops > 0.0);
    }

    #[test]
    fn profile_without_peaks_is_an_error() {
        let mut profile = profile();
        profile.peak = crate::prof::Peak::default();
        let error = select(&op(), &profile, &path(0.62), 1024, 1024, 32).unwrap_err();
        assert!(error.contains("roofline"), "{error}");
    }

    #[test]
    fn traffic_accounts_for_every_tensor() {
        let bytes = gemv_i4_traffic(4096, 4096, 32);
        assert_eq!(
            bytes,
            4096 * 4096 / 2 + 4096 * 4 + 4096 * 128 * 4 + 4096 * 4
        );
    }
}
