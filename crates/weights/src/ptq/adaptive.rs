//! 2–8bit 自适应组量化：按误差预算给每组从 `{2,3,4,8}` 里挑最小位宽，以及谱隙选秩后的自适应残差补偿。

use crate::bad;
use crate::ptq::{
    Components, I2, I3, I4, I8, Scheme, decompose, dequantize, quantize, span, validate_compensate,
};
use diagnostic::Result;

/// 候选档位，位宽从低到高（`Scheme::bits` 本身就是位宽，不再并行维护一张位宽表）。
const LADDER: [Scheme; 4] = [I2, I3, I4, I8];

/// 起始档位：所有组都从最宽的 `I8` 起，降到哪由预算决定。
const TOP: usize = 3;

/// 2–8bit 自适应分配的结果：每组的位宽、平均码位、是否达标、反解重构。
#[derive(Clone, Debug)]
pub struct Adaptive {
    /// 每组位宽（2 / 3 / 4 / 8，元素位宽而非档位下标）。
    pub widths: Vec<u8>,
    /// 平均码位（只计码本身，不含组标度与位宽表）。
    pub bpw: f64,
    /// 总误差是否落在目标误差预算内。
    pub met: bool,
    /// 反解重构 `Q(x)`。
    pub recon: Vec<f32>,
}

/// 2–8bit 自适应组量化：按误差预算给每组从 `{2,3,4,8}` 里挑最小位宽。
///
/// 判据是**误差驱动的贪心**：所有组先从 8bit 起，逐步把「每省 1 bit 增误最少」的组降一档
/// （8→4→3→2），直到再降就超出 `target_err`（Frobenius）为止。得到的位置在给定误差预算下近似
/// 最少码位，覆盖 2–8bit 候选（E5 的自适应策略口径）。
///
/// 返回 `recon` 是逐组用各自位宽量化再反解的 `Q(x)`；`met` 表示总误差是否落在预算内。
///
/// # Errors
///
/// 分组为 0、输入含非有限值、目标误差为负时返回说明。
pub fn adaptive_quantize(group: usize, x: &[f32], target_err: f64) -> Result<Adaptive> {
    check_budget(group, x, target_err)?;
    let groups = x.len().div_ceil(group);
    let maxes = maxima(group, x, groups);
    let target2 = target_err * target_err;
    // 内部用档位下标（0=I2 .. 3=I8）；对外只暴露真实位宽。
    let mut levels = vec![TOP; groups];
    let mut total = total_err2(&maxes, group, x.len(), &levels);
    descend(&maxes, group, x.len(), &mut levels, &mut total, target2);
    Ok(Adaptive {
        widths: levels.iter().map(|&level| LADDER[level].bits).collect(),
        bpw: average_bits(group, x.len(), &levels),
        met: total <= target2,
        recon: reconstruct(group, x, &levels)?,
    })
}

/// 自适应补偿的结果：谱隙选秩 + 低秩项 + 残差的 2–8bit 自适应量化。
#[derive(Clone, Debug)]
pub struct AdaptiveCompensated {
    /// 谱隙证书选出的秩。
    pub rank: usize,
    /// 最大奇异值。
    pub sigma1: f64,
    /// 秩 r 处的谱隙。
    pub gap: f64,
    /// 残差平均码位（只计码本身）。
    pub bpw: f64,
    /// 总误差是否落在目标误差预算内。
    pub met: bool,
    /// 每组位宽。
    pub widths: Vec<u8>,
    /// 重构 `Ŵ = L_r + Q_adaptive(C)`。
    pub recon: Vec<f32>,
}

/// 谱隙证书 + 2–8bit 自适应残差补偿：把 4bit 固定档换成按误差预算分配 `{2,3,4,8}`。
///
/// 与 [`crate::ptq::compensate`] 同一条低秩分解（`signal_rank` 选秩、无谱隙即拒压）；差别只在残差 `C`
/// 用 [`adaptive_quantize`] 量化，目标误差按 `target_snr_db` 从 `‖W‖_F` 推出。输出 SNR 的实测口径
/// 见调用方（Frobenius 口径对随机输入是输出空间 SNR 的下界，实测通常再高约 2 dB）。
///
/// 返回 `Ok(None)` 表示拒压（无谱隙）。
///
/// # Errors
///
/// 形状非法、输入含非有限值、`target_snr_db` 非有限，或 SVD 不收敛时返回说明。
pub fn compensate_adaptive(
    group: usize,
    w: &[f32],
    m: usize,
    n: usize,
    target_snr_db: f64,
) -> Result<Option<AdaptiveCompensated>> {
    validate(m, n, w, target_snr_db)?;
    let Some(parts) = decompose(w, m, n)? else {
        return Ok(None);
    };
    let adaptive = adaptive_for(group, &parts, target_snr_db)?;
    Ok(Some(assemble(parts, adaptive)))
}

/// 入口两项校验：矩阵几何与目标信噪比。
fn validate(m: usize, n: usize, w: &[f32], target_snr_db: f64) -> Result<()> {
    validate_compensate(m, n, w)?;
    check_target(target_snr_db)
}

/// 按目标信噪比推出误差预算并量化残差。
fn adaptive_for(group: usize, parts: &Components, target_snr_db: f64) -> Result<Adaptive> {
    let target_err = parts.norm * 10f64.powf(-target_snr_db / 20.0);
    adaptive_quantize(group, &parts.rest, target_err)
}

/// 把分解结果与量化结果合成一份反馈记录：低秩部分逐元素加回量化残差。
fn assemble(parts: Components, adaptive: Adaptive) -> AdaptiveCompensated {
    AdaptiveCompensated {
        rank: parts.rank,
        sigma1: parts.sigma1,
        gap: parts.gap,
        bpw: adaptive.bpw,
        met: adaptive.met,
        widths: adaptive.widths,
        recon: parts
            .low
            .iter()
            .zip(&adaptive.recon)
            .map(|(l, q)| l + q)
            .collect(),
    }
}

/// 入口三项契约：分组非零、输入有限、预算非负。
fn check_budget(group: usize, x: &[f32], target_err: f64) -> Result<()> {
    if group == 0 {
        return Err(bad("量化分组必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(bad("输入必须是有限 f32"));
    }
    if target_err.is_nan() || target_err < 0.0 {
        return Err(bad("目标误差必须非负"));
    }
    Ok(())
}

/// 目标 SNR 必须有限：`+inf` 会推出零预算、`NaN` 会推出 `NaN` 预算，两者都等于把「没给目标」
/// 当成「零误差」，而真实结果只会在别处以更难查的方式暴露。
fn check_target(target_snr_db: f64) -> Result<()> {
    if target_snr_db.is_finite() {
        return Ok(());
    }
    Err(bad("目标 SNR 必须有限"))
}

/// 每组最大绝对值：量化 `scale` 的分母只由它决定，与位宽无关，因此全档位共用一份。
fn maxima(group: usize, x: &[f32], groups: usize) -> Vec<f32> {
    (0..groups)
        .map(|g| {
            let range = span(group, x.len(), g);
            x[range].iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
        })
        .collect()
}

/// 第 `g` 组在档位 `level` 下的误差平方。
fn group_err2(maxes: &[f32], group: usize, len: usize, g: usize, level: usize) -> f64 {
    let size = span(group, len, g).count() as f64;
    let scale = f64::from(maxes[g]) / f64::from(LADDER[level].max_code);
    size * scale * scale / 12.0
}

/// 给定档位图的总误差平方（Frobenius 口径）。
fn total_err2(maxes: &[f32], group: usize, len: usize, levels: &[usize]) -> f64 {
    levels
        .iter()
        .enumerate()
        .map(|(g, &level)| group_err2(maxes, group, len, g, level))
        .sum()
}

/// 贪心降档：每轮降一档「每省 1 bit 增误最少」且仍不超预算的组，直到没有组还能降。
fn descend(
    maxes: &[f32],
    group: usize,
    len: usize,
    levels: &mut [usize],
    total: &mut f64,
    target2: f64,
) {
    while let Some((g, delta)) = cheapest(maxes, group, len, levels, *total, target2) {
        levels[g] -= 1;
        *total += delta;
    }
}

/// 这一轮最值得降的那一组：`(组号, 增误)`；没有组能在预算内降档时 `None`。
fn cheapest(
    maxes: &[f32],
    group: usize,
    len: usize,
    levels: &[usize],
    total: f64,
    target2: f64,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for (g, &from) in levels.iter().enumerate() {
        let Some((delta, cost)) = rung(maxes, group, len, g, from) else {
            continue;
        };
        if total + delta > target2 {
            continue;
        }
        if best.is_none_or(|(_, seen, _)| cost < seen) {
            best = Some((g, cost, delta));
        }
    }
    best.map(|(g, _, delta)| (g, delta))
}

/// 一组降一档的 `(增误, 每省 1 bit 的增误)`；已经到最窄档时 `None`。
fn rung(maxes: &[f32], group: usize, len: usize, g: usize, from: usize) -> Option<(f64, f64)> {
    let to = from.checked_sub(1)?;
    let delta = group_err2(maxes, group, len, g, to) - group_err2(maxes, group, len, g, from);
    let saved = f64::from(LADDER[from].bits - LADDER[to].bits);
    Some((delta, delta / saved))
}

/// 逐组按选定位宽量化再反解，拼出重构。
fn reconstruct(group: usize, x: &[f32], levels: &[usize]) -> Result<Vec<f32>> {
    let mut recon = vec![0.0f32; x.len()];
    for (g, &level) in levels.iter().enumerate() {
        let scheme = LADDER[level];
        let range = span(group, x.len(), g);
        let (packed, scales) = quantize(scheme, group, &x[range.clone()])?;
        dequantize(scheme, &packed, &scales, group, &mut recon[range])?;
    }
    Ok(recon)
}

/// 平均码位：只计码本身，不含组标度与位宽表（那两项按组摊薄，与逐元素码位不是一个量纲）。
fn average_bits(group: usize, len: usize, levels: &[usize]) -> f64 {
    let total: usize = levels
        .iter()
        .enumerate()
        .map(|(g, &level)| LADDER[level].bits as usize * span(group, len, g).count())
        .sum();
    total as f64 / len as f64
}

#[cfg(test)]
#[path = "../../tests/adaptive.rs"]
mod adaptive_tests;
