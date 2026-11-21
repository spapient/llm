//! 2–8bit 自适应组量化，以及它在谱隙补偿链路里的形态。
//!
//! 与父模块的分工：父模块回答「给定一档位宽怎么量化、怎么打包」，本模块只回答**每组该选哪一档**
//! ——按误差预算从 `{2,3,4,8}` 里降档，使位宽图在给定 Frobenius 预算下近似最少码位（E5 的自适应
//! 策略口径）。码本与编解码仍只有 `ptq` 那一份 [`crate::ptq::quantize`] / [`crate::ptq::dequantize`]，本模块
//! 不另写算术，也不定义第二个位宽闭集。

use std::ops::Range;

use diagnostic::Result;

use crate::ptq::{I2, I3, I4, I8, Scheme, dequantize, invalid, quantize};

/// 候选档位，位宽从低到高。
///
/// 次序即降档方向：`8 → 4 → 3 → 2`，每次只朝低一位走，因此舍入误差随档位单调增、码位数单调减，
/// 贪心的每一步换来的都是「省下的 bit」与「多出来的误差」的确定交换。
const LADDER: [Scheme; 4] = [I2, I3, I4, I8];

/// 与 [`LADDER`] 一一对应的元素位宽。
///
/// 对外只暴露真实位宽，档位下标是内部记账：调用方拿 `2` 而不是 `0` 去理解一个组。
const BITS: [u8; 4] = [2, 3, 4, 8];

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
        widths: levels.iter().map(|&level| BITS[level]).collect(),
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
    /// 重构 `Ŵ = L_r + Q_adaptive(C)`。
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
    let Some(parts) =     crate::ptq::decompose(w, m, n)? else {
        return Ok(None);
    };
    let adaptive = adaptive_for(group, &parts, target_snr_db)?;
    Ok(Some(assemble(parts, adaptive)))
}

/// 入口两项校验：矩阵几何与目标信噪比。
fn validate(m: usize, n: usize, w: &[f32], target_snr_db: f64) -> Result<()> {
    crate::ptq::validate_compensate(m, n, w)?;
    check_target(target_snr_db)
}

/// 按目标信噪比推出误差预算并量化残差。
fn adaptive_for(group: usize, parts: &crate::ptq::Components, target_snr_db: f64) -> Result<Adaptive> {
    let target_err = parts.norm * 10f64.powf(-target_snr_db / 20.0);
    adaptive_quantize(group, &parts.rest, target_err)
}

/// 把分解结果与量化结果合成一份反馈记录：低秩部分逐元素加回量化残差。
fn assemble(parts: crate::ptq::Components, adaptive: Adaptive) -> AdaptiveCompensated {
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
///
/// 负预算下**任何**位宽都算超支，降档会在第一步就停在 8bit——那不是「预算太紧」，而是调用方给错
/// 了数，所以在这里拒绝而不是静默退化成固定 8bit。
fn check_budget(group: usize, x: &[f32], target_err: f64) -> Result<()> {
    if group == 0 {
        return Err(invalid("量化分组必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(invalid("输入必须是有限 f32"));
    }
    if target_err.is_nan() || target_err < 0.0 {
        return Err(invalid("目标误差必须非负"));
    }
    Ok(())
}

/// 目标 SNR 必须有限：`+inf` 会推出零预算、`NaN` 会推出 `NaN` 预算，两者都等于把「没给目标」
/// 当成「零误差」，而真实结果只会在别处以「未知能否达标」的形态暴露。
fn check_target(target_snr_db: f64) -> Result<()> {
    if target_snr_db.is_finite() {
        return Ok(());
    }
    Err(invalid("目标 SNR 必须有限"))
}

/// 第 `g` 组在 `len` 个元素里的区间；末组可能短于 `group`。
fn span(group: usize, len: usize, g: usize) -> Range<usize> {
    let lo = g * group;
    lo..(lo + group).min(len)
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
///
/// 均匀量化误差的解析期望是 `n·(max_abs/max_code)²/12`：只有 `max_code` 随档位变，因此降档的增误
/// 完全由码域大小决定，不需要真量化一遍去测。
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
///
/// 排序按「每省 1 bit 的增误」而不是按绝对增误：8→4 省 4 bit、3→2 只省 1 bit，直接比增误会让宽档
/// 永远排在前面，得到的位宽图就不是最少码位而是最宽位宽。
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
    let saved = f64::from(BITS[from] - BITS[to]);
    Some((delta, delta / saved))
}

/// 逐组按选定位宽量化再反解，拼出重构。
///
/// 编解码只走父模块那一份：自适应改变的是「每组用哪档」，不是「一档怎么算」，所以这里没有任何
/// 自己的舍入与打包逻辑。
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
        .map(|(g, &level)| BITS[level] as usize * span(group, len, g).count())
        .sum();
    total as f64 / len as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 档位表与位宽表必须同长：两处错位会让 `widths` 报出另一档的位宽。
    #[test]
    fn ladder_and_bits_stay_in_step() {
        assert_eq!(LADDER.len(), BITS.len());
        for (level, &scheme) in LADDER.iter().enumerate() {
            assert_eq!(scheme.bits, BITS[level]);
        }
    }

    /// 分组区间：末组按剩余元素收窄，不越界也不丢元素。
    #[test]
    fn spans_cover_the_input_exactly() {
        assert_eq!(span(4, 10, 0), 0..4);
        assert_eq!(span(4, 10, 2), 8..10);
        assert_eq!(span(4, 8, 1), 4..8);
    }

    /// 极紧预算下所有组停在最宽档：`I8` 也有均匀量化误差，零预算因此一律未达标。
    ///
    /// 「停在 8bit」与「达标」是两件事：预估误差是 `n·scale²/12`，`scale` 只在 `max_abs = 0` 时
    /// 才为零，所以零预算下没有一档能过——把它读成「8bit 无损」会掩盖这一步其实没达标。
    #[test]
    fn tight_budget_keeps_every_group_wide() {
        let x = [0.5f32, -0.25, 1.0, -2.0];
        let adaptive = adaptive_quantize(2, &x, 0.0).expect("合法输入");
        assert_eq!(adaptive.widths, vec![8, 8]);
        assert!(!adaptive.met);
        assert_eq!(adaptive.bpw, 8.0);
        // 预算为零 ⇒ 全部停在 `I8`，逐位等于父模块的固定 `I8` 往返（不是「近似无损」）。
        let (packed, scales) = quantize(I8, 2, &x).expect("合法输入");
        let mut expect = vec![0.0f32; x.len()];
        dequantize(I8, &packed, &scales, 2, &mut expect).expect("合法输入");
        assert_eq!(adaptive.recon, expect);
    }

    /// 极宽预算下所有组降到最窄档：贪心没有「只降一档就停」的口子。
    #[test]
    fn loose_budget_reaches_the_narrowest_rung() {
        let x = [0.5f32, -0.25, 1.0, -2.0];
        let adaptive = adaptive_quantize(2, &x, 1e9).expect("合法输入");
        assert_eq!(adaptive.widths, vec![2, 2]);
        assert!(adaptive.met);
        assert_eq!(adaptive.bpw, 2.0);
    }

    /// 预算不可能达标时 `met = false`，位宽图仍给出（调用方据此决定拒压，而不是拿到空结果）。
    #[test]
    fn unmet_budget_still_reports_the_map() {
        let x = [0.5f32, -0.25, 1.0, -2.0];
        let adaptive = adaptive_quantize(2, &x, 0.0).expect("合法输入");
        let tight = adaptive_quantize(2, &x, 1e-12).expect("合法输入");
        assert!(!tight.met);
        assert_eq!(tight.widths, adaptive.widths);
    }

    /// 预算越宽，选出的位宽只会更低：单调性是贪心可用的前提。
    #[test]
    fn wider_budgets_never_pick_wider_widths() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut previous = u8::MAX;
        for budget in [0.0f64, 0.5, 2.0, 8.0, 32.0] {
            let adaptive = adaptive_quantize(8, &x, budget).expect("合法输入");
            let widest = adaptive.widths.iter().copied().max().expect("非空");
            assert!(widest <= previous, "预算 {budget} 反而选了更宽的档");
            previous = widest;
        }
    }

    /// 三项入口契约各自独立失败：分组为零、非有限输入、负预算。
    #[test]
    fn invalid_inputs_are_rejected_before_any_allocation() {
        assert!(adaptive_quantize(0, &[1.0f32], 1.0).is_err());
        assert!(adaptive_quantize(2, &[1.0f32, f32::NAN], 1.0).is_err());
        assert!(adaptive_quantize(2, &[1.0f32, 2.0], -0.5).is_err());
        assert!(LADDER.iter().all(|s| s.max_code > 0.0));
    }

    /// 补偿形态在无谱隙时拒压，而不是把满秩矩阵当成低秩。
    #[test]
    fn compensation_refuses_without_a_spectral_gap() {
        let w = [1.0f32, 0.0, 0.0, 1.0];
        assert!(
            compensate_adaptive(2, &w, 2, 2, 20.0)
                .expect("合法形状")
                .is_none()
        );
    }

    /// 目标 SNR 非有限在入口被拒：`+inf` 会静默退化成零误差预算。
    #[test]
    fn non_finite_target_snr_is_rejected() {
        let w = [1.0f32, 0.0, 0.0, 1.0];
        assert!(compensate_adaptive(2, &w, 2, 2, f64::INFINITY).is_err());
        assert!(compensate_adaptive(2, &w, 2, 2, f64::NAN).is_err());
    }

    /// 确定性合成矩阵：`r` 条 rank-1 项加正弦噪声（固定公式，无随机源）。
    ///
    /// E5 明确「不得使用合成矩阵代替模型级结果」，所以这条只作**机制探针**：它量的是「位宽图与
    /// 平均码位是否随预算单调、重构 SNR 是否达到同一预算」，不出 E5 的判决。
    fn synthetic(m: usize, n: usize, rank: usize, noise: f32) -> Vec<f32> {
        let mut w = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for k in 0..rank {
                    let a = ((row * (k + 1)) as f32 * 0.017).sin();
                    let b = ((col * (k + 1)) as f32 * 0.013).cos();
                    acc += a * b;
                }
                let jitter = ((row * 31 + col * 17) as f32 * 0.001).sin() * noise;
                w[row * n + col] = acc + jitter;
            }
        }
        w
    }

    /// 实测探针：一层合成权重上的秩、位宽图、平均码位与**权重空间** SNR。
    ///
    /// 命令：`cargo test -p llm-weights --release --lib -- --ignored e5_width_map --nocapture`。
    ///
    /// 报的 SNR 是权重空间口径——E5 禁止拿它替代输出空间 SNR，这里只用来确认「预算越紧、位宽
    /// 越宽、SNR 越高」这条单调关系在实现里真的成立。
    #[test]
    #[ignore = "实测探针：打印一层合成权重的位宽图与权重空间 SNR，不参与默认门禁"]
    fn e5_width_map() {
        let (m, n, group) = (512usize, 512usize, 32usize);
        let w = synthetic(m, n, 8, 0.01);
        let energy: f64 = w.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        println!("E5 探针：{m}×{n}（rank-8 + 噪声），group={group}，‖W‖²_F={energy:.1}");
        for target in [30.0f64, 40.0, 50.0, 60.0] {
            let Some(card) = compensate_adaptive(group, &w, m, n, target).expect("合法输入")
            else {
                println!("  target {target} dB：无谱隙，拒压");
                continue;
            };
            let residual: f64 = card
                .recon
                .iter()
                .zip(&w)
                .map(|(&a, &b)| {
                    let d = f64::from(a) - f64::from(b);
                    d * d
                })
                .sum();
            let snr = 10.0 * (energy / residual.max(f64::MIN_POSITIVE)).log10();
            let mut histogram = [0usize; 4];
            for width in &card.widths {
                let slot = match width {
                    2 => 0,
                    3 => 1,
                    4 => 2,
                    _ => 3,
                };
                histogram[slot] += 1;
            }
            println!(
                "  target {target} dB：rank={} gap={:.3e} sigma1={:.1} bpw={:.3} met={} \
                 位宽图 2/3/4/8 = {}/{}/{}/{}，权重空间 SNR={snr:.2} dB",
                card.rank,
                card.gap,
                card.sigma1,
                card.bpw,
                card.met,
                histogram[0],
                histogram[1],
                histogram[2],
                histogram[3]
            );
            assert!(
                card.met,
                "目标 {target} dB 未达标：位宽图仍给出，但预算没满足"
            );
        }
    }
}
