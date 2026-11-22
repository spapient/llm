//! 2/3/4/8-bit 对称训练后量化（PTQ）：把 f32 权重按组量化并打包，以及它的反解。

use std::ops::Range;

use crate::bad;
use crate::svd;
use diagnostic::Result;

/// 一档对称量化格式的运行时事实。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scheme {
    /// 元素位宽：2 / 3 / 4 / 8。
    pub bits: u8,
    /// 对称零点：偏移码（i2/i3/i4）的 `signed + zero` 平移量；i8 为 0。
    pub zero: i32,
    /// 正码上限：`scale = max_abs / max_code` 的分母。
    pub max_code: f32,
}

impl Scheme {
    /// 码域大小：`2^bits`。
    pub const fn domain(self) -> i32 {
        1 << self.bits
    }

    /// 每个码占的位数。
    pub const fn width(self) -> usize {
        self.bits as usize
    }

    /// 打包字节数：`ceil(len * bits / 8)`。
    pub fn packed_len(self, len: usize) -> usize {
        len.saturating_mul(self.bits as usize).div_ceil(8)
    }
}

/// 2 位打包：4 码/字节，零点 2，码表 `{-2,-1,0,1}`，正码上限 1。
pub const I2: Scheme = Scheme {
    bits: 2,
    zero: 2,
    max_code: 1.0,
};

/// 3 位打包：8 码/3 字节，零点 3，码表 `{-3..=3}`，正码上限 3。
pub const I3: Scheme = Scheme {
    bits: 3,
    zero: 3,
    max_code: 3.0,
};

/// 4 位打包：2 码/字节，零点 8，码表 `{-8..=7}`，正码上限 7。
pub const I4: Scheme = Scheme {
    bits: 4,
    zero: 8,
    max_code: 7.0,
};

/// 8 位逐元素：1 码/字节，零点 0（二补码 `i8`），正码上限 127。
pub const I8: Scheme = Scheme {
    bits: 8,
    zero: 0,
    max_code: 127.0,
};

/// 按组对称量化：把 `x` 按 `group` 个元素一组，每组独立 `scale = max_abs / max_code`，
/// 逐元素 `signed = round(x/scale)` 夹取到 `[-max_code, max_code]`，再按低码在前打包。
///
/// 返回 `(packed, scales)`：`packed` 长度 `ceil(len * bits / 8)`，`scales` 长度 `ceil(len / group)`。
/// 全零组取 `scale = 1.0`（码全为对称零点 `zero`），反解仍精确归零，同时避开零 scale 被反解侧拒绝。
///
/// # Errors
///
/// `group` 为 0、输入含非有限值，或 `max_code` 非法（0）时返回说明。
pub fn quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
    validate_quantize(scheme, group, x)?;

    let groups = x.len().div_ceil(group);
    let mut packed = vec![0u8; scheme.packed_len(x.len())];
    let mut scales = Vec::with_capacity(groups);

    for g in 0..groups {
        let lo = g * group;
        let hi = (lo + group).min(x.len());
        let slice = &x[lo..hi];
        let scale = group_scale(scheme, slice);
        scales.push(scale);

        for (i, &v) in slice.iter().enumerate() {
            let signed = (v / scale).round() as i32;
            let signed = signed.clamp(-(scheme.max_code as i32), scheme.max_code as i32);
            pack_bits(
                &mut packed,
                lo + i,
                scheme.bits,
                (signed + scheme.zero) as u8,
            );
        }
    }

    Ok((packed, scales))
}

/// 量化参数校验：分组、正码上限、输入有限性。
fn validate_quantize(scheme: Scheme, group: usize, x: &[f32]) -> Result<()> {
    if group == 0 {
        return Err(bad("量化分组必须大于 0"));
    }
    if scheme.max_code <= 0.0 {
        return Err(bad("正码上限必须大于 0"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(bad("输入必须是有限 f32"));
    }
    Ok(())
}

/// 一组元素的对称标度：全零组取 1（码全为对称零点，反解仍精确归零），否则 `max_abs / max_code`。
fn group_scale(scheme: Scheme, slice: &[f32]) -> f32 {
    let max_abs = slice.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / scheme.max_code
    }
}

/// 按组反解：`y = scale * (code - zero)`，与反量化公式一致，供 round-trip 对拍。
///
/// `packed` 长度必须为 `ceil(len * bits / 8)`，`scales` 长度必须为 `ceil(len / group)`，`out` 长度
/// 为元素数。i2/i3/i4 的码是无符号偏移码（减 `zero` 得符号值），i8 的码是二补码字节（`zero = 0`）。
///
/// # Errors
///
/// 输出长度不为 0 但分组/字节数不匹配时返回说明。
pub fn dequantize(
    scheme: Scheme,
    packed: &[u8],
    scales: &[f32],
    group: usize,
    out: &mut [f32],
) -> Result<()> {
    validate_dequantize(scheme, packed, scales, group, out)?;

    // 按组取 scale：`scales[g]` 在整组内不变，逐元素 `i / group` 是一次多余的整数除法，
    // 且把尺度的读取变成随索引跳动的访存。按组切片后每组只取一次 scale。
    let len = out.len();
    for (g, &scale) in scales.iter().enumerate() {
        let lo = g * group;
        let hi = (lo + group).min(len);
        if lo >= hi {
            break;
        }
        for (offset, slot) in out[lo..hi].iter_mut().enumerate() {
            let field = unpack_bits(packed, lo + offset, scheme.bits);
            *slot = scale * decode_code(scheme, field);
        }
    }
    Ok(())
}

/// 反解参数校验：分组、打包字节数与输出长度、标度条数与分组数。
fn validate_dequantize(
    scheme: Scheme,
    packed: &[u8],
    scales: &[f32],
    group: usize,
    out: &[f32],
) -> Result<()> {
    if group == 0 {
        return Err(bad("反解分组必须大于 0"));
    }
    if packed.len() != scheme.packed_len(out.len()) {
        return Err(bad("打包字节数与输出长度不一致"));
    }
    if scales.len() != out.len().div_ceil(group) {
        return Err(bad("标度条数与分组数不一致"));
    }
    Ok(())
}

/// 码 → 符号值：i8（`zero == 0`）按二补码解释，i2/i3/i4 是无符号偏移码减零点。
fn decode_code(scheme: Scheme, field: u8) -> f32 {
    if scheme.zero == 0 {
        field as i8 as f32
    } else {
        field as f32 - scheme.zero as f32
    }
}

/// 把第 `idx` 个 `width` 位码写入位流（低码在前）；跨字节边界时高位溢到下一字节。
fn pack_bits(packed: &mut [u8], idx: usize, width: u8, code: u8) {
    let bit = idx * width as usize;
    let byte = bit / 8;
    let shift = (bit % 8) as u32;
    let value = (code as u32) << shift;
    packed[byte] |= value as u8;
    if shift + width as u32 > 8 {
        packed[byte + 1] |= (value >> 8) as u8;
    }
}

/// 从位流读出第 `idx` 个 `width` 位码；跨字节边界时低位溢到下一字节。
fn unpack_bits(packed: &[u8], idx: usize, width: u8) -> u8 {
    let bit = idx * width as usize;
    let byte = bit / 8;
    let shift = (bit % 8) as u32;
    let mut value = (packed[byte] as u32) >> shift;
    if shift + width as u32 > 8 {
        value |= (packed[byte + 1] as u32) << (8 - shift);
    }
    (value & ((1u32 << width) - 1)) as u8
}

/// 秩-r 低秩 + 残差量化补偿的结果：`Ŵ = L_r + Q(W − L_r)`。
#[derive(Clone, Debug)]
pub struct Compensated {
    /// 实际使用的低秩项秩（`rank` 与 SVD 可取秩的较小值）。
    pub rank: usize,
    /// 最大奇异值 `σ_1`（谱隙的相对参照）。
    pub sigma1: f64,
    /// 秩 r 处的谱隙 `γ_r = σ_r − σ_{r+1}`（r 为末位时取 `σ_r`）。
    pub gap: f64,
    /// 低秩项 `L_r`（行主序 `m*n` f32）。
    pub low: Vec<f32>,
    /// 残差 `C` 的打包码。
    pub packed: Vec<u8>,
    /// 残差 `C` 的组标度。
    pub scales: Vec<f32>,
    /// 重构 `Ŵ = L_r + Q(C)`（行主序 `m*n` f32）。
    pub recon: Vec<f32>,
}

/// 秩-r 低秩 + 残差量化补偿（自动谱隙证书门控）。
///
/// 流程：`W` 行主序 `m*n` → 一次 f64 SVD → [`svd::signal_rank`] 找信号/噪声边界秩 `r`
/// （无谱隙即拒压）→ 取前 `r` 个奇异方向做低秩项 `L_r` → 残差 `C = W − L_r` 按
/// [`quantize`] 做 `scheme`/`group` 对称量化 → [`dequantize`] 回解 `Q(C)` →
/// 重构 `Ŵ = L_r + Q(C)`。
///
/// 返回 `Ok(None)` 表示**拒压回退**：满秩/平坦谱矩阵没有信号谱隙（`signal_rank` 返回 `None`），
/// 4bit 量化误差不会因低秩分解而缩小，此时不硬压、不颁发证书。
///
/// # Errors
///
/// 形状非法、分组为 0、输入含非有限值，或 SVD 不收敛时返回说明。
pub fn compensate(
    scheme: Scheme,
    group: usize,
    w: &[f32],
    m: usize,
    n: usize,
) -> Result<Option<Compensated>> {
    validate_compensate(m, n, w)?;
    rebuild(scheme, group, w, m, n)
}

/// 数值侧：一次 f64 分解定低秩项（连它撑起的谱隙），显式残差再走同一个量化器。
fn rebuild(
    scheme: Scheme,
    group: usize,
    w: &[f32],
    m: usize,
    n: usize,
) -> Result<Option<Compensated>> {
    let Some(parts) = decompose(w, m, n)? else {
        return Ok(None);
    };
    let (packed, scales, qrest) = pack_residual(scheme, group, &parts.rest)?;
    let recon: Vec<f32> = parts.low.iter().zip(&qrest).map(|(l, q)| l + q).collect();
    Ok(Some(Compensated {
        rank: parts.rank,
        sigma1: parts.sigma1,
        gap: parts.gap,
        low: parts.low,
        packed,
        scales,
        recon,
    }))
}

/// 一次分解的产物：秩、谱隙、低秩项、显式残差与整体范数同出一门。
pub(crate) struct Components {
    pub(crate) rank: usize,
    pub(crate) sigma1: f64,
    pub(crate) gap: f64,
    pub(crate) low: Vec<f32>,
    pub(crate) rest: Vec<f32>,
    pub(crate) norm: f64,
}

/// 一次 f64 SVD → 秩 / 低秩项 `L_r` / 显式残差 `C = W − L_r`；无谱隙即拒压。
pub(crate) fn decompose(w: &[f32], m: usize, n: usize) -> Result<Option<Components>> {
    let a: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let d = svd::compute_svd(&a, m, n)?;
    let Some(rank) = svd::signal_rank(&d.sigma) else {
        return Ok(None);
    };
    let rows = d.reconstruct(rank);
    let low: Vec<f32> = rows.iter().map(|&x| x as f32).collect();
    let rest: Vec<f32> = a
        .iter()
        .zip(&rows)
        .map(|(orig, l)| (orig - l) as f32)
        .collect();
    Ok(Some(Components {
        rank,
        sigma1: d.sigma[0],
        gap: spectral_gap(&d.sigma, rank),
        low,
        rest,
        norm: a.iter().map(|v| v * v).sum::<f64>().sqrt(),
    }))
}

/// 截断处上方那道缝：`σ_r − σ_{r+1}`；已到最后一个奇异值时退化成 `σ_r`（下面没有值可供相减）。
fn spectral_gap(sigma: &[f64], r: usize) -> f64 {
    match sigma.get(r) {
        Some(&below) => sigma[r - 1] - below,
        None => sigma[r - 1],
    }
}

/// 残差打包 → 回解：编解码只有 [`quantize`] / [`dequantize`] 一份定义，补偿不另写一套。
fn pack_residual(
    scheme: Scheme,
    group: usize,
    rest: &[f32],
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let (packed, scales) = quantize(scheme, group, rest)?;
    let mut qrest = vec![0.0f32; rest.len()];
    dequantize(scheme, &packed, &scales, group, &mut qrest)?;
    Ok((packed, scales, qrest))
}

/// 补偿入口的两项契约：形状、有限性，每项各答一个问题。
pub(crate) fn validate_compensate(m: usize, n: usize, w: &[f32]) -> Result<()> {
    let cells = m
        .checked_mul(n)
        .filter(|&cells| cells > 0 && cells == w.len());
    if cells.is_none() {
        return Err(bad(format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    if !w.iter().all(|v| v.is_finite()) {
        return Err(bad("输入必须是有限 f32"));
    }
    Ok(())
}

// —— 2–8bit 自适应组量化（原 `adaptive` 平铺进 `ptq`）——

/// 候选档位，位宽从低到高。
const LADDER: [Scheme; 4] = [I2, I3, I4, I8];

/// 与 [`LADDER`] 一一对应的元素位宽。
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
    let Some(parts) = crate::ptq::decompose(w, m, n)? else {
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
fn adaptive_for(
    group: usize,
    parts: &crate::ptq::Components,
    target_snr_db: f64,
) -> Result<Adaptive> {
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
/// 当成「零误差」，而真实结果只会在别处以「未知能否达标」的形态暴露。
fn check_target(target_snr_db: f64) -> Result<()> {
    if target_snr_db.is_finite() {
        return Ok(());
    }
    Err(bad("目标 SNR 必须有限"))
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
    let saved = f64::from(BITS[from] - BITS[to]);
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
        .map(|(g, &level)| BITS[level] as usize * span(group, len, g).count())
        .sum();
    total as f64 / len as f64
}

#[cfg(test)]
#[path = "../tests/adaptive.rs"]
mod adaptive_tests;

#[cfg(test)]
#[path = "../tests/ptq.rs"]
mod tests;
