//! 2/3/4/8-bit 对称训练后量化（PTQ）：把 f32 权重按组量化并打包，以及它的反解。

pub use crate::adaptive::{Adaptive, AdaptiveCompensated, adaptive_quantize, compensate_adaptive};

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

#[cfg(test)]
#[path = "../tests/ptq.rs"]
mod tests;
