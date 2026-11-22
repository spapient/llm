//! 职责：GGUF 分块反量化——`F32`/`F16` 标量与 `Q4_0`/`Q5_0`/`Q5_1`/`Q8_0`/`Q2_K`/`Q3_K`/
//! `Q4_K`/`Q5_K`/`Q6_K` 分块格式的位解包 → f32。
//!
//! 每块元素数与字节数不在这里，一律问 [`Kind`]（它再委托 `dtype` 的显式布局）：本模块只做
//! 「块内位域 → 数值」。传统 Q 与 K 超块各自一个还原函数，尺度（`d`/`dmin`/`scales`）与量化点
//! （`qs`/`qh`/`hmask`）的位宽、偏移都是 GGML 磁盘格式的固定约定，逐字节对照 ggml-quants.c。

use diagnostic::Result;

use wire::bad;

use crate::gguf::Kind;

/// 把一个张量的原始字节反量化进 `out`；长度不符或布局不符显式报错。
pub(crate) fn dequantize(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    if raw.len() != kind.bytes_of(out.len())? {
        return Err(bad("张量字节长度与块布局不符"));
    }
    decode(kind, raw, out)
}

/// 按类型分派：两种标量格式留在这里，分块格式下沉一层。
fn decode(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::F32 => decode_scalar::<4>(raw, out, "f32", f32::from_le_bytes),
        Kind::F16 => decode_scalar::<2>(raw, out, "f16", |b| f16_to_f32(u16::from_le_bytes(b))),
        _ => blocked(kind, raw, out),
    }
}

/// 分块格式的位解包：传统 Q（`Q4_0`/`Q5_0`/`Q5_1`/`Q8_0`）与 K 超块（`Q2_K`~`Q6_K`）各自分派。
fn blocked(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::Q4_0 => decode_q4_0(raw, out),
        Kind::Q5_0 => decode_q5_0(raw, out),
        Kind::Q5_1 => decode_q5_1(raw, out),
        Kind::Q8_0 => decode_q8(raw, out),
        Kind::Q2_K => decode_q2k(raw, out),
        Kind::Q3_K => decode_q3k(raw, out),
        Kind::Q4_K => decode_k::<false>(raw, out),
        Kind::Q5_K => decode_k::<true>(raw, out),
        Kind::Q6_K => decode_q6k(raw, out),
        _ => Err(bad(format!("{kind:?} 不是分块格式"))),
    }
}

/// 标量格式：固定宽度小端，逐元素转换；长度不符即报错（`what` 是类型名）。
fn decode_scalar<const N: usize>(
    raw: &[u8],
    out: &mut [f32],
    what: &str,
    convert: impl Fn([u8; N]) -> f32,
) -> Result<()> {
    let (chunks, tail) = raw.as_chunks::<N>();
    if !tail.is_empty() || chunks.len() != out.len() {
        return Err(bad(format!("{what} 长度不符")));
    }
    for (slot, block) in out.iter_mut().zip(chunks) {
        *slot = convert(*block);
    }
    Ok(())
}

/// 半精度位模式 → f32，含次正规、无穷与 NaN。
///
/// 写成无分支的位运算而不是三条 `return`：早退分支会让 LLVM 放弃对整个循环向量化
/// （F16 张量逐元素都要走这里）。实测同进程交错 20 轮、两种输入分布均稳定快 1.23×，
/// 且全部 65536 个位模式与旧实现逐位相同（NaN 也比 bits）。
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from(bits >> 10) & 0x1F;
    let frac = u32::from(bits & 0x03FF);
    // 指数：1..=30 直接 +112 重偏；31（inf/NaN）必须抬到 f32 的 255，用 select 而非分支。
    let e = if exp == 31 { 255 } else { exp + 112 };
    let normal = sign | (e << 23) | (frac << 13);
    // 次正规数 = frac×2⁻²⁴。把 (1+frac/1024)×2⁻¹⁴ 当作 f32 算出来再减去 2⁻¹⁴，
    // 正好剩 frac×2⁻²⁴；省掉「数前导零再移位」那条分支。
    let magic = f32::from_bits((frac + 0x1C400) << 13);
    let denorm = (magic - f32::from_bits(0x3880_0000)).to_bits();
    f32::from_bits(if exp == 0 { sign | denorm } else { normal })
}

fn half_at(raw: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

fn decode_q8(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<34>();
    let (slots, _) = out.as_chunks_mut::<32>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        for (value, &q) in slot.iter_mut().zip(&block[2..]) {
            *value = d * f32::from(q as i8);
        }
    }
    Ok(())
}

/// Q4_K 的 6 位 scale/min 解包。
fn scale_min(scales: &[u8], idx: usize) -> (u8, u8) {
    if idx < 4 {
        return (scales[idx] & 63, scales[idx + 4] & 63);
    }
    let high = (scales[idx - 4] >> 6) << 4;
    let low = (scales[idx] >> 6) << 4;
    (
        (scales[idx + 4] & 0x0F) | high,
        (scales[idx + 4] >> 4) | low,
    )
}

/// Q4_K / Q5_K 共用的解码：Q5_K 多一段 `qh`（每个元素的第 5 位），块长度多 32 字节。
/// 两者每块都是 4 个 64 元素的组，组内高低半字节分别写 `lane` 与 `lane + 32`。
///
/// `HIGH` 为编译期常量而不是运行时开关：Q4_K 是磁盘上最常见的方向，它的内循环里
/// 因此没有 `qh` 的取值与越界检查（`high_bit` 只剩 Q5_K 一份下游）。
fn decode_k<const HIGH: bool>(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (block, high_bytes) = if HIGH { (176, 32) } else { (144, 0) };
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, bytes) in slots.iter_mut().zip(raw.chunks_exact(block)) {
        let d = half_at(bytes, 0);
        let min = half_at(bytes, 2);
        let scales = &bytes[4..16];
        let (high, quantized) = bytes[16..].split_at(high_bytes);
        let (groups, _) = quantized.as_chunks::<32>();
        for (group, qs) in groups.iter().enumerate() {
            let (s0, m0) = scale_min(scales, group * 2);
            let (s1, m1) = scale_min(scales, group * 2 + 1);
            let (ds0, mm0) = (d * f32::from(s0), min * f32::from(m0));
            let (ds1, mm1) = (d * f32::from(s1), min * f32::from(m1));
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                let high0 = high_bit::<HIGH>(high, lane, group * 2);
                let high1 = high_bit::<HIGH>(high, lane, group * 2 + 1);
                let q0 = (qs[lane] & 0x0F) + high0;
                let q1 = (qs[lane] >> 4) + high1;
                row[lane] = ds0 * f32::from(q0) - mm0;
                row[lane + 32] = ds1 * f32::from(q1) - mm1;
            }
        }
    }
    Ok(())
}

/// Q5_K 的第 5 位；`HIGH` 为 `false`（Q4_K）时整条取值路径在编译期消失，恒为 0。
fn high_bit<const HIGH: bool>(high: &[u8], lane: usize, bit: usize) -> u8 {
    match high.get(lane) {
        Some(byte) if HIGH => u8::from(byte & (1 << bit) != 0) << 4,
        _ => 0,
    }
}

fn decode_q6k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<210>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 208);
        for part in 0..2 {
            let ql = &block[part * 64..][..64];
            let qh = &block[128 + part * 32..][..32];
            let scales = &block[192 + part * 8..][..8];
            let mut ds = [0.0f32; 8];
            for (slot, &s) in ds.iter_mut().zip(scales) {
                *slot = d * f32::from(s as i8);
            }
            let row = &mut slot[part * 128..][..128];
            for lane in 0..32 {
                let base = lane / 16;
                let q1 = (ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4);
                let q2 = (ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4);
                let q3 = (ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4);
                let q4 = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4);
                row[lane] = ds[base] * (q1 as i32 - 32) as f32;
                row[lane + 32] = ds[base + 2] * (q2 as i32 - 32) as f32;
                row[lane + 64] = ds[base + 4] * (q3 as i32 - 32) as f32;
                row[lane + 96] = ds[base + 6] * (q4 as i32 - 32) as f32;
            }
        }
    }
    Ok(())
}

/// `Q4_0`：块 32、18 字节。前 2 字节 f16 scale，后 16 字节 32 个 4-bit 量化点，低半字节是前 16 元素。
fn decode_q4_0(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<18>();
    let (slots, _) = out.as_chunks_mut::<32>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        for (j, &q) in block[2..].iter().enumerate() {
            slot[j] = d * (f32::from(q & 0x0F) - 8.0);
            slot[j + 16] = d * (f32::from(q >> 4) - 8.0);
        }
    }
    Ok(())
}

/// `Q5_0`：块 32、22 字节。f16 scale + 4 字节第 5 位 + 16 字节低 4 位；量化点有符号（减 16）。
fn decode_q5_0(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<22>();
    let (slots, _) = out.as_chunks_mut::<32>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for (j, &q) in block[6..].iter().enumerate() {
            let hi0 = (((qh >> j) & 1) as u8) << 4;
            let hi1 = (((qh >> (j + 16)) & 1) as u8) << 4;
            slot[j] = d * (f32::from((q & 0x0F) | hi0) - 16.0);
            slot[j + 16] = d * (f32::from((q >> 4) | hi1) - 16.0);
        }
    }
    Ok(())
}

/// `Q5_1`：块 32、24 字节。f16 scale + f16 min + 4 字节第 5 位 + 16 字节低 4 位；无符号量化点加 min。
fn decode_q5_1(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<24>();
    let (slots, _) = out.as_chunks_mut::<32>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        let m = half_at(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for (j, &q) in block[8..].iter().enumerate() {
            let hi0 = (((qh >> j) & 1) as u8) << 4;
            let hi1 = (((qh >> (j + 16)) & 1) as u8) << 4;
            slot[j] = d * f32::from((q & 0x0F) | hi0) + m;
            slot[j + 16] = d * f32::from((q >> 4) | hi1) + m;
        }
    }
    Ok(())
}

/// `Q2_K`：块 256、84 字节。f16 `d`+`dmin`、16 个 4-bit scale/min、64 字节 2-bit 量化点。
/// 每 16 元素一个 scale，低 4 位是尺度乘数、高 4 位是 min 乘数；量化点有符号（2-bit）。
fn decode_q2k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<84>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        let min = half_at(block, 2);
        let scales = &block[4..20];
        let qs = &block[20..84];
        let mut is = 0usize;
        let mut y = 0usize;
        for n in 0..2 {
            let q = &qs[n * 32..];
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    slot[y] = dl * f32::from(((q[l] >> shift) & 3) as i8) - ml;
                    y += 1;
                }
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0F);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    slot[y] = dl * f32::from(((q[l + 16] >> shift) & 3) as i8) - ml;
                    y += 1;
                }
                shift += 2;
            }
        }
    }
    Ok(())
}

/// `Q3_K`：块 256、110 字节。32 字节 mask + 64 字节 2-bit 量化点 + 12 字节 6-bit scale +
/// 尾 2 字节 f16 `d`。mask 的第 `m` 位补每个量化点的第 3 位；scale 按交叉排列解包（减 32 偏置）。
fn decode_q3k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<110>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d_all = half_at(block, 108);
        let hm = &block[0..32];
        let qs = &block[32..96];
        let scales = unpack_q3_scales(&block[96..108]);
        let mut is = 0usize;
        let mut y = 0usize;
        let mut m: u8 = 1;
        for n in 0..2 {
            let q = &qs[n * 32..];
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl = d_all * f32::from(scales[is] - 32);
                is += 1;
                for l in 0..16 {
                    let sub = if hm[l] & m == 0 { 4 } else { 0 };
                    slot[y] = dl * f32::from(((q[l] >> shift) & 3) as i8 - sub);
                    y += 1;
                }
                let dl = d_all * f32::from(scales[is] - 32);
                is += 1;
                for l in 0..16 {
                    let sub = if hm[l + 16] & m == 0 { 4 } else { 0 };
                    slot[y] = dl * f32::from(((q[l + 16] >> shift) & 3) as i8 - sub);
                    y += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(())
}

/// `Q3_K` 的 12 字节 scales → 16 个 6-bit 尺度：交叉排列的位解包，逐字节对照 ggml-quants.c。
fn unpack_q3_scales(raw: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    aux[1] = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    aux[2] = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (i, word) in aux.iter().enumerate() {
        for (j, byte) in word.to_le_bytes().iter().enumerate() {
            scales[i * 4 + j] = *byte as i8;
        }
    }
    scales
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 位模式 → 小端字节。
    fn f16_bytes(bits: u16) -> [u8; 2] {
        bits.to_le_bytes()
    }

    #[test]
    fn q4_0_dequantizes_a_known_block() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[2] = 0x08; // 元素 0 = 8-8 = 0；元素 16 = 0-8 = -8
        block[3] = 0x19; // 元素 1 = 9-8 = 1；元素 17 = 1-8 = -7
        let mut out = [0.0f32; 32];
        decode_q4_0(&block, &mut out).expect("解码");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], -8.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[17], -7.0);
    }

    #[test]
    fn q5_0_dequantizes_a_known_block() {
        let mut block = [0u8; 22];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[6] = 0x10; // 元素 0 = 0-16 = -16；元素 16 = 1-16 = -15
        block[2] = 0x01; // qh bit 0 = 1 → 元素 0 第 5 位 = 1 → 16-16 = 0
        let mut out = [0.0f32; 32];
        decode_q5_0(&block, &mut out).expect("解码");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], -15.0);
    }

    #[test]
    fn q5_1_dequantizes_a_known_block() {
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[2..4].copy_from_slice(&f16_bytes(0x3800)); // m = 0.5
        block[8] = 0x10; // 元素 0 = 0，元素 16 = 1
        let mut out = [0.0f32; 32];
        decode_q5_1(&block, &mut out).expect("解码");
        assert_eq!(out[0], 0.5);
        assert_eq!(out[16], 1.5);
    }

    #[test]
    fn q2_k_dequantizes_a_known_block() {
        let mut block = [0u8; 84];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[2..4].copy_from_slice(&f16_bytes(0x3800)); // dmin = 0.5
        block[4..20].fill(0x11); // 每个 scale：低 4 位 = 1（dl=1.0）、高 4 位 = 1（ml=0.5）
                                 // qs 全 0 → 量化点 0 → 输出 = 1.0*0 - 0.5 = -0.5
        let mut out = [0.0f32; 256];
        decode_q2k(&block, &mut out).expect("解码");
        assert!(out.iter().all(|&v| (v + 0.5).abs() < 1e-6));
    }

    #[test]
    fn q3_k_scales_unpack_crosses_6bit_fields() {
        // 全 0xFF 的 6-bit 交叉排列解包后是 16 个 63。
        assert_eq!(unpack_q3_scales(&[0xFF; 12]), [63i8; 16]);
    }

    #[test]
    fn q3_k_dequantizes_a_zero_block() {
        let mut block = [0u8; 110];
        block[108..110].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[0..32].fill(0xFF); // mask 全 1，不扣 4
                                 // scales 全 0 → dl = 1.0*(0-32) = -32；qs 全 0 → 量化点 0 → 输出 0
        let mut out = [0.0f32; 256];
        decode_q3k(&block, &mut out).expect("解码");
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
