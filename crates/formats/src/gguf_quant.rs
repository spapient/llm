//! 职责：GGUF 分块反量化——`F32`/`F16` 标量与 `Q8_0`/`Q4_K`/`Q5_K`/`Q6_K` 超块的位解包 → f32。
//!
//! 每块元素数与字节数不在这里，一律问 [`Kind`]（它再委托 `dtype` 的显式布局）：本模块只做
//! 「块内位域 → 数值」。四种 K 量化与 `Q8_0` 各自一个还原函数，尺度（`d`/`dmin`/`scales`）
//! 与量化点（`qs`/`qh`）的位宽、偏移都是 GGML 磁盘格式的固定约定。

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

/// 按类型分派：标量格式走定宽转换，分块格式走各自的位解包。
fn decode(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::F32 => decode_scalar::<4>(raw, out, "f32", f32::from_le_bytes),
        Kind::F16 => decode_scalar::<2>(raw, out, "f16", |b| f16_to_f32(u16::from_le_bytes(b))),
        Kind::Q8_0 => decode_q8(raw, out),
        Kind::Q4_K => decode_k(raw, out, 144, 0),
        Kind::Q5_K => decode_k(raw, out, 176, 32),
        Kind::Q6_K => decode_q6k(raw, out),
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
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x03FF);
    if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (frac << 13))
    } else if exp == 0 {
        let mag = frac as f32 * 5.960_464_5e-8;
        f32::from_bits(sign | mag.to_bits())
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
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
fn scale_min(scales: &[u8], index: usize) -> (u8, u8) {
    if index < 4 {
        return (scales[index] & 63, scales[index + 4] & 63);
    }
    let high = (scales[index - 4] >> 6) << 4;
    let low = (scales[index] >> 6) << 4;
    (
        (scales[index + 4] & 0x0F) | high,
        (scales[index + 4] >> 4) | low,
    )
}

/// Q4_K / Q5_K 共用的解码：Q5_K 多一段 `qh`（每个元素的第 5 位），块长度多 32 字节。
/// 两者每块都是 4 个 64 元素的组，组内高低半字节分别写 `lane` 与 `lane + 32`。
fn decode_k(raw: &[u8], out: &mut [f32], block: usize, high_bytes: usize) -> Result<()> {
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
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                let high0 = high_bit(high, lane, group * 2);
                let high1 = high_bit(high, lane, group * 2 + 1);
                let q0 = (qs[lane] & 0x0F) + high0;
                let q1 = (qs[lane] >> 4) + high1;
                row[lane] = d * f32::from(s0) * f32::from(q0) - min * f32::from(m0);
                row[lane + 32] = d * f32::from(s1) * f32::from(q1) - min * f32::from(m1);
            }
        }
    }
    Ok(())
}

/// Q5_K 的第 5 位：`high` 为空（Q4_K）时恒为 0。
fn high_bit(high: &[u8], lane: usize, bit: usize) -> u8 {
    high.get(lane)
        .map_or(0, |byte| u8::from(byte & (1 << bit) != 0) << 4)
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
            let row = &mut slot[part * 128..][..128];
            for lane in 0..32 {
                let base = lane / 16;
                let q1 = (ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4);
                let q2 = (ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4);
                let q3 = (ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4);
                let q4 = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4);
                row[lane] = d * f32::from(scales[base] as i8) * (q1 as i32 - 32) as f32;
                row[lane + 32] = d * f32::from(scales[base + 2] as i8) * (q2 as i32 - 32) as f32;
                row[lane + 64] = d * f32::from(scales[base + 4] as i8) * (q3 as i32 - 32) as f32;
                row[lane + 96] = d * f32::from(scales[base + 6] as i8) * (q4 as i32 - 32) as f32;
            }
        }
    }
    Ok(())
}
