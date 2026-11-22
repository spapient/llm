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

/// 按类型分派：两种标量格式留在这里，分块格式下沉一层。
fn decode(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::F32 => decode_scalar::<4>(raw, out, "f32", f32::from_le_bytes),
        Kind::F16 => decode_scalar::<2>(raw, out, "f16", |b| f16_to_f32(u16::from_le_bytes(b))),
        _ => blocked(kind, raw, out),
    }
}

/// 四种分块格式：各自的位解包；参数是同块字节数与前 32 个量化点的偏移。
fn blocked(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::Q8_0 => decode_q8(raw, out),
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
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                let high0 = high_bit::<HIGH>(high, lane, group * 2);
                let high1 = high_bit::<HIGH>(high, lane, group * 2 + 1);
                let q0 = (qs[lane] & 0x0F) + high0;
                let q1 = (qs[lane] >> 4) + high1;
                row[lane] = d * f32::from(s0) * f32::from(q0) - min * f32::from(m0);
                row[lane + 32] = d * f32::from(s1) * f32::from(q1) - min * f32::from(m1);
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
