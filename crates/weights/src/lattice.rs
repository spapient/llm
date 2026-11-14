//! 格向量量化：D24、E8 与 3×E8 直积编解码。
//!
//! `D24 = { x ∈ Z^24 : Σx_i 为偶数 }`：逐坐标舍入后修正奇偶，每块一个 scale，实现简单但每个
//! 24 维块浪费一个奇偶位。
//!
//! E8 路径把 24 维切成 3 个 8 维块，每块用完整 E8：`E8 = D8 ∪ (D8 + ½)`，即整数坐标偶和
//! ∪ 半整数坐标偶和。`8·bits` 位索引恰好覆盖盒内全部 `2^(8·bits)` 个 E8 点（两陪集各
//! `2^(8·bits-1)`），不浪费奇偶位；同 bits 下有效码点数是 D24 的 2 倍。
//!
//! 旧版的端到端实测指出：每 8 维一个 scale 是体积劣势主因，残差后精度优势被抵消，因此生产
//! 不启用。scale 开销的攻克在本阶段单独处理（见 scale 模块的对拍），本模块先保证格内编解码
//! 数学正确。3×E8 是 Leech Λ24 的直积退化，球填充系数低于真 Λ24，报告数字时不宣称达到 Leech 密度。

use diagnostic::{Code, Error, Result};

/// D24 格维数。
pub const DIM: usize = 24;

/// E8 子块维数。
const WIDE: usize = 8;

/// 合法位宽：1..=8（索引与坐标都用 u8/定长字节表示）。
fn check_bits(bits: u8) -> Result<()> {
    if (1..=8).contains(&bits) {
        Ok(())
    } else {
        Err(Error::new(
            Code::Invalid,
            format!("lattice: 位宽必须在 1..=8，实际 {bits}"),
        )
        .with("lattice"))
    }
}

// ---------------------------------------------------------------- D24

/// 每块 scale：让最大坐标落在 `2^(bits-1)-1`。
#[must_use]
pub fn fit(block: &[f32], bits: u8) -> f32 {
    let hi = ((1i32 << (bits - 1)) - 1) as f32;
    let peak = block
        .iter()
        .take(DIM)
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if peak <= 0.0 || hi <= 0.0 {
        1.0
    } else {
        peak / hi
    }
}

fn snap(block: &[f32], scale: f32, bits: u8) -> [i8; DIM] {
    let inv = 1.0 / scale;
    let lo = -(1i32 << (bits - 1));
    let hi = (1i32 << (bits - 1)) - 1;
    let mut q = [0i8; DIM];
    for (slot, &x) in q.iter_mut().zip(block.iter().take(DIM)) {
        *slot = ((x * inv).round() as i32).clamp(lo, hi) as i8;
    }
    q
}

fn parity(q: &[i8]) -> i32 {
    q.iter().map(|&x| x as i32).sum::<i32>() % 2
}

/// 误差最大且可挪动的坐标与方向。
fn target(block: &[f32], scale: f32, q: &[i8; DIM]) -> (usize, i32) {
    let inv = 1.0 / scale;
    let mut pick = 0usize;
    let mut best = -1.0f32;
    let mut dir = 1i32;
    for (i, &x) in block.iter().take(DIM).enumerate() {
        let err = (x * inv - q[i] as f32).abs();
        if err > best {
            best = err;
            pick = i;
            dir = if x * inv > q[i] as f32 { 1 } else { -1 };
        }
    }
    (pick, dir)
}

fn nudge(bits: u8, pick: usize, dir: i32, q: &mut [i8; DIM]) {
    let lo = -(1i32 << (bits - 1));
    let hi = (1i32 << (bits - 1)) - 1;
    let base = q[pick] as i32;
    let next = if within(base + dir, lo, hi) {
        base + dir
    } else {
        base - dir
    };
    if within(next, lo, hi) {
        q[pick] = next as i8;
    }
}

/// 最近 D24 点：舍入后若奇偶不为偶，挪动误差最大且未饱和的坐标一格。
#[must_use]
pub fn nearest(block: &[f32], scale: f32, bits: u8) -> [i8; DIM] {
    let mut q = snap(block, scale, bits);
    if parity(&q) != 0 {
        let (pick, dir) = target(block, scale, &q);
        nudge(bits, pick, dir, &mut q);
    }
    q
}

/// 每块 `DIM * bits / 8` 字节，小端位流；`bits` 位补码。
#[must_use]
pub fn pack(q: &[i8; DIM], bits: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(DIM * bits as usize / 8 + 1);
    let mut acc = 0u64;
    let mut have = 0u32;
    for &x in q {
        acc |= ((x as u8 as u64) & ((1u64 << bits) - 1)) << have;
        have += bits as u32;
        while have >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push(acc as u8);
    }
    out
}

/// 位流 → 坐标；短读按 0 补齐，不 panic。
#[must_use]
pub fn unpack(bytes: &[u8], bits: u8) -> [i8; DIM] {
    let mask = (1u64 << bits) - 1;
    let mut q = [0i8; DIM];
    let mut acc = 0u64;
    let mut have = 0u32;
    let mut at = 0usize;
    for slot in q.iter_mut() {
        while have < bits as u32 {
            acc |= (bytes.get(at).copied().unwrap_or(0) as u64) << have;
            at += 1;
            have += 8;
        }
        let raw = acc & mask;
        *slot = ((raw << (64 - bits as u32)) as i64 >> (64 - bits as u32)) as i8;
        acc >>= bits;
        have -= bits as u32;
    }
    q
}

/// 全量编码：24 维分块，返回位流与每块 scale。
pub fn quantize(data: &[f32], bits: u8) -> Result<(Vec<u8>, Vec<f32>)> {
    check_bits(bits)?;
    let stride = DIM * bits as usize / 8 + usize::from(!(DIM * bits as usize).is_multiple_of(8));
    let mut packed = Vec::with_capacity(data.len().div_ceil(DIM) * stride);
    let mut scales = Vec::with_capacity(data.len().div_ceil(DIM));
    for chunk in data.chunks(DIM) {
        let scale = fit(chunk, bits);
        scales.push(scale);
        packed.extend_from_slice(&pack(&nearest(chunk, scale, bits), bits));
    }
    Ok((packed, scales))
}

/// 全量解码；`dst` 可长于 24 的整数倍，尾部按块内有效长度写。
pub fn dequantize(packed: &[u8], scales: &[f32], bits: u8, dst: &mut [f32]) -> Result<()> {
    check_bits(bits)?;
    let stride = DIM * bits as usize / 8 + usize::from(!(DIM * bits as usize).is_multiple_of(8));
    for (i, out) in dst.chunks_mut(DIM).enumerate() {
        let block = i * stride..(i + 1) * stride;
        let Some((&scale, bytes)) = scales.get(i).zip(packed.get(block)) else {
            break;
        };
        for (slot, &q) in out.iter_mut().zip(unpack(bytes, bits).iter()) {
            *slot = q as f32 * scale;
        }
    }
    Ok(())
}

fn within(v: i32, lo: i32, hi: i32) -> bool {
    v <= hi && v >= lo
}

// ---------------------------------------------------------------- E8

/// 每坐标可用的量化级数。
fn level(bits: u8) -> u32 {
    1u32 << bits
}

/// E8 半格最大坐标：`2^(bits-1)-1/2`。
fn span(bits: u8) -> f32 {
    (1u32 << (bits - 1)) as f32 - 0.5
}

fn peak(block: &[f32]) -> f32 {
    block.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
}

/// E8 块的尺度：峰值映到半格边界。
#[must_use]
pub fn fit8(block: &[f32], bits: u8) -> f32 {
    let hi = span(bits);
    let top = peak(block);
    if top <= 0.0 || hi <= 0.0 {
        1.0
    } else {
        top / hi
    }
}

/// 把浮点块量化到指定陪集的数字；缺尾按 0 补齐。
fn digits(block: &[f32], scale: f32, bits: u8, coset: u8) -> [u8; WIDE] {
    let inv = 1.0 / scale;
    let bias = (level(bits) / 2) as f32;
    let hi = (level(bits) - 1) as i32;
    let mut d = [0u8; WIDE];
    for (i, slot) in d.iter_mut().enumerate() {
        let x = block.get(i).copied().unwrap_or(0.0) * inv - 0.5 * coset as f32;
        *slot = ((x.round() as i32 + bias as i32).clamp(0, hi)) as u8;
    }
    d
}

fn odd(d: &[u8; WIDE]) -> bool {
    d.iter().map(|&v| v as u32).sum::<u32>() % 2 == 1
}

fn loss(block: &[f32], scale: f32, bits: u8, coset: u8, d: &[u8; WIDE]) -> f32 {
    let inv = 1.0 / scale;
    let bias = (level(bits) / 2) as f32;
    let shift = bias - 0.5 * coset as f32;
    d.iter()
        .enumerate()
        .map(|(i, &v)| {
            let err = block.get(i).copied().unwrap_or(0.0) * inv - (v as f32 - shift);
            err * err
        })
        .sum()
}

/// 单格挪动的误差增量；越界返回无穷大。
fn gain(
    block: &[f32],
    scale: f32,
    bits: u8,
    coset: u8,
    d: &[u8; WIDE],
    i: usize,
    next: i32,
) -> f32 {
    if !within(next, 0, (level(bits) - 1) as i32) {
        return f32::MAX;
    }
    let mut probe = *d;
    probe[i] = next as u8;
    loss(block, scale, bits, coset, &probe) - loss(block, scale, bits, coset, d)
}

/// 奇偶修正：在界内挪一格，取误差增量最小者。
fn adjust8(block: &[f32], scale: f32, bits: u8, coset: u8, d: &mut [u8; WIDE]) {
    let mut best = (f32::MAX, 0usize, 0i32);
    for (i, slot) in d.iter().enumerate() {
        for delta in [-1i32, 1] {
            let step = gain(block, scale, bits, coset, d, i, *slot as i32 + delta);
            if step < best.0 {
                best = (step, i, delta);
            }
        }
    }
    if best.2 != 0 {
        d[best.1] = (d[best.1] as i32 + best.2) as u8;
    }
}

fn settle(block: &[f32], scale: f32, bits: u8, coset: u8, d: &mut [u8; WIDE]) -> f32 {
    if odd(d) {
        adjust8(block, scale, bits, coset, d);
    }
    loss(block, scale, bits, coset, d)
}

/// 最近 E8 点：D8 与半整数陪集各解一次，取误差小者，返回格点坐标。
#[must_use]
pub fn nearest8(block: &[f32], scale: f32, bits: u8) -> [f32; WIDE] {
    let mut best = (f32::MAX, 0u8, [0u8; WIDE]);
    for coset in 0..2u8 {
        let mut d = digits(block, scale, bits, coset);
        let err = settle(block, scale, bits, coset, &mut d);
        if err < best.0 {
            best = (err, coset, d);
        }
    }
    let bias = (level(bits) / 2) as f32;
    let mut p = [0.0f32; WIDE];
    for (i, slot) in p.iter_mut().enumerate() {
        *slot = best.2[i] as f32 - bias + 0.5 * best.1 as f32;
    }
    p
}

fn classify(point: &[f32; WIDE]) -> u8 {
    u8::from((point[0] - point[0].round()).abs() > 0.25)
}

/// 偶数和的 `slots` 位数字序列按给定奇偶的计数。
fn count(slots: usize, need: u32, bits: u8) -> u64 {
    if slots == 0 {
        return u64::from(need == 0);
    }
    1u64 << (bits as usize * slots - 1)
}

/// 偶和数字序列的字典序秩，值域 `[0, 2^(8·bits-1))`。
fn rank(d: &[u8; WIDE], bits: u8) -> u64 {
    let mut out = 0u64;
    let mut par = 0u32;
    for (i, &digit) in d.iter().enumerate() {
        for value in 0..digit as u32 {
            out += count(WIDE - 1 - i, (par + value) & 1, bits);
        }
        par = (par + digit as u32) & 1;
    }
    out
}

/// 秩 → 偶和数字序列；越界秩按末位收敛，不 panic。
fn unrank(mut code: u64, bits: u8) -> [u8; WIDE] {
    let mut d = [0u8; WIDE];
    let mut par = 0u32;
    for (i, slot) in d.iter_mut().enumerate() {
        for value in 0..level(bits) {
            let need = (par + value) & 1;
            let take = count(WIDE - 1 - i, need, bits);
            if code < take {
                *slot = value as u8;
                par = need;
                break;
            }
            code -= take;
        }
    }
    d
}

/// E8 点到 `8·bits` 位索引：高位是陪集，低位是偶和数字秩。
#[must_use]
pub fn index8(point: &[f32; WIDE], bits: u8) -> u64 {
    let coset = classify(point);
    let bias = (level(bits) / 2) as f32;
    let mut d = [0u8; WIDE];
    for (i, slot) in d.iter_mut().enumerate() {
        *slot = (point[i] + bias - 0.5 * coset as f32).round() as u8;
    }
    (u64::from(coset) << (8 * u32::from(bits) - 1)) | rank(&d, bits)
}

/// `8·bits` 位索引 → E8 点。
#[must_use]
pub fn point8(index: u64, bits: u8) -> [f32; WIDE] {
    let shift = 8 * u32::from(bits) - 1;
    let coset = ((index >> shift) & 1) as u8;
    let d = unrank(index & ((1u64 << shift) - 1), bits);
    let bias = (level(bits) / 2) as f32;
    let mut p = [0.0f32; WIDE];
    for (i, slot) in p.iter_mut().enumerate() {
        *slot = d[i] as f32 - bias + 0.5 * f32::from(coset);
    }
    p
}

fn pack8(index: u64, bits: u8) -> Vec<u8> {
    index.to_le_bytes()[..bits as usize].to_vec()
}

fn unpack8(bytes: &[u8], bits: u8) -> u64 {
    let mut raw = [0u8; 8];
    for (i, slot) in raw.iter_mut().take(bits as usize).enumerate() {
        *slot = bytes.get(i).copied().unwrap_or(0);
    }
    u64::from_le_bytes(raw)
}

/// 8 维全量编码：每块 `bits` 字节索引 + 一个 scale。
pub fn quantize8(data: &[f32], bits: u8) -> Result<(Vec<u8>, Vec<f32>)> {
    check_bits(bits)?;
    let mut packed = Vec::with_capacity(data.len().div_ceil(WIDE) * bits as usize);
    let mut scales = Vec::with_capacity(data.len().div_ceil(WIDE));
    for chunk in data.chunks(WIDE) {
        let scale = fit8(chunk, bits);
        scales.push(scale);
        packed.extend_from_slice(&pack8(index8(&nearest8(chunk, scale, bits), bits), bits));
    }
    Ok((packed, scales))
}

/// 8 维全量解码；短读时剩余 `dst` 不写（调用方按 0 初始化即可）。
pub fn dequantize8(packed: &[u8], scales: &[f32], bits: u8, dst: &mut [f32]) -> Result<()> {
    check_bits(bits)?;
    let width = bits as usize;
    for (i, out) in dst.chunks_mut(WIDE).enumerate() {
        let at = i * width;
        let Some((&scale, bytes)) = scales.get(i).zip(packed.get(at..at + width)) else {
            break;
        };
        for (slot, &q) in out
            .iter_mut()
            .zip(point8(unpack8(bytes, bits), bits).iter())
        {
            *slot = q * scale;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- 3×E8（24 维）

/// 24 维编码：3×E8 直积（Leech Λ24 的退化替代），单块 scale。
pub fn quantize24(data: &[f32], bits: u8) -> Result<(Vec<u8>, Vec<f32>)> {
    check_bits(bits)?;
    let stride = DIM * bits as usize / 8;
    let mut packed = Vec::with_capacity(data.len().div_ceil(DIM) * stride);
    let mut scales = Vec::with_capacity(data.len().div_ceil(DIM));
    for chunk in data.chunks(DIM) {
        let scale = fit8(chunk, bits);
        scales.push(scale);
        for j in 0..DIM / WIDE {
            let sub = chunk.get(j * WIDE..).unwrap_or(&[]);
            packed.extend_from_slice(&pack8(index8(&nearest8(sub, scale, bits), bits), bits));
        }
    }
    Ok((packed, scales))
}

/// 24 维解码：每块读 3 个 E8 索引。
pub fn dequantize24(packed: &[u8], scales: &[f32], bits: u8, dst: &mut [f32]) -> Result<()> {
    check_bits(bits)?;
    let stride = DIM * bits as usize / 8;
    for (i, out) in dst.chunks_mut(DIM).enumerate() {
        let block = i * stride..(i + 1) * stride;
        let Some((&scale, bytes)) = scales.get(i).zip(packed.get(block)) else {
            break;
        };
        for (slot, &q) in out.iter_mut().zip(block24(bytes, bits).iter()) {
            *slot = q * scale;
        }
    }
    Ok(())
}

/// 一块 24 维字节 → 3 个 E8 点首尾相接；每个 8 维子块一个 `bits` 字节索引。
fn block24(bytes: &[u8], bits: u8) -> [f32; DIM] {
    let mut point = [0.0f32; DIM];
    for (sub, index) in point.chunks_mut(WIDE).zip(0..) {
        let at = index * bits as usize;
        sub.copy_from_slice(&point8(unpack8(&bytes[at..], bits), bits));
    }
    point
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn mse(back: &[f32], want: &[f32]) -> f64 {
        let sum: f64 = back
            .iter()
            .zip(want)
            .map(|(a, b)| f64::powi((a - b) as f64, 2))
            .sum();
        sum / want.len() as f64
    }

    #[test]
    fn parity_is_even() {
        for seed in 0..32 {
            let v = sample(DIM, seed);
            for bits in [2u8, 3, 4] {
                let q = nearest(&v, fit(&v, bits), bits);
                assert_eq!(parity(&q), 0, "seed={seed} bits={bits}");
            }
        }
    }

    #[test]
    fn bitstream_len() {
        for bits in [2u8, 3, 4] {
            assert_eq!(pack(&[0i8; DIM], bits).len(), DIM * bits as usize / 8);
        }
    }

    #[test]
    fn roundtrip_is_bounded() {
        let v = sample(DIM * 8, 7);
        for bits in [2u8, 3, 4] {
            let (p, s) = quantize(&v, bits).expect("编码");
            let mut back = vec![0.0f32; v.len()];
            dequantize(&p, &s, bits, &mut back).expect("解码");
            for (a, b) in v.iter().zip(&back) {
                assert!((a - b).abs() <= s[0] * 1.5 + 1e-6);
            }
        }
    }

    #[test]
    fn pack_unpack_identity() {
        let v = sample(DIM, 3);
        let scale = fit(&v, 4);
        let q = nearest(&v, scale, 4);
        assert_eq!(unpack(&pack(&q, 4), 4), q);
    }

    fn record(code: &[i32; WIDE], sum: i32, half: f32, counts: &mut [u32; 9]) {
        if sum % 2 != 0 {
            return;
        }
        let norm: f32 = code.iter().map(|&k| (k as f32 + half).powi(2)).sum();
        let norm = norm.round() as usize;
        if norm <= 8 {
            counts[norm] += 1;
        }
    }

    fn walk(
        pos: usize,
        code: &mut [i32; WIDE],
        sum: i32,
        span: (i32, i32),
        half: f32,
        counts: &mut [u32; 9],
    ) {
        if pos == WIDE {
            record(code, sum, half, counts);
            return;
        }
        for k in span.0..=span.1 {
            code[pos] = k;
            walk(pos + 1, code, sum + k, span, half, counts);
        }
    }

    /// E8 前四个壳的计数：240、2160、6720、17520。
    #[test]
    fn e8_shell_counts() {
        let mut counts = [0u32; 9];
        let mut code = [0i32; WIDE];
        walk(0, &mut code, 0, (-2, 2), 0.0, &mut counts);
        walk(0, &mut code, 0, (-3, 2), 0.5, &mut counts);
        assert_eq!(
            [counts[2], counts[4], counts[6], counts[8]],
            [240, 2160, 6720, 17520]
        );
    }

    #[test]
    fn e8_index_roundtrip() {
        for bits in [2u8, 3, 4, 8] {
            let room = 1u64 << (8 * u32::from(bits) - 1);
            for seed in 0..64u64 {
                let raw = sample(1, seed)[0].abs();
                let index =
                    ((raw as f64 * room as f64) as u64) | (seed & 1) << (8 * u32::from(bits) - 1);
                assert_eq!(index8(&point8(index, bits), bits), index);
            }
        }
    }

    #[test]
    fn e8_nearest_is_lattice() {
        for seed in 0..16 {
            let v = sample(WIDE, seed);
            for bits in [2u8, 3, 4] {
                let p = nearest8(&v, fit8(&v, bits), bits);
                let first = (p[0] - p[0].round()).abs();
                let sum: f32 = p.iter().sum();
                assert!((sum - sum.round()).abs() < 1e-4);
                assert_eq!((first - first.round()).abs() < 1e-4, first < 1e-4);
            }
        }
    }

    #[test]
    fn e8_roundtrip_bounded() {
        let v = sample(WIDE * 16, 11);
        for bits in [2u8, 3, 4] {
            let (p, s) = quantize8(&v, bits).expect("编码");
            let mut back = vec![0.0f32; v.len()];
            dequantize8(&p, &s, bits, &mut back).expect("解码");
            for (a, b) in v.iter().zip(&back) {
                assert!((a - b).abs() <= s[0] * 2.0 + 1e-6);
            }
        }
    }

    /// 同 bits 下 3×E8 的往返 MSE 必须低于 D24。
    #[test]
    fn e8_beats_d24() {
        let v = sample(DIM * 256, 23);
        for bits in [2u8, 3, 4] {
            let (dp, ds) = quantize(&v, bits).expect("d24");
            let (ep, es) = quantize24(&v, bits).expect("e8x3");
            let mut dback = vec![0.0f32; v.len()];
            let mut eback = vec![0.0f32; v.len()];
            dequantize(&dp, &ds, bits, &mut dback).expect("d24 解码");
            dequantize24(&ep, &es, bits, &mut eback).expect("e8 解码");
            let (dm, em) = (mse(&dback, &v), mse(&eback, &v));
            assert!(em < dm, "bits={bits} D24={dm} E8x3={em}");
        }
    }

    #[test]
    fn e8_short_tail() {
        let v = sample(11, 5);
        for bits in [2u8, 3] {
            let (p, s) = quantize24(&v, bits).expect("编码");
            assert_eq!(p.len(), 3 * bits as usize);
            let mut back = vec![0.0f32; v.len()];
            dequantize24(&p, &s, bits, &mut back).expect("解码");
            assert_eq!(back.len(), v.len());
        }
    }

    #[test]
    fn an_out_of_range_bit_width_is_rejected() {
        assert!(quantize(&[1.0, 2.0], 9).is_err());
        assert!(check_bits(0).is_err());
    }
}
