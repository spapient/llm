//! 五轴自适应选码：`raw / dict / sparse / quant / rank` 五个候选，每个候选只有在解码后与原始
//! 位模式逐元素一致（`to_bits` 相等）时才被采纳，最终在通过验证的候选里取字节最小者。
//!
//! `raw` 是恒可用的基线（f32 字节走 rANS），因此非空输入总有返回。`quant` 要求精确落在
//! `2^e` 网格、`rank` 要求低秩+残差的逐位还原，不满足即丢弃，绝不谎称无损。

use diagnostic::{Code, Error, Result};

use crate::leb::Cursor;
use crate::{ledger, rans};

/// artifact 魔数。
const MAGIC: [u8; 4] = *b"AX01";

/// 固定头长度：magic(4) + axis(1) + m(4) + n(4)。
const HEAD: usize = 13;

/// 候选轴（闭集，码位稳定）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// 原始 f32 字节，rANS 熵编码基线。
    Raw = 0,
    /// 唯一值字典 + 索引。
    Dict = 1,
    /// 非零位置 + 值。
    Sparse = 2,
    /// 精确 2^e 网格量化。
    Quant = 3,
    /// 精确低秩 + 残差。
    Rank = 4,
}

/// 全部轴，下标即码位（[`Axis::code`] 的逆）。
const AXES: [Axis; 5] = [Axis::Raw, Axis::Dict, Axis::Sparse, Axis::Quant, Axis::Rank];

impl Axis {
    fn code(self) -> u8 {
        self as u8
    }
    fn from_code(code: u8) -> Result<Self> {
        let axis = AXES.get(usize::from(code)).copied();
        axis.ok_or_else(|| bad(&format!("未知轴码 {code}")))
    }
}

/// 通过逐位验证的最小候选。
#[derive(Clone, Debug)]
pub struct Pick {
    /// 胜出的轴。
    pub axis: Axis,
    /// 完整自描述 artifact。
    pub artifact: Vec<u8>,
}

impl Pick {
    /// artifact 字节数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.artifact.len()
    }

    /// 是否为空（artifact 不会空，仅为 clippy 完整性）。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.artifact.is_empty()
    }
}

/// 五轴搜索：返回逐位验证通过且字节最小的候选；形状不合法返回错误。
pub fn search(w: &[f32], m: usize, n: usize) -> Result<Pick> {
    check_shape(w, m, n)?;
    let base = frame(Axis::Raw, m, n, pack_bytes(&f32_bytes(w)));
    // 只和基线比长度：能胜出的候选必然也已短于基线，与逐个替换的旧口径等价。
    let smaller = [Axis::Dict, Axis::Sparse, Axis::Quant, Axis::Rank]
        .into_iter()
        .filter_map(|axis| candidate(axis, w, m, n).map(|payload| frame(axis, m, n, payload)))
        .filter(|pick| verified(pick, w) && pick.len() < base.len())
        .min_by_key(Pick::len);
    let best = smaller.unwrap_or(base);
    // 基线必须自身逐位成立；这是压缩器的不变式，破了就是实现 bug，因此报错而不是 panic。
    verified(&best, w)
        .then_some(best)
        .ok_or_else(|| bad("raw 基线没能逐位还原，压缩器自身不自洽"))
}

/// 形状判据：`m`、`n` 非零且元素数与乘积一致；不回答权重内容。
fn check_shape(w: &[f32], m: usize, n: usize) -> Result<()> {
    if m == 0 || n == 0 || w.len() != m * n {
        return Err(bad(&format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    Ok(())
}

/// 解码 artifact 为 f32 权重；任何截断/未知轴/长度不符显式报错。
pub fn decode(artifact: &[u8]) -> Result<Vec<f32>> {
    let (axis, want) = head(artifact)?;
    let mut cur = Cursor::new(&artifact[HEAD..]);
    let out = decode_body(axis, &mut cur, want)?;
    if out.len() != want {
        return Err(bad("解码长度与形状不符"));
    }
    Ok(out)
}

/// 头判据：长度至少 `HEAD`、魔数匹配；返回轴与形状（乘积溢出即报错，不回绕）。
fn head(artifact: &[u8]) -> Result<(Axis, usize)> {
    if artifact.len() < HEAD || artifact[..4] != MAGIC {
        return Err(bad("artifact 头无效"));
    }
    let axis = Axis::from_code(artifact[4])?;
    let m = u32::from_le_bytes([artifact[5], artifact[6], artifact[7], artifact[8]]) as usize;
    let n = u32::from_le_bytes([artifact[9], artifact[10], artifact[11], artifact[12]]) as usize;
    let want = m.checked_mul(n).ok_or_else(shape_overflow)?;
    Ok((axis, want))
}

/// 形状乘积溢出。
fn shape_overflow() -> Error {
    bad("形状溢出")
}

/// 按轴分派到各解码器，与 [`candidate`] 一一对应。
fn decode_body(axis: Axis, cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    match axis {
        Axis::Raw => decode_raw(cur, want),
        Axis::Dict => decode_dict(cur, want),
        Axis::Sparse => decode_sparse(cur, want),
        Axis::Quant => decode_quant(cur, want),
        Axis::Rank => decode_rank(cur, want),
    }
}

/// 生成四轴载荷（`raw` 由基线直接给出）；`None` 表示该轴对本输入不适用。
fn candidate(axis: Axis, w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    match axis {
        Axis::Raw => Some(pack_bytes(&f32_bytes(w))),
        Axis::Dict => dict_payload(w),
        Axis::Sparse => sparse_payload(w),
        Axis::Quant => quant_payload(w),
        Axis::Rank => rank_payload(w, m, n),
    }
}

fn frame(axis: Axis, m: usize, n: usize, payload: Vec<u8>) -> Pick {
    let mut artifact = Vec::with_capacity(HEAD + payload.len());
    artifact.extend_from_slice(&MAGIC);
    artifact.push(axis.code());
    artifact.extend_from_slice(&(m as u32).to_le_bytes());
    artifact.extend_from_slice(&(n as u32).to_le_bytes());
    artifact.extend(payload);
    Pick { axis, artifact }
}

fn verified(pick: &Pick, w: &[f32]) -> bool {
    decode(&pick.artifact).is_ok_and(|got| {
        got.len() == w.len() && got.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits())
    })
}

/// 标记位：0 = store（原样），1 = rans 帧。
const STORE_MARK: u8 = 0;
const RANS_MARK: u8 = 1;

fn pack_bytes(src: &[u8]) -> Vec<u8> {
    let frame = rans::encode(src);
    // 严格变小才用 rans，否则 store 原样（含一个标记字节，小数据不放大）。
    let (mark, body) = if frame.len() < src.len() {
        (RANS_MARK, frame)
    } else {
        (STORE_MARK, src.to_vec())
    };
    let mut out = vec![mark];
    out.extend(body);
    out
}

/// 熵帧输出上限：每元素至多 8 字节外加 16 字节头；不让压缩帧自报的长度决定分配大小。
fn payload_cap(want: usize) -> usize {
    want.saturating_mul(8).saturating_add(16)
}

/// 带标记的熵帧：标记字节 + 载荷；store 原样返回，rans 解压。
fn unpack_body(mark: u8, body: &[u8], cap: usize) -> Result<Vec<u8>> {
    match mark {
        STORE_MARK => Ok(body.to_vec()),
        RANS_MARK => rans::decode(body, cap),
        other => Err(bad(&format!("未知熵编码标记 {other}"))),
    }
}

fn unpack_bytes(cur: &mut Cursor<'_>, cap: usize) -> Result<Vec<u8>> {
    let mark = cur.read_u8()?;
    unpack_body(mark, cur.remaining(), cap)
}

/// 对一段带标记的完整帧解码（用于 rank 里已知边界的 low/rest）。
fn unpack_marked(frame: &[u8], cap: usize) -> Result<Vec<u8>> {
    let (&mark, body) = frame.split_first().ok_or_else(|| bad("空熵帧"))?;
    unpack_body(mark, body, cap)
}

fn decode_raw(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur, payload_cap(want))?;
    bytes_to_f32(&bytes, want)
}

/// 唯一值（按首次出现排序）+ 等长索引序列：先建表，再按定宽写出。
fn dict_payload(w: &[f32]) -> Option<Vec<u8>> {
    let mut map = std::collections::HashMap::new();
    let mut vals: Vec<u32> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(w.len());
    for v in w {
        let id = *map.entry(v.to_bits()).or_insert_with(|| {
            vals.push(v.to_bits());
            vals.len() - 1
        });
        ids.push(id as u32);
    }
    let width = index_width(vals.len());
    let mut p = Vec::new();
    p.extend_from_slice(&(width as u32).to_le_bytes());
    p.extend_from_slice(&(vals.len() as u32).to_le_bytes());
    for bits in &vals {
        p.extend_from_slice(&bits.to_le_bytes());
    }
    // 定宽小端：取 `u32` 的低 `width` 字节，与 `as u8`/`as u16` 的截断逐字节一致。
    for id in &ids {
        p.extend_from_slice(&id.to_le_bytes()[..width]);
    }
    Some(pack_bytes(&p))
}

/// 索引宽度：`256`/`65536` 个值以内用 1/2 字节，其余 4 字节。
fn index_width(count: usize) -> usize {
    match count {
        n if n <= 256 => 1,
        n if n <= 65536 => 2,
        _ => 4,
    }
}

fn decode_dict(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur, payload_cap(want))?;
    let mut at = 0;
    let (width, vals) = dict_table(&bytes, &mut at)?;
    (0..want)
        .map(|_| {
            let id = read_index(&bytes, &mut at, width)?;
            vals.get(id).copied().ok_or_else(|| bad("dict 索引越界"))
        })
        .collect()
}

/// dict 头：索引宽度与值表；`at` 停在索引区起点。
fn dict_table(bytes: &[u8], at: &mut usize) -> Result<(usize, Vec<f32>)> {
    let width = word(bytes, at, "字节截断")? as usize;
    let count = word(bytes, at, "字节截断")? as usize;
    let vals = (0..count).map(|_| word(bytes, at, "字节截断").map(f32::from_bits));
    Ok((width, vals.collect::<Result<_>>()?))
}

/// 定宽索引（1/2/4 字节小端）；宽度只可能取自 `dict_payload` 写出的三种。
fn read_index(bytes: &[u8], at: &mut usize, width: usize) -> Result<usize> {
    match width {
        1 => take::<1>(bytes, at, "字节截断").map(|b| usize::from(b[0])),
        2 => take::<2>(bytes, at, "字节截断").map(|b| usize::from(u16::from_le_bytes(b))),
        _ => take::<4>(bytes, at, "字节截断").map(|b| u32::from_le_bytes(b) as usize),
    }
}

fn sparse_payload(w: &[f32]) -> Option<Vec<u8>> {
    let pos: Vec<u32> = w
        .iter()
        .enumerate()
        .filter(|(_, v)| v.to_bits() != 0)
        .map(|(i, _)| i as u32)
        .collect();
    if pos.len() == w.len() {
        return None; // 全非零，稀疏无收益。
    }
    let mut p = Vec::new();
    p.extend_from_slice(&(w.len() as u32).to_le_bytes());
    p.extend_from_slice(&(pos.len() as u32).to_le_bytes());
    for &i in &pos {
        p.extend_from_slice(&i.to_le_bytes());
        p.extend_from_slice(&w[i as usize].to_bits().to_le_bytes());
    }
    Some(pack_bytes(&p))
}

fn decode_sparse(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur, payload_cap(want))?;
    let mut at = 0;
    let (total, nnz) = sparse_head(&bytes, &mut at, want)?;
    let mut out = vec![0.0f32; total];
    (0..nnz).try_for_each(|_| sparse_put(&mut out, &bytes, &mut at))?;
    Ok(out)
}

/// sparse 头：总长必须等于形状给出的元素数；返回总长与 nnz。
fn sparse_head(bytes: &[u8], at: &mut usize, want: usize) -> Result<(usize, usize)> {
    let total = word(bytes, at, "sparse 头截断")? as usize;
    let nnz = word(bytes, at, "sparse 头截断")? as usize;
    if total != want {
        return Err(bad("sparse 长度不符"));
    }
    Ok((total, nnz))
}

/// 写一项 (位置, 值)；位置越界报错。
fn sparse_put(out: &mut [f32], bytes: &[u8], at: &mut usize) -> Result<()> {
    let pos = word(bytes, at, "sparse 项截断")? as usize;
    let val = word(bytes, at, "sparse 项截断").map(f32::from_bits)?;
    *out.get_mut(pos).ok_or_else(|| bad("sparse 位置越界"))? = val;
    Ok(())
}

fn quant_payload(w: &[f32]) -> Option<Vec<u8>> {
    if w.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let min = w.iter().map(|v| v.to_bits()).min().map(f32::from_bits)?;
    (-30i32..=30).find_map(|e| {
        let step = 2.0f32.powi(e);
        let (codes, max) = fit(w, min, step)?;
        // `max == 0` 时 `leading_zeros` 为 32，宽度自然是 0。
        let width = 32 - max.leading_zeros();
        let mut p = min.to_le_bytes().to_vec();
        p.push(e as i8 as u8);
        p.push(width as u8);
        p.extend(pack_bits(&codes, width as u8));
        Some(pack_bytes(&p))
    })
}

/// 在一档步长上试量化：任一点落在 `0..=255` 之外或不能逐位还原，就判该档不可用。
fn fit(w: &[f32], min: f32, step: f32) -> Option<(Vec<u32>, u32)> {
    let mut codes = Vec::with_capacity(w.len());
    let mut max = 0u32;
    for &v in w {
        let q = (v - min) / step;
        if !(0.0..=255.0).contains(&q) {
            return None;
        }
        let c = q.round();
        if (min + c * step).to_bits() != v.to_bits() {
            return None;
        }
        max = max.max(c as u32);
        codes.push(c as u32);
    }
    Some((codes, max))
}

fn decode_quant(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur, payload_cap(want))?;
    let mut at = 0;
    let min = word(&bytes, &mut at, "quant 头截断").map(f32::from_bits)?;
    let [exponent, width] = take::<2>(&bytes, &mut at, "quant 头截断")?;
    if width > 8 {
        return Err(bad("quant 位宽超过 8"));
    }
    let codes = unpack_bits(&bytes[at..], want, width);
    let step = 2.0f32.powi(exponent as i8 as i32);
    Ok(codes.iter().map(|&c| min + c as f32 * step).collect())
}

fn rank_payload(w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    let r = m.min(n) / 2;
    if r == 0 {
        return None;
    }
    let t = ledger::truncate(w, m, n, r).ok()?;
    let low = pack_bytes(&f32_bytes(&t.low));
    let mut p = (r as u32).to_le_bytes().to_vec();
    p.extend_from_slice(&(low.len() as u32).to_le_bytes());
    p.extend(low);
    p.extend(pack_bytes(&f32_bytes(&t.rest)));
    Some(p)
}

fn decode_rank(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let all = cur.remaining();
    let low_len = rank_head(all)?;
    let (low, rest) = rank_frames(all, low_len)?;
    let cap = want.saturating_mul(4);
    let low = rank_part(low, cap, want)?;
    let rest = rank_part(rest, cap, want)?;
    Ok(low.iter().zip(&rest).map(|(a, b)| a + b).collect())
}

/// rank 头判据：秩不能为 0；返回 `low` 段的字节长度。
fn rank_head(all: &[u8]) -> Result<usize> {
    let mut at = 0;
    if word(all, &mut at, "rank 头截断")? == 0 {
        return Err(bad("rank r=0"));
    }
    word(all, &mut at, "rank 头截断").map(|len| len as usize)
}

/// 低秩两段：`low` 由头部的长度界定，`rest` 一直取到末尾。
fn rank_frames(all: &[u8], low_len: usize) -> Result<(&[u8], &[u8])> {
    let low = all
        .get(8..8 + low_len)
        .ok_or_else(|| bad("rank low 截断"))?;
    let rest = all
        .get(8 + low_len..)
        .ok_or_else(|| bad("rank rest 截断"))?;
    Ok((low, rest))
}

/// 解一段带标记的低秩分量。
fn rank_part(frame: &[u8], cap: usize, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_marked(frame, cap)?;
    bytes_to_f32(&bytes, want)
}

/// 从 `buf` 的 `at` 处取 `N` 字节并推进偏移；越界用 `missing` 文案报错。
fn take<const N: usize>(buf: &[u8], at: &mut usize, missing: &str) -> Result<[u8; N]> {
    let chunk = buf.get(*at..).and_then(<[u8]>::first_chunk::<N>);
    let chunk = chunk.ok_or_else(|| bad(missing))?;
    *at += N;
    Ok(*chunk)
}

/// 读一个小端 `u32`。
fn word(buf: &[u8], at: &mut usize, missing: &str) -> Result<u32> {
    take::<4>(buf, at, missing).map(u32::from_le_bytes)
}

fn f32_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8], want: usize) -> Result<Vec<f32>> {
    if bytes.len() != want * 4 {
        return Err(bad(&format!(
            "f32 字节 {} != 期望 {}",
            bytes.len(),
            want * 4
        )));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

fn pack_bits(codes: &[u32], width: u8) -> Vec<u8> {
    if width == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((codes.len() * width as usize).div_ceil(8));
    let mut acc = 0u64;
    let mut bits = 0u32;
    for &c in codes {
        acc |= (c as u64) << bits;
        bits += width as u32;
        while bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push(acc as u8);
    }
    out
}

fn unpack_bits(src: &[u8], count: usize, width: u8) -> Vec<u32> {
    if width == 0 {
        return vec![0; count];
    }
    let mut out = Vec::with_capacity(count);
    let mut acc = 0u64;
    let mut bits = 0u32;
    let mut at = 0usize;
    for _ in 0..count {
        while bits < width as u32 {
            acc |= (src.get(at).copied().unwrap_or(0) as u64) << bits;
            at += 1;
            bits += 8;
        }
        out.push((acc & ((1u64 << width) - 1)) as u32);
        acc >>= width;
        bits -= width as u32;
    }
    out
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("axes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_baseline_roundtrips() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 1.3).collect();
        let pick = search(&w, 8, 8).expect("搜索");
        assert_eq!(pick.axis, Axis::Raw);
        // 不可压缩数据走 store：只增加固定头与一个标记字节，有界常量，不被 rans 头部放大。
        assert!(pick.len() <= w.len() * 4 + HEAD + 1);
        // 核心：逐位往返。
        let got = decode(&pick.artifact).expect("解码");
        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn dict_wins_on_low_cardinality() {
        let vals = [0.1f32, 0.3, 0.7, -0.2];
        let w: Vec<f32> = (0..4096).map(|i| vals[i % vals.len()]).collect();
        let pick = search(&w, 64, 64).expect("搜索");
        assert_eq!(pick.axis, Axis::Dict);
        assert!(pick.len() < w.len() * 4);
    }

    #[test]
    fn sparse_roundtrips_zero_heavy() {
        let mut w = vec![0.0f32; 4096];
        for i in (0..4096).step_by(97) {
            w[i] = (i as f32).sqrt();
        }
        let pick = search(&w, 64, 64).expect("搜索");
        let got = decode(&pick.artifact).expect("解码");
        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn quant_wins_on_exact_grid() {
        let w: Vec<f32> = (0..4096)
            .map(|i| {
                let code = ((i as u32).wrapping_mul(2654435761) >> 12) & 255;
                1.0 + code as f32 * 0.25
            })
            .collect();
        let pick = search(&w, 64, 64).expect("搜索");
        assert_eq!(pick.axis, Axis::Quant);
    }

    #[test]
    fn rank_inexact_is_dropped_not_faked() {
        let w: Vec<f32> = (0..64)
            .map(|i| ((i as f32 * 1.7).cos() * 0.9) + 0.01)
            .collect();
        let pick = search(&w, 8, 8).expect("搜索");
        // 不逐位精确的低秩不得胜出；raw 兜底。
        assert_ne!(pick.axis, Axis::Rank);
    }

    #[test]
    fn malformed_artifacts_are_errors() {
        assert!(decode(b"").is_err());
        assert!(decode(b"AX01").is_err());
        assert!(decode(b"BAD!000000000").is_err());
    }

    #[test]
    fn bad_shape_is_an_error() {
        assert!(search(&[1.0], 3, 1).is_err());
    }
}
