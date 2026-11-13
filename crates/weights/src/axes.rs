//! 五轴自适应选码：`raw / dict / sparse / quant / rank` 五个候选，每个候选只有在解码后与原始
//! 位模式逐元素一致（`to_bits` 相等）时才被采纳，最终在通过验证的候选里取字节最小者。
//!
//! `raw` 是恒可用的基线（f32 字节走 rANS），因此非空输入总有返回。`quant` 要求精确落在
//! `2^e` 网格、`rank` 要求低秩+残差能逐位还原；f32 低秩重建不保证逐位一致，不满足即丢弃，
//! 绝不谎称无损。熵编码复用 `crate::rans`，低秩截断复用 [`crate::ledger`]，零第三方依赖。

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

impl Axis {
    fn code(self) -> u8 {
        self as u8
    }
    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Dict),
            2 => Ok(Self::Sparse),
            3 => Ok(Self::Quant),
            4 => Ok(Self::Rank),
            other => Err(bad(&format!("未知轴码 {other}"))),
        }
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
    if m == 0 || n == 0 || w.len() != m * n {
        return Err(bad(&format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    let mut best = frame(Axis::Raw, m, n, raw_payload(w));
    for axis in [Axis::Dict, Axis::Sparse, Axis::Quant, Axis::Rank] {
        let Some(payload) = candidate(axis, w, m, n) else {
            continue;
        };
        let pick = frame(axis, m, n, payload);
        if pick.len() < best.len() && verified(&pick, w) {
            best = pick;
        }
    }
    // 基线必须自身逐位成立。
    assert!(verified(&best, w), "raw 基线必须逐位还原");
    Ok(best)
}

/// 解码 artifact 为 f32 权重；任何截断/未知轴/长度不符显式报错。
pub fn decode(artifact: &[u8]) -> Result<Vec<f32>> {
    if artifact.len() < HEAD || artifact[..4] != MAGIC {
        return Err(bad("artifact 头无效"));
    }
    let axis = Axis::from_code(artifact[4])?;
    let m = u32::from_le_bytes(artifact[5..9].try_into().expect("4")) as usize;
    let n = u32::from_le_bytes(artifact[9..13].try_into().expect("4")) as usize;
    let want = m.checked_mul(n).ok_or_else(|| bad("形状溢出"))?;
    let mut cur = Cursor::new(&artifact[HEAD..]);
    let out = match axis {
        Axis::Raw => decode_raw(&mut cur, want)?,
        Axis::Dict => decode_dict(&mut cur, want)?,
        Axis::Sparse => decode_sparse(&mut cur, want)?,
        Axis::Quant => decode_quant(&mut cur, want)?,
        Axis::Rank => decode_rank(&mut cur, want)?,
    };
    if out.len() != want {
        return Err(bad("解码长度与形状不符"));
    }
    Ok(out)
}

// ------------------------------------------------ 候选分发

fn candidate(axis: Axis, w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    match axis {
        Axis::Raw => Some(raw_payload(w)),
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
    match decode(&pick.artifact) {
        Ok(got) => {
            got.len() == w.len() && got.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        Err(_) => false,
    }
}

// ------------------------------------------------ 熵编码辅助

/// 标记位：0 = store（原样），1 = rans 帧。
const STORE_MARK: u8 = 0;
const RANS_MARK: u8 = 1;

fn pack_bytes(src: &[u8]) -> Vec<u8> {
    let frame = rans::encode(src);
    // 严格变小才用 rans，否则 store 原样（含一个标记字节，小数据不放大）。
    if frame.len() < src.len() {
        let mut out = Vec::with_capacity(1 + frame.len());
        out.push(RANS_MARK);
        out.extend(frame);
        out
    } else {
        let mut out = Vec::with_capacity(1 + src.len());
        out.push(STORE_MARK);
        out.extend_from_slice(src);
        out
    }
}

fn unpack_bytes(cur: &mut Cursor<'_>) -> Result<Vec<u8>> {
    let mark = cur.read_u8()?;
    let rest = cur.remaining();
    match mark {
        STORE_MARK => Ok(rest.to_vec()),
        RANS_MARK => rans::decode(rest),
        other => Err(bad(&format!("未知熵编码标记 {other}"))),
    }
}

/// 对一段带标记的完整帧解码（用于 rank 里已知边界的 low/rest）。
fn unpack_marked(frame: &[u8]) -> Result<Vec<u8>> {
    let mark = *frame.first().ok_or_else(|| bad("空熵帧"))?;
    let body = frame.get(1..).ok_or_else(|| bad("熵帧截断"))?;
    match mark {
        STORE_MARK => Ok(body.to_vec()),
        RANS_MARK => rans::decode(body),
        other => Err(bad(&format!("未知熵编码标记 {other}"))),
    }
}

// ------------------------------------------------ raw

fn raw_payload(w: &[f32]) -> Vec<u8> {
    let bytes = f32_bytes(w);
    pack_bytes(&bytes)
}

fn decode_raw(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur)?;
    bytes_to_f32(&bytes, want)
}

// ------------------------------------------------ dict

fn dict_payload(w: &[f32]) -> Option<Vec<u8>> {
    // 唯一值（按首次出现）与索引。
    let mut map = std::collections::HashMap::new();
    let mut vals: Vec<u32> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(w.len());
    for v in w {
        let bits = v.to_bits();
        let id = *map.entry(bits).or_insert_with(|| {
            vals.push(bits);
            vals.len() - 1
        });
        ids.push(id as u32);
    }
    let index_width = match vals.len() {
        n if n <= 256 => 1usize,
        n if n <= 65536 => 2,
        _ => 4,
    };
    let mut p = Vec::new();
    p.extend_from_slice(&(index_width as u32).to_le_bytes());
    p.extend_from_slice(&(vals.len() as u32).to_le_bytes());
    for bits in &vals {
        p.extend_from_slice(&bits.to_le_bytes());
    }
    for id in &ids {
        match index_width {
            1 => p.push(*id as u8),
            2 => p.extend_from_slice(&(*id as u16).to_le_bytes()),
            _ => p.extend_from_slice(&id.to_le_bytes()),
        }
    }
    Some(pack_bytes(&p))
}

fn decode_dict(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur)?;
    let mut at = 0usize;
    let width = u32::from_le_bytes(grab(&bytes, &mut at, 4)?.try_into().expect("4")) as usize;
    let count = u32::from_le_bytes(grab(&bytes, &mut at, 4)?.try_into().expect("4")) as usize;
    let mut vals = Vec::with_capacity(count);
    for _ in 0..count {
        let word = grab(&bytes, &mut at, 4)?;
        vals.push(f32::from_le_bytes(word.try_into().expect("4")));
    }
    let mut out = Vec::with_capacity(want);
    for _ in 0..want {
        let id = match width {
            1 => *grab(&bytes, &mut at, 1)?.first().expect("1") as usize,
            2 => u16::from_le_bytes(grab(&bytes, &mut at, 2)?.try_into().expect("2")) as usize,
            _ => u32::from_le_bytes(grab(&bytes, &mut at, 4)?.try_into().expect("4")) as usize,
        };
        out.push(*vals.get(id).ok_or_else(|| bad("dict 索引越界"))?);
    }
    Ok(out)
}

/// 从 `buf` 的 `at` 处取 `n` 字节并推进偏移；越界报错。
fn grab<'b>(buf: &'b [u8], at: &mut usize, n: usize) -> Result<&'b [u8]> {
    let slice = buf.get(*at..*at + n).ok_or_else(|| bad("字节截断"))?;
    *at += n;
    Ok(slice)
}

// ------------------------------------------------ sparse

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
    let bytes = unpack_bytes(cur)?;
    if bytes.len() < 8 {
        return Err(bad("sparse 头截断"));
    }
    let total = u32::from_le_bytes(bytes[0..4].try_into().expect("4")) as usize;
    let nnz = u32::from_le_bytes(bytes[4..8].try_into().expect("4")) as usize;
    if total != want {
        return Err(bad("sparse 长度不符"));
    }
    let mut out = vec![0.0f32; total];
    let mut at = 8usize;
    for _ in 0..nnz {
        let entry = bytes.get(at..at + 8).ok_or_else(|| bad("sparse 项截断"))?;
        let pos = u32::from_le_bytes(entry[0..4].try_into().expect("4")) as usize;
        let val = f32::from_le_bytes(entry[4..8].try_into().expect("4"));
        *out.get_mut(pos).ok_or_else(|| bad("sparse 位置越界"))? = val;
        at += 8;
    }
    Ok(out)
}

// ------------------------------------------------ quant（精确网格）

fn quant_payload(w: &[f32]) -> Option<Vec<u8>> {
    if w.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let min = w.iter().map(|v| v.to_bits()).min().map(f32::from_bits)?;
    (-30i32..=30).find_map(|e| {
        let step = 2.0f32.powi(e);
        let mut codes = Vec::with_capacity(w.len());
        let mut max = 0u32;
        for &v in w {
            let q = (v - min) / step;
            if !(0.0..=255.0).contains(&q) {
                return None;
            }
            let c = q.round();
            // 逐位可还原才接受。
            if (min + c * step).to_bits() != v.to_bits() {
                return None;
            }
            max = max.max(c as u32);
            codes.push(c as u32);
        }
        let width = if max == 0 {
            0
        } else {
            32 - max.leading_zeros()
        };
        let mut p = Vec::new();
        p.extend_from_slice(&min.to_le_bytes());
        p.push(e as i8 as u8);
        p.push(width as u8);
        p.extend(pack_bits(&codes, width as u8));
        Some(pack_bytes(&p))
    })
}

fn decode_quant(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_bytes(cur)?;
    if bytes.len() < 6 {
        return Err(bad("quant 头截断"));
    }
    let min = f32::from_le_bytes(bytes[0..4].try_into().expect("4"));
    let e = bytes[4] as i8 as i32;
    let width = bytes[5];
    if width > 8 {
        return Err(bad("quant 位宽超过 8"));
    }
    let codes = unpack_bits(&bytes[6..], want, width);
    let step = 2.0f32.powi(e);
    Ok(codes.iter().map(|&c| min + c as f32 * step).collect())
}

// ------------------------------------------------ rank（精确低秩+残差）

fn rank_payload(w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    let r = m.min(n) / 2;
    if r == 0 {
        return None;
    }
    let t = ledger::truncate(w, m, n, r).ok()?;
    let mut p = Vec::new();
    p.extend_from_slice(&(r as u32).to_le_bytes());
    let low = pack_bytes(&f32_bytes(&t.low));
    let rest = pack_bytes(&f32_bytes(&t.rest));
    p.extend_from_slice(&(low.len() as u32).to_le_bytes());
    p.extend(low);
    p.extend(rest);
    Some(p)
}

fn decode_rank(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let all = cur.remaining();
    if all.len() < 8 {
        return Err(bad("rank 头截断"));
    }
    let r = u32::from_le_bytes(all[0..4].try_into().expect("4"));
    if r == 0 {
        return Err(bad("rank r=0"));
    }
    let low_len = u32::from_le_bytes(all[4..8].try_into().expect("4")) as usize;
    let low_bytes = unpack_marked(
        all.get(8..8 + low_len)
            .ok_or_else(|| bad("rank low 截断"))?,
    )?;
    let low = bytes_to_f32(&low_bytes, want)?;
    let rest_start = 8 + low_len;
    let rest_bytes = unpack_marked(all.get(rest_start..).ok_or_else(|| bad("rank rest 截断"))?)?;
    let rest = bytes_to_f32(&rest_bytes, want)?;
    Ok(low.iter().zip(&rest).map(|(a, b)| a + b).collect())
}

// ------------------------------------------------ 通用位/字节辅助

fn f32_bytes(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() * 4);
    for v in w {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32(bytes: &[u8], want: usize) -> Result<Vec<f32>> {
    if bytes.len() != want * 4 {
        return Err(bad(&format!(
            "f32 字节 {} != 期望 {}",
            bytes.len(),
            want * 4
        )));
    }
    let (chunks, tail) = bytes.as_chunks::<4>();
    if !tail.is_empty() {
        return Err(bad("f32 尾部不是 4 字节整数倍"));
    }
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
