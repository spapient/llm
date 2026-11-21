//! `axes` 的秩轴载荷：证书 + 两个低秩因子 + 残差的落盘与读回，读端自证。

use diagnostic::Result;

use crate::bad;
use crate::ledger;
use wire::leb::Cursor;

use crate::axes::area;
use crate::axes::codec::{bytes_to_f32, f32_bytes, pack_bytes, payload_cap, unpack_marked, word};

/// 低秩载荷：秩由证书给出，证书不通过就不压，证书本身随载荷落盘。
pub(crate) fn rank_payload(w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    let cert = ledger::certify(w, m, n).ok().flatten()?;
    let mut p = framed(&cert.note.to_bytes());
    p.extend(framed(&pack_bytes(&f64_bytes(&cert.left))));
    p.extend(framed(&pack_bytes(&f64_bytes(&cert.right))));
    p.extend(pack_bytes(&f32_bytes(&cert.rest)));
    Some(p)
}

/// 长度前缀 + 本体：两段的边界由长度给出，`rest` 一直取到末尾故不带前缀。
pub(crate) fn framed(body: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(body.len())
        .map(|len| len.to_le_bytes().to_vec())
        .unwrap_or_default();
    out.extend_from_slice(body);
    out
}

pub(crate) fn decode_rank(cur: &mut Cursor<'_>, (m, n): (usize, usize)) -> Result<Vec<f32>> {
    let all = cur.remaining();
    let head = rank_head(all)?;
    let (low, rest) = rank_parts(all, &head, (m, n))?;
    Ok(low.iter().zip(&rest).map(|(a, b)| a + b).collect())
}

/// 解两段因子与残差、还原低秩项并对质证书：`?` 预算因此留在门禁内。
fn rank_parts(all: &[u8], head: &RankHead, (m, n): (usize, usize)) -> Result<(Vec<f32>, Vec<f32>)> {
    let rank = head.note.rank as usize;
    let want = area((m, n));
    let (left, right, at) = factors_of(all, head, (m, n), rank)?;
    let rest = rank_part(&all[at..], payload_cap(want), want)?;
    let low = low_of(&left, &right, (m, n), rank);
    // 证书与它认证的两段对质：这一行才是「可证」——读到的载荷自己能把它重算一遍，
    // 而不是要读者相信写端曾经的声称。
    ledger::audit(&head.note, m, n, &low, &rest)?;
    Ok((low, rest))
}

/// 一对因子：左因子区间由头给出，右因子紧随其后；第三个返回值是残差段起点。
fn factors_of(
    all: &[u8],
    head: &RankHead,
    (m, n): (usize, usize),
    rank: usize,
) -> Result<(Vec<f64>, Vec<f64>, usize)> {
    let left = factor_part(&all[head.left.clone()], m * rank)?;
    let mut at = head.left.end;
    let right = factor_part(&all[sliced(all, &mut at, "rank 右因子")?], rank * n)?;
    Ok((left, right, at))
}

/// 因子 → 低秩项（`f32`）：与写端同一个 [`ledger::factor_low`]，读端不另算一遍。
fn low_of(left: &[f64], right: &[f64], (m, n): (usize, usize), rank: usize) -> Vec<f32> {
    ledger::factor_low(left, right, (m, n), rank)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

/// 秩载荷的头：证书本体与左因子段（右因子长度在它之后读）。
pub(crate) struct RankHead {
    pub(crate) note: ledger::Note,
    /// 左因子段在载荷内的区间。
    left: std::ops::Range<usize>,
}

/// rank 头判据：证书长度前缀、证书本体（必须恰好读完）、左因子长度。
pub(crate) fn rank_head(all: &[u8]) -> Result<RankHead> {
    let mut at = 0;
    let note = sliced(all, &mut at, "rank 证书")?;
    let left = sliced(all, &mut at, "rank 左因子")?;
    Ok(RankHead {
        note: rank_note(all, note.start, note.len())?,
        left,
    })
}

/// 读侧的长度前缀段：先读 `u32` 长度，再按它切出紧随其后的字节区间；越界一律按截断报错。
fn sliced(all: &[u8], at: &mut usize, what: &str) -> Result<std::ops::Range<usize>> {
    let len = word(all, at, what)? as usize;
    let end = at
        .checked_add(len)
        .filter(|&e| e <= all.len())
        .ok_or_else(|| {
            bad(format!(
                "{what} 越界：长 {len}，余 {}",
                all.len().saturating_sub(*at)
            ))
        })?;
    let range = *at..end;
    *at = end;
    Ok(range)
}

/// 解证书本体：切片必须恰好是 `len` 字节，且证书解码必须恰好读完。
fn rank_note(all: &[u8], at: usize, len: usize) -> Result<ledger::Note> {
    let bytes = all
        .get(at..)
        .and_then(|tail| tail.get(..len))
        .ok_or_else(|| bad("rank 证书截断"))?;
    let mut cur = Cursor::new(bytes);
    let note = ledger::Note::decode(&mut cur)?;
    if !cur.remaining().is_empty() {
        return Err(bad("rank 证书有多余字节"));
    }
    Ok(note)
}

/// 解一段带标记的低秩分量。
fn rank_part(frame: &[u8], cap: usize, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_marked(frame, cap)?;
    bytes_to_f32(&bytes, want)
}

/// 因子的字节形式：`f64` 保留分解的原始精度，`f32` 因子会让 `Σ uσv` 舍入两次而丢掉逐位判据。
pub(crate) fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 解一段带标记的因子（`f64`）。
fn factor_part(frame: &[u8], want: usize) -> Result<Vec<f64>> {
    let bytes = unpack_marked(frame, payload_cap(want * 2))?;
    if bytes.len() != want * 8 {
        return Err(bad(format!(
            "因子字节 {} != 期望 {}",
            bytes.len(),
            want * 8
        )));
    }
    let (chunks, _) = bytes.as_chunks::<8>();
    Ok(chunks.iter().map(|c| f64::from_le_bytes(*c)).collect())
}
