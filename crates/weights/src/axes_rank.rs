//! rank 轴编解码：低秩因子 + 残差 + 谱隙证书随载荷落盘与对质。

use diagnostic::Result;

use crate::axes::{area, bad};
use crate::axes_codec::{bytes_to_f32, f32_bytes, pack_bytes, payload_cap, word};
use crate::leb::Cursor;
use crate::ledger;

pub(crate) fn rank_payload((m, n): (usize, usize), w: &[f32]) -> Option<Vec<u8>> {
    let cert = ledger::certify(w, m, n)?;
    let mut p = cert.note.to_bytes();
    p.extend_from_slice(&f64_bytes(&cert.left));
    p.extend_from_slice(&f64_bytes(&cert.right));
    p.extend_from_slice(&f32_bytes(&cert.rest));
    Some(pack_bytes(&p))
}

/// 有证书就压缩；否则按需直出 raw 帧（不再回退到单独证书帧）。
fn framed(m: usize, n: usize, w: &[f32]) -> (bool, Vec<u8>) {
    match ledger::certify(w, m, n) {
        Some(cert) => (true, {
            let mut p = cert.note.to_bytes();
            p.extend_from_slice(&f64_bytes(&cert.left));
            p.extend_from_slice(&f64_bytes(&cert.right));
            p.extend_from_slice(&f32_bytes(&cert.rest));
            pack_bytes(&p)
        }),
        None => (false, pack_bytes(&f32_bytes(w))),
    }
}

pub(crate) fn decode_rank(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
    let cap = payload_cap(want);
    let bytes = unpack_marked(cur, cap)?;
    let (m, n, rank, factors, low) = rank_parts(&bytes, want)?;
    let (mut left, mut right, mut rest, mut note) =
        factors_of(&bytes, rank, factors, m, n, low)?;
    ledger::audit(&mut note, &left, &right, &mut rest)?;
    if note.note != ledger::Note::new() {
        return Err(bad("rank 证书未认证"));
    }
    (0..n)
        .flat_map(|j| (0..m).map(move |i| left[i] * right[j] + rest[i + j * m]))
        .collect()
}

/// rank 头：先导张量维度、秩、因子宽、残差宽（全为零则 `low = 0`）。
fn rank_parts(bytes: &[u8], want: usize) -> Result<(usize, usize, usize, usize, usize)> {
    let mut at = 0;
    let m = word(bytes, &mut at, "rank 头截断")? as usize;
    let n = word(bytes, &mut at, "rank 头截断")? as usize;
    let rank = word(bytes, &mut at, "rank 头截断")? as usize;
    let factors = word(bytes, &mut at, "rank 头截断")? as usize;
    let low = word(bytes, &mut at, "rank 头截断")? as usize;
    if area((m, n)) != want {
        return Err(bad("rank 维度不符"));
    }
    Ok((m, n, rank, factors, low))
}

fn factors_of(
    bytes: &[u8],
    rank: usize,
    factors: usize,
    m: usize,
    n: usize,
    low: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, ledger::Note)> {
    let (L, R) = (rank, factors);
    let size = L * m + R * n;
    let vals = bytes_to_f32(bytes, size)?;
    let mut left = vals[..L * m].to_vec();
    let mut right = vals[L * m..].to_vec();
    let rest = if low == 0 {
        vec![0.0; m * n]
    } else {
        rank_part(bytes, &mut 0, m * n, L * m + R * n, &vals)?
    };
    let note = ledger::Note::decode(&mut Cursor::new(&bytes[..]))?;
    Ok((left, right, rest, note))
}

fn low_of(c: &ledger::Note, rank: usize) -> usize {
    ledger::factor_low(c, rank).unwrap_or(0)
}

struct RankHead {
    m: usize,
    n: usize,
    rank: usize,
    factors: usize,
    note: ledger::Note,
}

pub(crate) fn rank_head(bytes: &[u8]) -> Result<RankHead> {
    let mut at = 0;
    let m = word(bytes, &mut at, "rank 头截断")? as usize;
    let n = word(bytes, &mut at, "rank 头截断")? as usize;
    let rank = word(bytes, &mut at, "rank 头截断")? as usize;
    let factors = word(bytes, &mut at, "rank 头截断")? as usize;
    let note = ledger::Note::decode(&mut Cursor::new(&bytes[..]))?;
    Ok(RankHead {
        m,
        n,
        rank,
        factors,
        note,
    })
}

fn sliced(bytes: &[u8], base: usize, len: usize, want: usize) -> Result<Vec<f32>> {
    let mut at = base;
    let (m, n, rank, factors, _) = {
        let a = word(bytes, &mut at, "rank 头截断")? as usize;
        let b = word(bytes, &mut at, "rank 头截断")? as usize;
        let c = word(bytes, &mut at, "rank 头截断")? as usize;
        let d = word(bytes, &mut at, "rank 头截断")? as usize;
        let e = word(bytes, &mut at, "rank 头截断")? as usize;
        (a, b, c, d, e)
    };
    let (L, R) = (rank, factors);
    let vals = bytes_to_f32(&bytes[at..at + (L * m + R * n) * 4], L * m + R * n)?;
    Ok((0..want)
        .map(|i| {
            let j = i / m;
            let k = i % m;
            vals[k] * vals[L * m + j] + if j < low_of(&ledger::Note::new(), rank) {
                vals[L * m + R * n + i]
            } else {
                0.0
            }
        })
        .collect())
}

fn rank_note(bytes: &[u8], rank: usize) -> Result<ledger::Note> {
    let note = ledger::Note::decode(&mut Cursor::new(&bytes[..]))?;
    if ledger::factor_low(&note, rank).is_none() {
        return Err(bad("rank 头无证书"));
    }
    Ok(note)
}

fn rank_part(
    bytes: &[u8],
    _at: &mut usize,
    want: usize,
    _off: usize,
    _vals: &[f32],
) -> Result<Vec<f32>> {
    // 残差帧从字节起点重读（与头共享前缀），保持“读一帧”语义。
    let cap = payload_cap(want);
    let bytes = unpack_marked(&mut Cursor::new(bytes), cap)?;
    bytes_to_f32(&bytes, want)
}

/// 读一个带标记帧；rank 载荷整体带标记（首字节 0/1），随后是证书与因子。
fn unpack_marked(cur: &mut Cursor<'_>, cap: usize) -> Result<Vec<u8>> {
    let mark = cur.read_u8()?;
    match mark {
        0 => Ok(cur.remaining().to_vec()),
        1 => crate::rans::decode(cur.remaining(), cap),
        other => Err(bad(&format!("未知熵编码标记 {other}"))),
    }
}

/// `f64` 小端字节。
fn f64_bytes(w: &[f64]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 谱隙因子帧：`2 * want` 字节（每因子一个 `f64` 或 `f32`），拆成两部分。
fn factor_part(bytes: &[u8], _cap: usize) -> Vec<u8> {
    let want = bytes.len() / 4;
    let mut out = Vec::with_capacity(want);
    let mut at = 0;
    for _ in 0..want {
        out.extend_from_slice(&f32_bytes(&[word(bytes, &mut at, "factor 项截断")
            .map(f32::from_bits)
            .unwrap_or(0.0)]));
    }
    out
}
