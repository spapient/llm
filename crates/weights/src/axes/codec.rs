//! `axes` 的编解码机制：熵帧、字典/稀疏/量化三种载荷的读写，以及定宽取字节。
//!
//! 与 [`crate::axes`] 分家只为把公开面（轴闭集、搜索、验证）与字节细节分开：这里每个函数都
//! 只被 `axes` 的 `AxisCfg` 实现调用，入口不在此处。

use diagnostic::Result;

use crate::bad;
use crate::rans;
use wire::leb::Cursor;

/// 标记位：0 = store（原样），1 = rans 帧。
const STORE_MARK: u8 = 0;
const RANS_MARK: u8 = 1;

pub(crate) fn pack_bytes(src: &[u8]) -> Vec<u8> {
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
pub(crate) fn payload_cap(want: usize) -> usize {
    want.saturating_mul(8).saturating_add(16)
}

/// 带标记的熵帧：标记字节 + 载荷；store 原样返回，rans 解压。
fn unpack_body(mark: u8, body: &[u8], cap: usize) -> Result<Vec<u8>> {
    match mark {
        STORE_MARK => Ok(body.to_vec()),
        RANS_MARK => rans::decode(body, cap),
        other => Err(bad(format!("未知熵编码标记 {other}"))),
    }
}

pub(crate) fn unpack_bytes(cur: &mut Cursor<'_>, cap: usize) -> Result<Vec<u8>> {
    let mark = cur.read_u8()?;
    unpack_body(mark, cur.remaining(), cap)
}

/// 对一段带标记的完整帧解码（用于 rank 里已知边界的 low/rest）。
pub(crate) fn unpack_marked(frame: &[u8], cap: usize) -> Result<Vec<u8>> {
    let (&mark, body) = frame.split_first().ok_or_else(|| bad("空熵帧"))?;
    unpack_body(mark, body, cap)
}

/// 唯一值排序表 + 等长索引序列：排序去重后二分定位，零关联容器依赖。
pub(crate) fn dict_payload(w: &[f32]) -> Option<Vec<u8>> {
    let mut vals: Vec<u32> = w.iter().map(|v| v.to_bits()).collect();
    vals.sort_unstable();
    vals.dedup();
    if vals.len() == w.len() {
        return None; // 全唯一，字典表 + 全宽索引必然大于 raw，无收益。
    }
    let width = index_width(vals.len());
    let mut p = Vec::new();
    p.extend_from_slice(&(width as u32).to_le_bytes());
    p.extend_from_slice(&(vals.len() as u32).to_le_bytes());
    for bits in &vals {
        p.extend_from_slice(&bits.to_le_bytes());
    }
    // 定宽小端：取 `u32` 的低 `width` 字节，与 `as u8`/`as u16` 的截断逐字节一致。
    for v in w {
        let id = vals.binary_search(&v.to_bits()).ok()?;
        p.extend_from_slice(&(id as u32).to_le_bytes()[..width]);
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

pub(crate) fn decode_dict(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
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

pub(crate) fn sparse_payload(w: &[f32]) -> Option<Vec<u8>> {
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

pub(crate) fn decode_sparse(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
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

pub(crate) fn quant_payload(w: &[f32]) -> Option<Vec<u8>> {
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

pub(crate) fn decode_quant(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
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

pub(crate) fn f32_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(crate) fn bytes_to_f32(bytes: &[u8], want: usize) -> Result<Vec<f32>> {
    if bytes.len() != want * 4 {
        return Err(bad(format!(
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

/// 从 `buf` 的 `at` 处取 `N` 字节并推进偏移；越界用 `missing` 文案报错。
fn take<const N: usize>(buf: &[u8], at: &mut usize, missing: &str) -> Result<[u8; N]> {
    let chunk = buf.get(*at..).and_then(<[u8]>::first_chunk::<N>);
    let chunk = chunk.ok_or_else(|| bad(missing))?;
    *at += N;
    Ok(*chunk)
}

/// 读一个小端 `u32`。
pub(crate) fn word(buf: &[u8], at: &mut usize, missing: &str) -> Result<u32> {
    take::<4>(buf, at, missing).map(u32::from_le_bytes)
}
