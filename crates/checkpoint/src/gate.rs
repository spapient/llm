//! `gate`: rANS（range Asymmetric Numeral Systems）每字节平面熵编码。
//!
//! `.u` 格式的唯一压缩 codec（`CODECGATE`）。对 bit-shuffle + delta 变换后的
//! 字节流按平面独立做静态 rANS 熵编码，逼近信息熵下界。纯 safe Rust、零
//! unsafe、零第三方依赖。
//!
//! 关键参数：归一化因子 `M = 2^16`、状态下界 `L = 2^23`、32 位状态、8 位
//! renorm。每平面独立建 256 符号频率表并编码，避免把分布迥异的字节平面混在
//! 一起（整体编码会损失约 5pp，见设计记录）。

use crate::error::checkpointerror;

const NSYMBOLS: usize = 256;
const LOG2_M: u32 = 16;
const M: u32 = 1 << LOG2_M;
const L: u32 = 1 << 23;

/// 归一化 256 符号频率到总和精确为 `M`（出现符号频率 ≥1，未出现为 0）。
fn normalize(counts: &[u32; NSYMBOLS]) -> [u32; NSYMBOLS] {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut freq = [0u32; NSYMBOLS];
    if total == 0 {
        freq[0] = M;
        return freq;
    }
    let mut sum = 0i64;
    let mut max_s = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            let f = ((c as u64 * M as u64 + total / 2) / total).max(1) as u32;
            freq[s] = f;
            sum += f as i64;
            if f > freq[max_s] {
                max_s = s;
            }
        }
    }
    let mut diff = M as i64 - sum;
    while diff > 0 {
        freq[max_s] += 1;
        diff -= 1;
    }
    while diff < 0 && freq[max_s] > 1 {
        freq[max_s] -= 1;
        diff += 1;
    }
    if diff < 0 {
        // 兜底（理论上不可达）：把 M 均匀分给出现符号。
        let present: Vec<usize> = (0..NSYMBOLS).filter(|&s| freq[s] > 0).collect();
        let n = present.len().max(1);
        let base = M as usize / n;
        let rem = M as usize % n;
        for (i, &s) in present.iter().enumerate() {
            freq[s] = (base + usize::from(i < rem)) as u32;
        }
    }
    freq
}

/// 一个平面的频率表与累积频率表。
struct Table {
    freq: [u32; NSYMBOLS],
    cum: [u32; NSYMBOLS + 1],
}

impl Table {
    fn build(counts: &[u32; NSYMBOLS]) -> Self {
        let freq = normalize(counts);
        let mut cum = [0u32; NSYMBOLS + 1];
        let mut acc = 0u32;
        for s in 0..NSYMBOLS {
            cum[s] = acc;
            acc += freq[s];
        }
        cum[NSYMBOLS] = acc;
        Self { freq, cum }
    }

    /// 由 `slot ∈ [0, M)` 反查符号（二分）。
    fn find_symbol(&self, slot: u32) -> usize {
        let mut lo = 0usize;
        let mut hi = NSYMBOLS;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.cum[mid] <= slot {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// 编码一个平面，把 rANS 流追加到 `out`。
///
/// 字节序与解码对称：先 flush 最终状态 `x` 的 4 字节（little-endian），再按
/// 逆序输出 renorm 期间积累的字节（先高字节后低字节），使解码按序读入。
fn encode_plane(out: &mut Vec<u8>, data: &[u8], table: &Table) {
    let mut x: u32 = L;
    let mut renorm = Vec::new();
    for &byte in data {
        let s = byte as usize;
        let f = table.freq[s];
        let c = table.cum[s];
        let x_max = ((L >> LOG2_M) << 8) * f;
        while x >= x_max {
            renorm.push((x & 0xFF) as u8);
            x >>= 8;
        }
        x = (x / f) * M + (x % f) + c;
    }
    for shift in [0, 8, 16, 24] {
        out.push(((x >> shift) & 0xFF) as u8);
    }
    for &b in renorm.iter().rev() {
        out.push(b);
    }
}

fn invalid(reason: &str) -> checkpointerror {
    checkpointerror::invalidmanifest {
        reason: reason.to_string(),
    }
}

fn take_u32(data: &mut &[u8]) -> Option<u32> {
    if data.len() < 4 {
        return None;
    }
    let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    *data = &data[4..];
    Some(v)
}

/// 解码一个平面，从 `stream` 读出 `n` 字节追加到 `out`。
///
/// rANS 是栈式编码：解码按编码的逆序弹出符号，故先在局部缓冲收集、最后
/// reverse 后写入 `out`，恢复原始顺序。
fn decode_plane(
    stream: &[u8],
    out: &mut Vec<u8>,
    table: &Table,
    n: usize,
) -> Result<(), checkpointerror> {
    let mut data = stream;
    let mut x = take_u32(&mut data).ok_or_else(|| invalid("gate stream truncated"))?;
    let mut buf = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = x & (M - 1);
        let s = table.find_symbol(slot);
        buf.push(s as u8);
        x = table.freq[s] * (x >> LOG2_M) + slot - table.cum[s];
        while x < L {
            if data.is_empty() {
                return Err(invalid("gate stream truncated"));
            }
            x = (x << 8) | u32::from(data[0]);
            data = &data[1..];
        }
    }
    out.extend(buf.iter().rev());
    Ok(())
}

/// 压缩 `src`，`elem` 为平面数（0 = 整体单平面，2 = f16，4 = f32）。
///
/// 格式：`[u32 src_len][elem 平面 × 256×u32 频率表][elem × u32 流长]
/// [elem 段 rANS 流][tail 原样]`。`elem = 0` 时把整个 `src` 作为一个平面、
/// 无 tail。
pub fn compress(src: &[u8], elem: usize) -> Vec<u8> {
    let nplanes = if elem == 0 { 1 } else { elem };
    let n = src.len().checked_div(elem).unwrap_or(src.len());
    let tail = if elem == 0 { &src[..0] } else { &src[elem * n..] };

    let mut tables = Vec::with_capacity(nplanes);
    for p in 0..nplanes {
        let plane = &src[p * n..(p + 1) * n];
        let mut counts = [0u32; NSYMBOLS];
        for &b in plane {
            counts[b as usize] += 1;
        }
        tables.push(Table::build(&counts));
    }

    let mut streams = Vec::with_capacity(nplanes);
    for (p, table) in tables.iter().enumerate() {
        let plane = &src[p * n..(p + 1) * n];
        let mut stream = Vec::new();
        encode_plane(&mut stream, plane, table);
        streams.push(stream);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&(src.len() as u32).to_le_bytes());
    for table in &tables {
        for &f in &table.freq {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    for stream in &streams {
        out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    }
    for stream in &streams {
        out.extend_from_slice(stream);
    }
    out.extend_from_slice(tail);
    out
}

/// 解压 `stored`（`compress` 的输出），`elem` 须与压缩时一致。
pub fn decompress(stored: &[u8], elem: usize) -> Result<Vec<u8>, checkpointerror> {
    let mut data = stored;
    let src_len = take_u32(&mut data).ok_or_else(|| invalid("gate header truncated"))? as usize;
    let nplanes = if elem == 0 { 1 } else { elem };
    let n = src_len.checked_div(elem).unwrap_or(src_len);
    let tail_len = src_len - nplanes * n;

    let mut tables = Vec::with_capacity(nplanes);
    for _ in 0..nplanes {
        let mut freq = [0u32; NSYMBOLS];
        for f in &mut freq {
            *f = take_u32(&mut data).ok_or_else(|| invalid("gate freq table truncated"))?;
        }
        let mut cum = [0u32; NSYMBOLS + 1];
        let mut acc = 0u32;
        for s in 0..NSYMBOLS {
            cum[s] = acc;
            acc += freq[s];
        }
        cum[NSYMBOLS] = acc;
        tables.push(Table { freq, cum });
    }

    let mut lens = Vec::with_capacity(nplanes);
    for _ in 0..nplanes {
        lens.push(take_u32(&mut data).ok_or_else(|| invalid("gate stream length truncated"))? as usize);
    }

    let mut out = Vec::with_capacity(src_len);
    for p in 0..nplanes {
        if data.len() < lens[p] {
            return Err(invalid("gate stream truncated"));
        }
        decode_plane(&data[..lens[p]], &mut out, &tables[p], n)?;
        data = &data[lens[p]..];
    }
    if data.len() < tail_len {
        return Err(invalid("gate tail truncated"));
    }
    out.extend_from_slice(&data[..tail_len]);
    Ok(out)
}
