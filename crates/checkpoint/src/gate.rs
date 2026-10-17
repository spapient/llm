//! `gate`: rANS（range Asymmetric Numeral Systems）每字节平面熵编码。
//!
//! `.u` 格式的唯一压缩 codec（`CODECGATE`）。对 bit-shuffle + delta 变换后的
//! 字节流按平面独立做静态 rANS 熵编码，逼近信息熵下界。纯 safe Rust、零
//! unsafe、零第三方依赖。
//!
//! 关键参数：归一化因子 `TOTALFREQ = 2^16`、状态下界 `STATEMIN = 2^23`、32 位状态、8 位
//! renorm。每平面独立建 256 符号频率表并编码，避免把分布迥异的字节平面混在
//! 一起（整体编码会损失约 5pp，见设计记录）。

use crate::error::checkpointerror;

const NSYMBOLS: usize = 256;
const LOG2_M: u32 = 16;
const TOTALFREQ: u32 = 1 << LOG2_M;
const STATEMIN: u32 = 1 << 23;

/// 归一化 256 符号频率到总和精确为 `TOTALFREQ`（出现符号频率 ≥1，未出现为 0）。
fn normalize(counts: &[u32; NSYMBOLS]) -> [u32; NSYMBOLS] {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut freq = [0u32; NSYMBOLS];
    if total == 0 {
        freq[0] = TOTALFREQ;
        return freq;
    }
    let mut sum = 0i64;
    let mut max_s = 0usize;
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            let f = ((c as u64 * TOTALFREQ as u64 + total / 2) / total).max(1) as u32;
            freq[s] = f;
            sum += f as i64;
            if f > freq[max_s] {
                max_s = s;
            }
        }
    }
    let mut diff = TOTALFREQ as i64 - sum;
    while diff > 0 {
        freq[max_s] += 1;
        diff -= 1;
    }
    while diff < 0 && freq[max_s] > 1 {
        freq[max_s] -= 1;
        diff += 1;
    }
    if diff < 0 {
        // 兜底（理论上不可达）：把 TOTALFREQ 均匀分给出现符号。
        let present: Vec<usize> = (0..NSYMBOLS).filter(|&s| freq[s] > 0).collect();
        let n = present.len().max(1);
        let base = TOTALFREQ as usize / n;
        let rem = TOTALFREQ as usize % n;
        for (i, &s) in present.iter().enumerate() {
            freq[s] = (base + usize::from(i < rem)) as u32;
        }
    }
    freq
}

/// Packed per-symbol encode parameters, so the encoder fetches all four with a
/// single 16-byte load from one cache line instead of four separate arrays.
#[derive(Clone, Copy, Default)]
struct encentry {
    freq: u32,
    cum: u32,
    rcp: u32,
    xmax: u32,
}

/// 一个平面的频率表、累积频率表与打包的编码参数。
struct Table {
    freq: [u32; NSYMBOLS],
    cum: [u32; NSYMBOLS + 1],
    enc: [encentry; NSYMBOLS],
}

impl Table {
    fn fromcounts(counts: &[u32; NSYMBOLS]) -> Self {
        let freq = normalize(counts);
        let mut cum = [0u32; NSYMBOLS + 1];
        let mut acc = 0u32;
        for s in 0..NSYMBOLS {
            cum[s] = acc;
            acc += freq[s];
        }
        cum[NSYMBOLS] = acc;
        let mut enc = [encentry::default(); NSYMBOLS];
        for s in 0..NSYMBOLS {
            let f = freq[s];
            // ceil(2^32 / f) 定点逆元，用乘法替代 u32 除法；renorm 阈值 = 32768*f。
            let rcp = if f >= 2 {
                (1u64 << 32).div_ceil(f as u64) as u32
            } else {
                0
            };
            enc[s] = encentry {
                freq: f,
                cum: cum[s],
                rcp,
                xmax: ((STATEMIN >> LOG2_M) << 8) * f,
            };
        }
        Self { freq, cum, enc }
    }
}

/// 用预计算的定点逆元 `rcp` 做 `x / f` 与 `x % f`，结果与硬件除法精确一致。
#[inline]
fn div_rem(x: u32, f: u32, rcp: u32) -> (u32, u32) {
    if f == 1 {
        return (x, 0);
    }
    let q = ((x as u64 * rcp as u64) >> 32) as u32;
    let rem = x.wrapping_sub(q * f);
    if rem >= f {
        (q - 1, rem.wrapping_add(f))
    } else {
        (q, rem)
    }
}

/// 交错状态数：符号 `i` 编码到状态 `i % NLANES`，解码 SIMD 并行 `NLANES` 路。
const NLANES: usize = 8;

/// 编码一个平面，把 interleaved rANS 流追加到 `out`。
///
/// `NLANES` 个独立状态交错编码：符号 `i` 进入 `i % NLANES` 状态，各状态独立
/// renorm，最后 flush 各状态的 4 字节、各 renorm 流长度与逆序字节。解码逆序
/// 弹符号、按 `i % NLANES` 分派回同一状态。
fn encode_plane(out: &mut Vec<u8>, data: &[u8], table: &Table) {
    let mut xs = [STATEMIN; NLANES];
    // One flat per-lane renorm buffer instead of NLANES `Vec`s: a symbol whose
    // state sits just under 2^32 and whose frequency is 1 renormalizes at most
    // three bytes (x >> 24 < 2^8 below x_max >= 2^15), so `3 * symbols` per lane
    // is a provable bound and the buffer is allocated once.
    let per = data.len() / NLANES;
    let cap = per.saturating_mul(3).saturating_add(8);
    let mut rbuf = vec![0u8; cap * NLANES];
    let mut blen = [0usize; NLANES];
    for (i, &byte) in data.iter().enumerate() {
        let lane = i % NLANES;
        let e = table.enc[byte as usize];
        let mut x = xs[lane];
        let base = lane * cap;
        let mut at = base + blen[lane];
        while x >= e.xmax {
            rbuf[at] = (x & 0xFF) as u8;
            at += 1;
            x >>= 8;
        }
        blen[lane] = at - base;
        let (q, rem) = div_rem(x, e.freq, e.rcp);
        x = (q << LOG2_M) + rem + e.cum;
        xs[lane] = x;
    }
    for &x in &xs {
        for shift in [0, 8, 16, 24] {
            out.push(((x >> shift) & 0xFF) as u8);
        }
    }
    for &l in &blen {
        out.extend_from_slice(&(l as u32).to_le_bytes());
    }
    for lane in 0..NLANES {
        let seg = &rbuf[lane * cap..lane * cap + blen[lane]];
        for &b in seg.iter().rev() {
            out.push(b);
        }
    }
}

fn invalid(reason: &str) -> checkpointerror {
    checkpointerror::invalidmanifest {
        reason: reason.to_string(),
    }
}

fn take_u16(data: &mut &[u8]) -> Option<u16> {
    if data.len() < 2 {
        return None;
    }
    let v = u16::from_le_bytes([data[0], data[1]]);
    *data = &data[2..];
    Some(v)
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
/// interleaved 栈式解码：`NLANES` 个状态逆序弹符号，符号 `i` 由状态
/// `i % NLANES` 恢复，直接落入输出 `i` 位，无需 reverse。
fn decode_plane(stream: &[u8], dst: &mut [u8], table: &Table) -> Result<(), checkpointerror> {
    let n = dst.len();
    // slot → symbol 查找表，按符号区间 `fill`（每区间一次 memset 式填充），
    // 取代逐槽 `while` 推进——区间恰好覆盖 [0, TOTALFREQ)，故无需清零后逐槽判定。
    let mut lut = vec![0u8; TOTALFREQ as usize];
    for s in 0..NSYMBOLS {
        let start = table.cum[s] as usize;
        let end = table.cum[s + 1] as usize;
        if start < end {
            lut[start..end].fill(s as u8);
        }
    }

    let mut data = stream;
    let mut xs = [0u32; NLANES];
    for x in &mut xs {
        *x = take_u32(&mut data).ok_or_else(|| invalid("gate stream truncated"))?;
    }
    let mut lens = [0usize; NLANES];
    for l in &mut lens {
        *l = take_u32(&mut data).ok_or_else(|| invalid("gate stream truncated"))? as usize;
    }
    let mut streams: [&[u8]; NLANES] = [&[]; NLANES];
    for (l, st) in lens.iter().zip(streams.iter_mut()) {
        if data.len() < *l {
            return Err(invalid("gate stream truncated"));
        }
        *st = &data[..*l];
        data = &data[*l..];
    }

    // Unrolled by lane: symbol `i` uses lane `i % NLANES`, so once the base of a
    // block is a multiple of NLANES every lane index is a compile-time constant
    // and the eight states stay in registers across the block.
    macro_rules! DECODELANE {
        ($base:expr, $lane:literal) => {{
            let slot = (xs[$lane] & (TOTALFREQ - 1)) as usize;
            let s = lut[slot] as usize;
            dst[$base + $lane] = s as u8;
            xs[$lane] = table.freq[s] * (xs[$lane] >> LOG2_M) + slot as u32 - table.cum[s];
            while xs[$lane] < STATEMIN {
                let st = streams[$lane];
                if st.is_empty() {
                    return Err(invalid("gate stream truncated"));
                }
                xs[$lane] = (xs[$lane] << 8) | u32::from(st[0]);
                streams[$lane] = &st[1..];
            }
        }};
    }
    let mut i = n;
    while !i.is_multiple_of(NLANES) {
        i -= 1;
        let lane = i % NLANES;
        let slot = (xs[lane] & (TOTALFREQ - 1)) as usize;
        let s = lut[slot] as usize;
        dst[i] = s as u8;
        xs[lane] = table.freq[s] * (xs[lane] >> LOG2_M) + slot as u32 - table.cum[s];
        while xs[lane] < STATEMIN {
            let st = streams[lane];
            if st.is_empty() {
                return Err(invalid("gate stream truncated"));
            }
            xs[lane] = (xs[lane] << 8) | u32::from(st[0]);
            streams[lane] = &st[1..];
        }
    }
    while i > 0 {
        i -= NLANES;
        DECODELANE!(i, 7);
        DECODELANE!(i, 6);
        DECODELANE!(i, 5);
        DECODELANE!(i, 4);
        DECODELANE!(i, 3);
        DECODELANE!(i, 2);
        DECODELANE!(i, 1);
        DECODELANE!(i, 0);
    }
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
    let tail = if elem == 0 {
        &src[..0]
    } else {
        &src[elem * n..]
    };

    let mut tables = Vec::with_capacity(nplanes);
    for p in 0..nplanes {
        let plane = &src[p * n..(p + 1) * n];
        let mut counts = [0u32; NSYMBOLS];
        for &b in plane {
            counts[b as usize] += 1;
        }
        tables.push(Table::fromcounts(&counts));
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
    // Table encoding: a 256-bit presence bitmap followed by `u16` `freq-1` for
    // each present symbol. Frequencies are in 1..=TOTALFREQ=65536, so `freq-1` fits u16;
    // at u32 the table was ~1.6% of a 16 MiB snapshot, and this cuts it by more
    // than half (and far more for sparse planes).
    for table in &tables {
        let mut bitmap = [0u8; NSYMBOLS / 8];
        for s in 0..NSYMBOLS {
            if table.freq[s] != 0 {
                bitmap[s / 8] |= 1 << (s % 8);
            }
        }
        out.extend_from_slice(&bitmap);
        for s in 0..NSYMBOLS {
            if table.freq[s] != 0 {
                out.extend_from_slice(&((table.freq[s] - 1) as u16).to_le_bytes());
            }
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
        if data.len() < NSYMBOLS / 8 {
            return Err(invalid("gate freq bitmap truncated"));
        }
        let mut bitmap = [0u8; NSYMBOLS / 8];
        bitmap.copy_from_slice(&data[..NSYMBOLS / 8]);
        data = &data[NSYMBOLS / 8..];
        let mut freq = [0u32; NSYMBOLS];
        for s in 0..NSYMBOLS {
            if bitmap[s / 8] & (1 << (s % 8)) != 0 {
                let raw =
                    take_u16(&mut data).ok_or_else(|| invalid("gate freq table truncated"))?;
                freq[s] = u32::from(raw) + 1;
            }
        }
        let mut cum = [0u32; NSYMBOLS + 1];
        let mut acc = 0u32;
        for s in 0..NSYMBOLS {
            cum[s] = acc;
            acc += freq[s];
        }
        cum[NSYMBOLS] = acc;
        tables.push(Table {
            freq,
            cum,
            enc: [encentry::default(); NSYMBOLS],
        });
    }

    let mut lens = Vec::with_capacity(nplanes);
    for _ in 0..nplanes {
        lens.push(
            take_u32(&mut data).ok_or_else(|| invalid("gate stream length truncated"))? as usize,
        );
    }

    let mut out = vec![0u8; src_len];
    for p in 0..nplanes {
        if data.len() < lens[p] {
            return Err(invalid("gate stream truncated"));
        }
        decode_plane(&data[..lens[p]], &mut out[p * n..(p + 1) * n], &tables[p])?;
        data = &data[lens[p]..];
    }
    if data.len() < tail_len {
        return Err(invalid("gate tail truncated"));
    }
    out[nplanes * n..].copy_from_slice(&data[..tail_len]);
    Ok(out)
}
