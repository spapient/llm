//! 字节级 rANS 熵编码：8-lane 并行、频率归一化到 2^16，纯安全 Rust。

use crate::bad;
use diagnostic::Result;

/// 字母表大小（字节）。
const NSYMBOLS: usize = 256;

/// 频率总精度的对数：`M = 2^16`。
const LOG2_M: u32 = 16;

/// 归一化频率总和。
const TOTAL_FREQ: u32 = 1 << LOG2_M;

/// 解码符号表长度（按状态低 16 位索引）。
const LUT_LEN: usize = TOTAL_FREQ as usize;

/// 合法状态下界；低于它就在解码端重规范化补字节。
const STATE_MIN: u32 = 1 << 23;

/// 并行 lane 数。
const NLANES: usize = 8;

/// 单个符号的编码入口。
#[derive(Clone, Copy, Default)]
struct Entry {
    /// 归一化频率。
    freq: u32,
    /// 累积频率（起始槽位）。
    cum: u32,
    /// 定点倒数，用于以乘法替代除法。
    rcp: u32,
    /// 状态上界：达到它就要先 renorm 再编码。
    xmax: u32,
}

/// 一张完整的频率表，编码与解码共用同一份归一化结果。
#[derive(Clone, Copy)]
struct Table {
    /// 各符号频率。
    freq: [u32; NSYMBOLS],
    /// 累积频率，`cum[256] = TOTAL_FREQ`。
    cum: [u32; NSYMBOLS + 1],
    /// 编码入口。
    enc: [Entry; NSYMBOLS],
}

impl Table {
    /// 从已归一化、总和恰为 `TOTAL_FREQ` 的频率构建编码/解码入口。
    fn build(freq: [u32; NSYMBOLS]) -> Self {
        let mut cum = [0u32; NSYMBOLS + 1];
        let mut enc = [Entry::default(); NSYMBOLS];
        let mut acc = 0u32;
        for symbol in 0..NSYMBOLS {
            let f = freq[symbol];
            cum[symbol] = acc;
            acc += f;
            // f == 1 时无需倒数（divrem 对 f==1 有专用分支）。
            let rcp = if f >= 2 {
                (1u64 << 32).div_ceil(u64::from(f)) as u32
            } else {
                0
            };
            enc[symbol] = Entry {
                freq: f,
                cum: cum[symbol],
                rcp,
                xmax: ((STATE_MIN >> LOG2_M) << 8) * f,
            };
        }
        cum[NSYMBOLS] = acc;
        Self { freq, cum, enc }
    }

    /// 构建 64 KiB 解码符号表：槽位 `[cum[s]..cum[s+1])` 填符号 `s`。
    fn symbol_lut(&self) -> Box<[u8; LUT_LEN]> {
        let mut lut = Box::new([0u8; LUT_LEN]);
        for symbol in 0..NSYMBOLS {
            let start = self.cum[symbol] as usize;
            let end = self.cum[symbol + 1] as usize;
            if start < end {
                lut[start..end].fill(symbol as u8);
            }
        }
        lut
    }
}

/// 定点 `(x / f, x % f)`：用倒数乘法求商，再修正一次可能的取整误差。
#[inline]
fn divrem(x: u32, f: u32, rcp: u32) -> (u32, u32) {
    if f == 1 {
        return (x, 0);
    }
    let q = ((u64::from(x) * u64::from(rcp)) >> 32) as u32;
    let rem = x.wrapping_sub(q * f);
    if rem >= f {
        (q - 1, rem.wrapping_add(f))
    } else {
        (q, rem)
    }
}

/// 直方图副本数：偏斜输入（比如 80% 是同一个符号）会让"同一个 bin 的 `+= 1`"退化
/// 成一条串行的 store-to-load 转发链，实测占编码总时间的一半以上。
const WAYS: usize = 8;

/// 统计字节出现次数。
fn counts(src: &[u8]) -> [u32; NSYMBOLS] {
    merge(&scatter(src))
}

/// 按位置分流：相邻字节落到不同副本上，依赖链被打断。
fn scatter(src: &[u8]) -> [[u32; NSYMBOLS]; WAYS] {
    let mut lanes = [[0u32; NSYMBOLS]; WAYS];
    let (chunks, rest) = src.as_chunks::<WAYS>();
    for chunk in chunks {
        for way in 0..WAYS {
            lanes[way][chunk[way] as usize] += 1;
        }
    }
    for &b in rest {
        lanes[0][b as usize] += 1;
    }
    lanes
}

/// 合并各副本：只是 256×8 次加法。
fn merge(lanes: &[[u32; NSYMBOLS]; WAYS]) -> [u32; NSYMBOLS] {
    let mut out = lanes[0];
    for lane in lanes.iter().skip(1) {
        for (slot, &count) in out.iter_mut().zip(lane.iter()) {
            *slot += count;
        }
    }
    out
}

/// 把原始计数归一化到总和精确等于 `TOTAL_FREQ`。
fn normalize(raw: &[u32; NSYMBOLS]) -> [u32; NSYMBOLS] {
    let total: u64 = raw.iter().map(|&c| u64::from(c)).sum();
    let mut freq = [0u32; NSYMBOLS];
    if total == 0 {
        freq[0] = TOTAL_FREQ;
        return freq;
    }
    for (symbol, &count) in raw.iter().enumerate() {
        freq[symbol] = scaled(count, total);
    }
    let mut max_symbol = 0usize;
    for (symbol, &f) in freq.iter().enumerate() {
        if f > freq[max_symbol] {
            max_symbol = symbol;
        }
    }
    let sum: i64 = freq.iter().map(|&f| i64::from(f)).sum();
    absorb(&mut freq, max_symbol, i64::from(TOTAL_FREQ) - sum);
    freq
}

/// 单个计数的归一化值：四舍五入到 `TOTAL_FREQ` 的总量上，在场符号至少拿到 1。
fn scaled(count: u32, total: u64) -> u32 {
    if count == 0 {
        return 0;
    }
    ((u64::from(count) * u64::from(TOTAL_FREQ) + total / 2) / total).max(1) as u32
}

/// 缩放误差全部归入最大符号；它在场且足够大，可完全吸收。
fn absorb(freq: &mut [u32; NSYMBOLS], symbol: usize, mut diff: i64) {
    while diff > 0 {
        freq[symbol] += 1;
        diff -= 1;
    }
    while diff < 0 && freq[symbol] > 1 {
        freq[symbol] -= 1;
        diff += 1;
    }
    debug_assert_eq!(diff, 0, "归一化必须吸收全部舍入误差");
}

/// 写出频率表：32 字节位图 + 在场符号的 `(freq - 1)` u16。
fn write_freq(out: &mut Vec<u8>, freq: &[u32; NSYMBOLS]) {
    let mut bitmap = [0u8; NSYMBOLS / 8];
    for symbol in 0..NSYMBOLS {
        if freq[symbol] != 0 {
            bitmap[symbol / 8] |= 1 << (symbol % 8);
        }
    }
    out.extend_from_slice(&bitmap);
    for &f in freq {
        if f != 0 {
            out.extend_from_slice(&((f - 1) as u16).to_le_bytes());
        }
    }
}

/// 读回频率表并校验总和。
fn read_freq(cur: &mut &[u8]) -> Result<[u32; NSYMBOLS]> {
    let bitmap = take::<{ NSYMBOLS / 8 }>(cur)?;
    let mut freq = [0u32; NSYMBOLS];
    let present = (0..NSYMBOLS).filter(|&symbol| bitmap[symbol / 8] & (1 << (symbol % 8)) != 0);
    for symbol in present {
        freq[symbol] = u32::from(le_u16(cur)?) + 1;
    }
    let sum = freq.iter().try_fold(0u32, |acc, &f| acc.checked_add(f));
    if sum != Some(TOTAL_FREQ) {
        return Err(bad("rANS 频率表总和不为 2^16"));
    }
    Ok(freq)
}

/// 从流头取出固定长度字节，不足即报错。
fn take<const N: usize>(cur: &mut &[u8]) -> Result<[u8; N]> {
    let (head, rest) = cur
        .split_first_chunk::<N>()
        .ok_or_else(|| bad("rANS 流截断"))?;
    *cur = rest;
    Ok(*head)
}

/// 定宽小端 `u16`：`take::<2>` 加一次 `from_le_bytes` 收窄。
#[inline]
fn le_u16(cur: &mut &[u8]) -> Result<u16> {
    take::<2>(cur).map(u16::from_le_bytes)
}

/// 定宽小端 `u32`：`take::<4>` 加一次 `from_le_bytes` 收窄。
#[inline]
fn le_u32(cur: &mut &[u8]) -> Result<u32> {
    take::<4>(cur).map(u32::from_le_bytes)
}

/// 编码一个符号：必要时先把状态高位字节写进 lane 缓冲，再推进状态。
#[inline]
fn emit(buf: &mut [u8], mut at: usize, state: &mut u32, entry: Entry) -> usize {
    while *state >= entry.xmax {
        buf[at] = (*state & 0xFF) as u8;
        at += 1;
        *state >>= 8;
    }
    let (q, rem) = divrem(*state, entry.freq, entry.rcp);
    *state = (q << LOG2_M) + rem + entry.cum;
    at
}

/// 解码端重规范化：状态不足 `STATE_MIN` 时补 1–2 字节。
#[inline]
fn renorm(state: &mut u32, stream: &mut &[u8]) -> Result<()> {
    if *state >= STATE_MIN {
        return Ok(());
    }
    match stream.get(..2) {
        Some(two) => {
            pair(state, stream, two);
            Ok(())
        }
        None => trickle(state, stream),
    }
}

/// 两字节都在时的快路径：一次算出补 1 与补 2 的候选状态，按 `STATE_MIN` 选一个。
#[inline]
fn pair(state: &mut u32, stream: &mut &[u8], two: &[u8]) {
    let once = (*state << 8) | u32::from(two[0]);
    let twice = (once << 8) | u32::from(two[1]);
    let need_two = once < STATE_MIN;
    *state = if need_two { twice } else { once };
    *stream = &stream[usize::from(need_two) + 1..];
}

/// 流末尾的逐字节路径：至多补两次，每次只在仍不足时才取字节。
#[inline]
fn trickle(state: &mut u32, stream: &mut &[u8]) -> Result<()> {
    for _ in 0..2 {
        if *state < STATE_MIN {
            let (byte, rest) = stream
                .split_first()
                .ok_or_else(|| bad("rANS 重规范化截断"))?;
            *state = (*state << 8) | u32::from(*byte);
            *stream = rest;
        }
    }
    Ok(())
}

/// 把输入按位置轮流分配给 8 条 lane，各 lane 独立编码。
fn encode_lanes(src: &[u8], table: &Table) -> ([u32; NLANES], [Vec<u8>; NLANES]) {
    let mut states = [STATE_MIN; NLANES];
    // 每条 lane 最坏情况每符号写 3 字节，先按上界定长分配再按索引写，末尾截断。
    let per = src.len().div_ceil(NLANES);
    let cap = per.saturating_mul(3).saturating_add(8);
    let mut bufs: [Vec<u8>; NLANES] = std::array::from_fn(|_| vec![0u8; cap]);
    let mut lens = [0usize; NLANES];

    for (i, &b) in src.iter().enumerate() {
        let lane = i % NLANES;
        let at = emit(
            &mut bufs[lane],
            lens[lane],
            &mut states[lane],
            table.enc[b as usize],
        );
        lens[lane] = at;
    }
    for lane in 0..NLANES {
        bufs[lane].truncate(lens[lane]);
    }
    (states, bufs)
}

/// 压缩为自描述帧；返回字节数（含头部），调用方据此判断是否值得保留。
pub(crate) fn encode(src: &[u8]) -> Vec<u8> {
    let table = Table::build(normalize(&counts(src)));
    let (states, mut bufs) = encode_lanes(src, &table);

    let mut out = Vec::new();
    let src_len = u32::try_from(src.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&src_len.to_le_bytes());
    write_freq(&mut out, &table.freq);

    for state in states {
        out.extend_from_slice(&state.to_le_bytes());
    }
    for buf in &bufs {
        let len = u32::try_from(buf.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_le_bytes());
    }

    // rANS 解码顺序相反：逐条翻转后再落盘。
    for buf in &mut bufs {
        buf.reverse();
        out.extend_from_slice(buf);
    }
    out
}

/// 解码自描述帧，返回恢复出的原始字节。
pub(crate) fn decode(stored: &[u8], cap: usize) -> Result<Vec<u8>> {
    let mut cur = stored;
    let (src_len, table, words) = header(&mut cur, cap)?;
    let (mut states, mut streams) = lanes(&mut cur, words)?;
    let mut dst = vec![0u8; src_len];
    symbols(&mut dst, &mut states, &mut streams, &table)?;
    Ok(dst)
}

/// 帧头：原始长度（已核对上限）、归一化频率表，以及状态与长度两组的 16 个 `u32` 原文。
fn header(cur: &mut &[u8], cap: usize) -> Result<(usize, Table, [u32; 2 * NLANES])> {
    let len = declared_len(cur, cap)?;
    let table = Table::build(read_freq(cur)?);
    let mut words = [0u32; 2 * NLANES];
    for word in &mut words {
        *word = le_u32(cur)?;
    }
    Ok((len, table, words))
}

/// 把两组 `u32` 拆成状态与长度，再按长度切出 8 条 lane 流；切完必须正好到帧尾。
fn lanes<'a>(
    cur: &mut &'a [u8],
    words: [u32; 2 * NLANES],
) -> Result<([u32; NLANES], [&'a [u8]; NLANES])> {
    let mut states = [0u32; NLANES];
    states.copy_from_slice(&words[..NLANES]);
    let mut streams: [&[u8]; NLANES] = [&[]; NLANES];
    for (stream, len) in streams.iter_mut().zip(&words[NLANES..]) {
        let (head, rest) = cur
            .split_at_checked(*len as usize)
            .ok_or_else(|| bad("rANS lane 流截断"))?;
        *stream = head;
        *cur = rest;
    }
    if cur.is_empty() {
        return Ok((states, streams));
    }
    Err(bad("rANS 帧尾部有多余字节"))
}

/// 与编码相反的次序解出全部符号：位置从末尾往前，`at % NLANES` 就给回编码时的 lane。
fn symbols(
    dst: &mut [u8],
    states: &mut [u32; NLANES],
    streams: &mut [&[u8]; NLANES],
    table: &Table,
) -> Result<()> {
    let lut = table.symbol_lut();
    for at in (0..dst.len()).rev() {
        let lane = at % NLANES;
        decode_one(&mut states[lane], &mut streams[lane], table, &lut, dst, at)?;
    }
    Ok(())
}

/// 解码单个符号：查表定符号，回退状态后重规范化。
#[inline]
fn decode_one(
    state: &mut u32,
    stream: &mut &[u8],
    table: &Table,
    lut: &[u8; LUT_LEN],
    dst: &mut [u8],
    at: usize,
) -> Result<()> {
    let slot = *state & (TOTAL_FREQ - 1);
    let symbol = lut[slot as usize] as usize;
    dst[at] = symbol as u8;
    *state = table.freq[symbol] * (*state >> LOG2_M) + slot - table.cum[symbol];
    renorm(state, stream)
}

/// 读出帧头声明的原始长度，并核对调用方给定的上限。
fn declared_len(cur: &mut &[u8], cap: usize) -> Result<usize> {
    let len = le_u32(cur)? as usize;
    if len > cap {
        return Err(bad(format!("rANS 帧声明 {len} 字节，超过上限 {cap}")));
    }
    Ok(len)
}

#[cfg(test)]
#[path = "../tests/rans.rs"]
mod tests;
