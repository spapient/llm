//! 字节级 rANS 熵编码：8-lane 并行、频率归一化到 2^16，纯安全 Rust。
//!
//! 编码帧自描述，布局固定为：
//! 1. 原始长度 `u32`（小端）；
//! 2. 频率表：32 字节在场位图 + 每个在场符号的 `(freq - 1)`（`u16` 小端）；
//! 3. 8 条 lane 的最终状态（各 `u32` 小端）；
//! 4. 8 条 lane 的流长度（各 `u32` 小端）；
//! 5. 8 条 lane 的压缩流（编码时反向写入，落盘前逐条翻转）。
//!
//! rANS 要求解码顺序与编码相反，因此每条 lane 的字节流在 [`encode`] 末尾整体翻转，
//! [`decode`] 从头正向消费即可。所有算术只用 `u32/u64`，无浮点、无 `unsafe`。

use diagnostic::{Code, Error, Result};

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
        let mut acc = 0u32;
        for symbol in 0..NSYMBOLS {
            cum[symbol] = acc;
            acc += freq[symbol];
        }
        cum[NSYMBOLS] = acc;

        let mut enc = [Entry::default(); NSYMBOLS];
        for symbol in 0..NSYMBOLS {
            let f = freq[symbol];
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
        Self { freq, cum, enc }
    }

    /// 构建 64 KiB 解码符号表：槽位 `[cum[s]..cum[s+1])` 填符号 `s`。
    fn symbol_lut(&self) -> Box<[u8; LUT_LEN]> {
        let mut lut = vec![0u8; LUT_LEN].into_boxed_slice();
        for symbol in 0..NSYMBOLS {
            let start = self.cum[symbol] as usize;
            let end = self.cum[symbol + 1] as usize;
            if start < end {
                lut[start..end].fill(symbol as u8);
            }
        }
        // 长度由常量 LUT_LEN 保证，转换恒成立。
        lut.try_into()
            .unwrap_or_else(|_| panic!("LUT 长度必须为 {LUT_LEN}"))
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

/// 统计字节出现次数。
fn counts(src: &[u8]) -> [u32; NSYMBOLS] {
    let mut out = [0u32; NSYMBOLS];
    for &b in src {
        out[b as usize] += 1;
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

    let mut sum = 0i64;
    let mut max_symbol = 0usize;
    for (symbol, &c) in raw.iter().enumerate() {
        if c > 0 {
            let f = ((u64::from(c) * u64::from(TOTAL_FREQ) + total / 2) / total).max(1) as u32;
            freq[symbol] = f;
            sum += i64::from(f);
            if f > freq[max_symbol] {
                max_symbol = symbol;
            }
        }
    }

    // 缩放误差全部归入最大符号；它在场且足够大，可完全吸收。
    let mut diff = i64::from(TOTAL_FREQ) - sum;
    while diff > 0 {
        freq[max_symbol] += 1;
        diff -= 1;
    }
    while diff < 0 && freq[max_symbol] > 1 {
        freq[max_symbol] -= 1;
        diff += 1;
    }
    debug_assert_eq!(diff, 0, "归一化必须吸收全部舍入误差");
    freq
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
    for symbol in 0..NSYMBOLS {
        if bitmap[symbol / 8] & (1 << (symbol % 8)) != 0 {
            let pair = take::<2>(cur)?;
            freq[symbol] = u32::from(u16::from_le_bytes(pair)) + 1;
        }
    }
    let sum = freq.iter().try_fold(0u32, |acc, &f| acc.checked_add(f));
    if sum != Some(TOTAL_FREQ) {
        return Err(fault("rANS 频率表总和不为 2^16"));
    }
    Ok(freq)
}

/// 从流头取出固定长度字节，不足即报错。
fn take<const N: usize>(cur: &mut &[u8]) -> Result<[u8; N]> {
    let (head, rest) = cur
        .split_first_chunk::<N>()
        .ok_or_else(|| fault("rANS 流截断"))?;
    *cur = rest;
    Ok(*head)
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
    // 快路径：两字节都在，一次算出补 1 或补 2 的候选状态。
    if let Some(two) = stream.get(..2) {
        let once = (*state << 8) | u32::from(two[0]);
        let twice = (once << 8) | u32::from(two[1]);
        let need_two = once < STATE_MIN;
        *state = if need_two { twice } else { once };
        *stream = &stream[if need_two { 2 } else { 1 }..];
        return Ok(());
    }
    // 流末尾退回逐字节路径，至多补两次。
    for _ in 0..2 {
        if *state < STATE_MIN {
            let (byte, rest) = stream
                .split_first()
                .ok_or_else(|| fault("rANS 重规范化截断"))?;
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
    let mut buffers: [Vec<u8>; NLANES] = std::array::from_fn(|_| vec![0u8; cap]);
    let mut lens = [0usize; NLANES];

    for (i, &b) in src.iter().enumerate() {
        let lane = i % NLANES;
        let at = emit(
            &mut buffers[lane],
            lens[lane],
            &mut states[lane],
            table.enc[b as usize],
        );
        lens[lane] = at;
    }
    for lane in 0..NLANES {
        buffers[lane].truncate(lens[lane]);
    }
    (states, buffers)
}

/// 压缩为自描述帧；返回字节数（含头部），调用方据此判断是否值得保留。
pub(crate) fn encode(src: &[u8]) -> Vec<u8> {
    let table = Table::build(normalize(&counts(src)));
    let (states, mut buffers) = encode_lanes(src, &table);

    let mut out = Vec::new();
    let src_len = u32::try_from(src.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&src_len.to_le_bytes());
    write_freq(&mut out, &table.freq);

    for state in states {
        out.extend_from_slice(&state.to_le_bytes());
    }
    for buffer in &buffers {
        let len = u32::try_from(buffer.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_le_bytes());
    }

    // rANS 解码顺序相反：逐条翻转后再落盘。
    for buffer in &mut buffers {
        buffer.reverse();
        out.extend_from_slice(buffer);
    }
    out
}

/// 解码自描述帧，返回恢复出的原始字节。
pub(crate) fn decode(stored: &[u8]) -> Result<Vec<u8>> {
    let mut cur = stored;
    let src_len = u32::from_le_bytes(take::<4>(&mut cur)?) as usize;
    let table = Table::build(read_freq(&mut cur)?);

    let mut states = [0u32; NLANES];
    for state in &mut states {
        *state = u32::from_le_bytes(take::<4>(&mut cur)?);
    }
    let mut lens = [0usize; NLANES];
    for len in &mut lens {
        *len = u32::from_le_bytes(take::<4>(&mut cur)?) as usize;
    }

    let mut streams: [&[u8]; NLANES] = [&[]; NLANES];
    for lane in 0..NLANES {
        let (head, rest) = cur
            .split_at_checked(lens[lane])
            .ok_or_else(|| fault("rANS lane 流截断"))?;
        streams[lane] = head;
        cur = rest;
    }
    if !cur.is_empty() {
        return Err(fault("rANS 帧尾部有多余字节"));
    }

    let lut = table.symbol_lut();
    let mut dst = vec![0u8; src_len];

    // 先处理尾部不对齐的符号（与编码 lane 分配规则一致）。
    let mut at = src_len;
    while !at.is_multiple_of(NLANES) {
        at -= 1;
        let lane = at % NLANES;
        decode_one(
            &mut states[lane],
            &mut streams[lane],
            &table,
            &lut,
            &mut dst,
            at,
        )?;
    }
    // 主体按 8 个一组反向推进。
    while at >= NLANES {
        at -= NLANES;
        for lane in 0..NLANES {
            decode_one(
                &mut states[lane],
                &mut streams[lane],
                &table,
                &lut,
                &mut dst,
                at + lane,
            )?;
        }
    }
    Ok(dst)
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

/// 构造一个内部错误，统一携带模块定位。
fn fault(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("weights::rans")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 任意字节往返：解码必须与输入逐字节相等。
    fn assert_roundtrip(src: &[u8]) {
        let frame = encode(src);
        let back = decode(&frame).expect("解码");
        assert_eq!(back, src, "往返不一致");
    }

    #[test]
    fn empty_roundtrips() {
        assert_roundtrip(b"");
    }

    #[test]
    fn single_byte_roundtrips() {
        assert_roundtrip(&[42]);
    }

    #[test]
    fn repeated_value_roundtrips() {
        assert_roundtrip(&vec![7u8; 10_000]);
    }

    #[test]
    fn all_symbols_roundtrips() {
        let src: Vec<u8> = (0..=255u16).map(|b| b as u8).collect();
        assert_roundtrip(&src);
        assert_roundtrip(&src.repeat(64));
    }

    #[test]
    fn pseudo_random_roundtrips() {
        // 线性同余生成确定性的"高熵"字节，覆盖近似均匀分布。
        let mut x = 0x1234_5678u32;
        let src: Vec<u8> = (0..50_000)
            .map(|_| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (x >> 16) as u8
            })
            .collect();
        assert_roundtrip(&src);
    }

    #[test]
    fn skewed_distribution_roundtrips() {
        // 90% 为 0，其余少量符号，压缩后应明显变小。
        let mut src = vec![0u8; 20_000];
        for (i, slot) in src.iter_mut().enumerate() {
            if i % 10 == 0 {
                *slot = (i % 7) as u8 + 1;
            }
        }
        let frame = encode(&src);
        assert!(frame.len() < src.len(), "高偏斜数据应被压缩");
        assert_roundtrip(&src);
    }

    #[test]
    fn truncated_frame_is_rejected() {
        let frame = encode(&vec![3u8; 1000]);
        for cut in [0, 1, 4, 33, frame.len() - 1] {
            assert!(decode(&frame[..cut]).is_err(), "截断到 {cut} 必须报错");
        }
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(decode(&[0xFF; 64]).is_err());
    }

    #[test]
    fn normalized_freq_sums_exactly() {
        let raw = [
            counts(&[1u8, 1, 2, 3, 3, 3]),
            counts(&(0..255u16).map(|b| b as u8).collect::<Vec<u8>>()),
        ];
        for r in raw {
            let freq = normalize(&r);
            assert_eq!(freq.iter().sum::<u32>(), TOTAL_FREQ);
        }
    }

    /// 香农熵（bits/符号），无损压缩的物理下界。
    fn entropy(src: &[u8]) -> f64 {
        let n = src.len() as f64;
        let mut h = 0.0f64;
        for &c in &counts(src) {
            if c > 0 {
                let p = c as f64 / n;
                h -= p * p.log2();
            }
        }
        h
    }

    /// 从帧头解析出纯 rans payload 长度（8 条 lane 流之和）。
    fn payload_len(frame: &[u8]) -> usize {
        let mut cur = frame;
        take::<4>(&mut cur).expect("src_len");
        let bitmap = take::<32>(&mut cur).expect("bitmap");
        let present = bitmap.iter().map(|b| b.count_ones()).sum::<u32>() as usize;
        for _ in 0..present {
            take::<2>(&mut cur).expect("freq");
        }
        for _ in 0..NLANES {
            take::<4>(&mut cur).expect("state");
        }
        let mut total = 0usize;
        for _ in 0..NLANES {
            total += u32::from_le_bytes(take::<4>(&mut cur).expect("lane len")) as usize;
        }
        total
    }

    #[test]
    fn payload_approaches_entropy_bound() {
        // 确定性的偏斜分布：80% 为 0，其余在 4 个符号间，约 1 bit/符号。
        let n = 200_000usize;
        let src: Vec<u8> = (0..n)
            .map(|i| if i % 5 == 0 { (i % 4) as u8 + 1 } else { 0 })
            .collect();

        let h = entropy(&src);
        let ideal = (n as f64) * h / 8.0;
        let frame = encode(&src);
        let payload = payload_len(&frame) as f64;

        // rans 核心效率不低于 99%：payload 与熵下界之差在 1% 内。
        let efficiency = ideal / payload;
        assert!(
            efficiency >= 0.99,
            "rANS 效率 {efficiency:.4} 低于 0.99（payload={payload}, 熵下界={ideal}）"
        );

        // 帧总开销（头部）是有界常数，不随 n 增长：头部 ≤ 200 字节。
        let overhead = frame.len() as f64 - payload;
        assert!(overhead <= 200.0, "头部开销 {overhead} 应为有界常数");
    }

    #[test]
    fn uniform_data_is_not_amplified() {
        // 近似均匀（熵≈8 bit）的数据：rans 帧不会比原始短，写端因此回退 store。
        let mut x = 0xABCDEF01u32;
        let src: Vec<u8> = (0..10_000)
            .map(|_| {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(1);
                (x >> 8) as u8
            })
            .collect();
        let frame = encode(&src);
        // 编码帧本身允许含头部，但写端策略保证只有严格变小才选 rans。
        assert!(frame.len() >= src.len(), "均匀数据不应被 rans 压缩");
    }
}
