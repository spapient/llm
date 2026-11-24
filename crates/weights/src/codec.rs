//! `axes` 的编解码机制：熵帧、字典/稀疏/量化三种载荷的读写，以及定宽取字节。

use std::collections::HashMap;

use diagnostic::Result;

use crate::bad;
use storage::leb::Cursor;

/// 标记位：0 = store（原样），1 = rans 帧。
const STORE_MARK: u8 = 0;
const RANS_MARK: u8 = 1;

pub(crate) fn pack_bytes(src: &[u8]) -> Vec<u8> {
    let frame = encode_rans(src);
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
        RANS_MARK => decode_rans(body, cap),
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

/// 唯一值排序表 + 等长索引序列。排序表保持线上格式稳定，哈希表只加速写端定位。
pub(crate) fn dict_payload(w: &[f32]) -> Option<Vec<u8>> {
    let mut values: Vec<u32> = w.iter().map(|v| v.to_bits()).collect();
    values.sort_unstable();
    values.dedup();
    if values.len() == w.len() {
        return None; // 全唯一，字典表 + 全宽索引必然大于 raw，无收益。
    }
    let width = idx_width(values.len());
    let idxs: HashMap<u32, u32> = values
        .iter()
        .enumerate()
        .map(|(id, &bits)| (bits, id as u32))
        .collect();
    let mut payload = Vec::with_capacity(8 + values.len() * 4 + w.len() * width);
    payload.extend_from_slice(&(width as u32).to_le_bytes());
    payload.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for bits in &values {
        payload.extend_from_slice(&bits.to_le_bytes());
    }
    // 定宽小端：哈希查询均摊 O(1)，索引仍取排序表序号，输出与二分实现逐位一致。
    for value in w {
        let id = idxs.get(&value.to_bits()).copied()?;
        payload.extend_from_slice(&id.to_le_bytes()[..width]);
    }
    Some(pack_bytes(&payload))
}

/// 索引宽度：`256`/`65536` 个值以内用 1/2 字节，其余 4 字节。
fn idx_width(count: usize) -> usize {
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
            let id = read_idx(&bytes, &mut at, width)?;
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
fn read_idx(bytes: &[u8], at: &mut usize, width: usize) -> Result<usize> {
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
    let min = w.iter().copied().reduce(f32::min)?;
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
    let (min, exponent, width, at) = quant_head(&bytes)?;
    let codes = unpack_bits(&bytes[at..], want, width)?;
    let step = 2.0f32.powi(exponent as i8 as i32);
    Ok(codes.iter().map(|&c| min + c as f32 * step).collect())
}

/// quant 头：最小基准、指数、位宽与码流起点；位宽 > 8 拒绝。
fn quant_head(bytes: &[u8]) -> Result<(f32, u8, u8, usize)> {
    let mut at = 0;
    let min = word(bytes, &mut at, "quant 头截断").map(f32::from_bits)?;
    let [exponent, width] = take::<2>(bytes, &mut at, "quant 头截断")?;
    if width > 8 {
        return Err(bad("quant 位宽超过 8"));
    }
    Ok((min, exponent, width, at))
}

/// 小端字节 ↔ 标量的宽度事实：f32/f64 各一条实现。`to_le_bytes`/`from_le_bytes` 据此泛化，
/// 不再每个标量类型各写一份「flat_map / as_chunks + from_le_bytes」。
pub(crate) trait LittleEndian: Copy + 'static {
    const WIDTH: usize;
    /// 把标量的小端字节追加进缓冲。
    fn push_le(self, out: &mut Vec<u8>);
    /// 从恰好 `WIDTH` 字节的切片解出一个标量。
    fn from_le(chunk: &[u8]) -> Self;
}

impl LittleEndian for f32 {
    const WIDTH: usize = 4;
    fn push_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&f32::to_le_bytes(self));
    }
    fn from_le(chunk: &[u8]) -> Self {
        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
    }
}

impl LittleEndian for f64 {
    const WIDTH: usize = 8;
    fn push_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&f64::to_le_bytes(self));
    }
    fn from_le(chunk: &[u8]) -> Self {
        f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ])
    }
}

/// 标量序列 → 小端字节。
pub(crate) fn to_le_bytes<T: LittleEndian>(values: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * T::WIDTH);
    for &v in values {
        v.push_le(&mut out);
    }
    out
}

/// 小端字节 → 标量序列：字节数必须恰为 `want` 个标量，截断与余量都显式报错。
pub(crate) fn from_le_bytes<T: LittleEndian>(bytes: &[u8], want: usize) -> Result<Vec<T>> {
    if bytes.len() != want * T::WIDTH {
        return Err(bad(format!(
            "{} 字节 {} != 期望 {}",
            std::any::type_name::<T>(),
            bytes.len(),
            want * T::WIDTH
        )));
    }
    let mut out = Vec::with_capacity(want);
    for chunk in bytes.chunks_exact(T::WIDTH) {
        out.push(T::from_le(chunk));
    }
    Ok(out)
}

pub(crate) fn f32_bytes(w: &[f32]) -> Vec<u8> {
    to_le_bytes(w)
}

pub(crate) fn bytes_to_f32(bytes: &[u8], want: usize) -> Result<Vec<f32>> {
    from_le_bytes(bytes, want)
}

pub(crate) fn pack_bits(codes: &[u32], width: u8) -> Vec<u8> {
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

pub(crate) fn unpack_bits(src: &[u8], count: usize, width: u8) -> Result<Vec<u32>> {
    if width == 0 {
        return Ok(vec![0; count]);
    }
    let mut out = Vec::with_capacity(count);
    let mut acc = 0u64;
    let mut bits = 0u32;
    let mut at = 0usize;
    for _ in 0..count {
        refill(src, &mut at, &mut acc, &mut bits, width)?;
        out.push((acc & ((1u64 << width) - 1)) as u32);
        acc >>= width;
        bits -= width as u32;
    }
    Ok(out)
}

/// 补足累加器到 `width` 位：位流逐字节串行推进，无法整块预取，故保留 while；
/// 原地推进字节位置、位缓存与已用位数。
fn refill(src: &[u8], at: &mut usize, acc: &mut u64, bits: &mut u32, width: u8) -> Result<()> {
    while *bits < width as u32 {
        let byte = src.get(*at).copied().ok_or_else(|| bad("quant 码流截断"))?;
        *acc |= (byte as u64) << *bits;
        *at += 1;
        *bits += 8;
    }
    Ok(())
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

// —— 字节级 rANS 熵编码：8-lane 并行、频率归一化到 2^16，纯安全 Rust ——

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
    let bitmap = take_rans::<{ NSYMBOLS / 8 }>(cur)?;
    let mut freq = [0u32; NSYMBOLS];
    let present = (0..NSYMBOLS).filter(|&symbol| bitmap[symbol / 8] & (1 << (symbol % 8)) != 0);
    for symbol in present {
        freq[symbol] = u32::from(u16::from_le_bytes(take_rans::<2>(cur)?)) + 1;
    }
    let sum = freq.iter().try_fold(0u32, |acc, &f| acc.checked_add(f));
    if sum != Some(TOTAL_FREQ) {
        return Err(bad("rANS 频率表总和不为 2^16"));
    }
    Ok(freq)
}

/// 从流头取出固定长度字节，不足即报错。
fn take_rans<const N: usize>(cur: &mut &[u8]) -> Result<[u8; N]> {
    let (head, rest) = cur
        .split_first_chunk::<N>()
        .ok_or_else(|| bad("rANS 流截断"))?;
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
pub(crate) fn encode_rans(src: &[u8]) -> Vec<u8> {
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
pub(crate) fn decode_rans(stored: &[u8], cap: usize) -> Result<Vec<u8>> {
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
        *word = u32::from_le_bytes(take_rans::<4>(cur)?);
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
    let len = u32::from_le_bytes(take_rans::<4>(cur)?) as usize;
    if len > cap {
        return Err(bad(format!("rANS 帧声明 {len} 字节，超过上限 {cap}")));
    }
    Ok(len)
}

#[cfg(test)]
#[path = "../tests/rans.rs"]
mod rans_tests;
