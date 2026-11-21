//! 五轴自适应选码（配置化）：`raw / dict / sparse / quant / rank` 五个候选，每个候选只有在解码后
//! 与原始位模式逐元素一致（`to_bits` 相等）时才被采纳，最终在通过验证的候选里取字节最小者。
//!
//! 五轴各自实现 `AxisCfg`，搜索与逐位验证是泛型闭集：新增一轴只加一个 `impl AxisCfg` 与
//! `AXES` 常量表一行，不碰分派、验码与形状判据。`raw` 是恒可用的基线（f32 字节走 rANS），
//! 因此非空输入总有返回；其余轴不满足逐位精确即判不适用，绝不谎称无损。
//!
//! `rank` 轴比其余四轴多一道门：[`ledger::certify`] 先判谱隙，判定不过（无谱隙、形状退化）就
//! 整轴不参与，秩与证书出自同一次分解——于是「证书不过就拒压回退」不是文档里的口号，而是
//! [`search`] 里 `encode` 返回 `None` 的那条分支。
//!
//! 证书本身随载荷落盘（布局 `note_len | note | low_len | low | rest`）：它不再是写完就丢的事后
//! 账目，而是与两段残差同受一条 CRC 保护的载荷内容，读端 [`decode`] 用载荷自己把证书的断言重算
//! 一遍（[`ledger::audit`]），对不上就拒绝——「可证压缩」因此有一个可执行的判据。

use diagnostic::{Code, Error, Result};

use crate::leb::Cursor;
use crate::ledger;
use crate::rans;

/// artifact 魔数（v4：`rank` 载荷改存低秩项的两个因子而非整张 `L`，v1–v3 不再可读）。
const MAGIC: [u8; 4] = *b"AX04";

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
    /// 精确低秩 + 残差，秩由 L0 谱隙证书给出。
    Rank = 4,
}

/// 轴配置闭集：下标即码位（[`Axis`] 判别值的逆）。
const AXES: [&dyn AxisCfg; 5] = [&Raw, &Dict, &Sparse, &Quant, &Rank];

/// 一轴的编码/解码配置：搜索与验证只面向本 trait，不感知具体轴。
trait AxisCfg {
    /// 本轴码位。
    fn kind(&self) -> Axis;
    /// 编码为载荷；`None` 表示该轴对输入不适用。
    fn encode(&self, w: &[f32], m: usize, n: usize) -> Option<Vec<u8>>;
    /// 从载荷游标解码一张 `shape = (m, n)` 的矩阵（游标已越过 [`HEAD`]）。
    ///
    /// 传形状而不是只传元素数：秩轴的载荷自带证书，复核「秩落在 `1 <= r < min(m,n)`」要用原始
    /// 形状，而元素数已经把它丢了。
    fn decode(&self, cur: &mut Cursor<'_>, shape: (usize, usize)) -> Result<Vec<f32>>;
}

/// 原始字节轴。
struct Raw;
/// 唯一值字典轴。
struct Dict;
/// 结构化稀疏轴。
struct Sparse;
/// 精确网格量化轴。
struct Quant;
/// 精确低秩 + 残差轴。
struct Rank;

impl AxisCfg for Raw {
    fn kind(&self) -> Axis {
        Axis::Raw
    }
    fn encode(&self, w: &[f32], _: usize, _: usize) -> Option<Vec<u8>> {
        Some(pack_bytes(&f32_bytes(w)))
    }
    fn decode(&self, cur: &mut Cursor<'_>, shape: (usize, usize)) -> Result<Vec<f32>> {
        let want = area(shape);
        bytes_to_f32(&unpack_bytes(cur, payload_cap(want))?, want)
    }
}

impl AxisCfg for Dict {
    fn kind(&self) -> Axis {
        Axis::Dict
    }
    fn encode(&self, w: &[f32], _: usize, _: usize) -> Option<Vec<u8>> {
        dict_payload(w)
    }
    fn decode(&self, cur: &mut Cursor<'_>, shape: (usize, usize)) -> Result<Vec<f32>> {
        decode_dict(cur, area(shape))
    }
}

impl AxisCfg for Sparse {
    fn kind(&self) -> Axis {
        Axis::Sparse
    }
    fn encode(&self, w: &[f32], _: usize, _: usize) -> Option<Vec<u8>> {
        sparse_payload(w)
    }
    fn decode(&self, cur: &mut Cursor<'_>, shape: (usize, usize)) -> Result<Vec<f32>> {
        decode_sparse(cur, area(shape))
    }
}

impl AxisCfg for Quant {
    fn kind(&self) -> Axis {
        Axis::Quant
    }
    fn encode(&self, w: &[f32], _: usize, _: usize) -> Option<Vec<u8>> {
        quant_payload(w)
    }
    fn decode(&self, cur: &mut Cursor<'_>, shape: (usize, usize)) -> Result<Vec<f32>> {
        decode_quant(cur, area(shape))
    }
}

impl AxisCfg for Rank {
    fn kind(&self) -> Axis {
        Axis::Rank
    }
    fn encode(&self, w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
        rank_payload(w, m, n)
    }
    fn decode(&self, cur: &mut Cursor<'_>, shape: (usize, usize)) -> Result<Vec<f32>> {
        decode_rank(cur, shape)
    }
}

impl Axis {
    fn from_code(code: u8) -> Result<Self> {
        AXES.get(usize::from(code))
            .map(|cfg| cfg.kind())
            .ok_or_else(|| bad(&format!("未知轴码 {code}")))
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
    // `Raw` 恒可用，是基线；其余四轴只在与基线比长度时才有胜出可能。
    let base = frame(Axis::Raw, m, n, pack_bytes(&f32_bytes(w)));
    let smaller = AXES[1..]
        .iter()
        .filter_map(|cfg| {
            cfg.encode(w, m, n)
                .map(|payload| frame(cfg.kind(), m, n, payload))
        })
        .filter(|pick| verified(pick, w) && pick.len() < base.len())
        .min_by_key(Pick::len);
    let best = smaller.unwrap_or(base);
    // 基线必须自身逐位成立；这是压缩器的不变式，破了就是实现 bug，因此报错而不是 panic。
    verified(&best, w)
        .then_some(best)
        .ok_or_else(|| bad("raw 基线没能逐位还原，压缩器自身不自洽"))
}

/// 字节级无损压缩入口：`data` 是行主序 f32 小端字节（`m*n` 个元素）。仅当五轴 artifact
/// 严格小于原字节时才返回 `Some`，供 `.u` 容器写端在 rANS 之外择优。
pub fn compress(data: &[u8], m: usize, n: usize) -> Result<Option<Vec<u8>>> {
    let w = bytes_to_f32(data, m * n)?;
    let pick = search(&w, m, n)?;
    Ok((pick.len() < data.len()).then_some(pick.artifact))
}

/// 字节级无损解压入口：与 [`decode`] 同构，但把结果重新序列化为小端字节，供容器读端使用。
pub fn expand(artifact: &[u8]) -> Result<Vec<u8>> {
    Ok(f32_bytes(&decode(artifact)?))
}

/// 解码 artifact 为 f32 权重；任何截断/未知轴/长度不符/证书与载荷不符显式报错。
pub fn decode(artifact: &[u8]) -> Result<Vec<f32>> {
    let (axis, m, n) = head(artifact)?;
    let want = area((m, n));
    let cfg = AXES[axis as usize];
    let mut cur = Cursor::new(&artifact[HEAD..]);
    let out = cfg.decode(&mut cur, (m, n))?;
    if out.len() != want {
        return Err(bad("解码长度与形状不符"));
    }
    Ok(out)
}

/// 只读证书：不解码权重也能拿到秩轴那张证书（其余轴返回 `None`）。
///
/// [`decode`] 本来就会复核证书，这里给的是另一条路：审计与工具只想问「凭什么这么压」时，
/// 不必先落地一张张量。两边读的都是私有的 `rank_head`，因此“证书长什么样”只有一种解析。
pub fn certificate(artifact: &[u8]) -> Result<Option<ledger::Note>> {
    let (axis, _, _) = head(artifact)?;
    if axis != Axis::Rank {
        return Ok(None);
    }
    rank_head(&artifact[HEAD..]).map(|head| Some(head.note))
}

/// 形状判据：`m`、`n` 非零且元素数与乘积一致；不回答权重内容。
fn check_shape(w: &[f32], m: usize, n: usize) -> Result<()> {
    if m == 0 || n == 0 || w.len() != m * n {
        return Err(bad(&format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    Ok(())
}

/// 头判据：长度至少 `HEAD`、魔数匹配；返回轴与形状（乘积溢出即报错，不回绕）。
fn head(artifact: &[u8]) -> Result<(Axis, usize, usize)> {
    if artifact.len() < HEAD || artifact[..4] != MAGIC {
        return Err(bad("artifact 头无效"));
    }
    let axis = Axis::from_code(artifact[4])?;
    let m = u32::from_le_bytes([artifact[5], artifact[6], artifact[7], artifact[8]]) as usize;
    let n = u32::from_le_bytes([artifact[9], artifact[10], artifact[11], artifact[12]]) as usize;
    m.checked_mul(n).ok_or_else(shape_overflow)?;
    Ok((axis, m, n))
}

/// 元素数：`head` 已判过乘积不溢出，这里的乘法不会回绕。
pub(crate) fn area((m, n): (usize, usize)) -> usize {
    m * n
}

/// 形状乘积溢出。
fn shape_overflow() -> Error {
    bad("形状溢出")
}

fn frame(axis: Axis, m: usize, n: usize, payload: Vec<u8>) -> Pick {
    let mut artifact = Vec::with_capacity(HEAD + payload.len());
    artifact.extend_from_slice(&MAGIC);
    artifact.push(axis as u8);
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

pub(crate) fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("axes")
}

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
        other => Err(bad(&format!("未知熵编码标记 {other}"))),
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

/// 低秩载荷：秩由证书给出，证书不通过就不压，证书本身随载荷落盘。
///
/// 这里不再自己挑秩（旧版取 `min(m,n)/2`，与谱隙无关）：[`ledger::certify`] 用一次分解同时给出
/// 受认证的秩与 `L`、`C`，`None` 就是「拒压」——秩轴对这张矩阵不适用，让给别的轴。
///
/// 布局：`note_len | note | left_len | left | right_len | right | rest`。低秩项不落 `L` 本身，
/// 落它的两个因子 `left[m,r]`（`UΣ`）与 `right[r,n]`（`Vᵀ`）——`L` 是整张 `m×n`，存一遍等于没省；
/// 因子只有 `(m+n)·r`，读端用 [`ledger::factor_low`] 这一份算式还原。
///
/// 秩不再是一个裸数字：证书自带秩，并且多带「凭什么」——模式、稳定性、谱隙、扰动、截断误差、
/// 逐位往返。伪造一个更大的秩就得同时伪造谱隙，而谱隙那一条在读端会被载荷重算
/// （[`ledger::audit`]）。
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
///
/// 秩载荷里的每段都走这一处，读端「区间越界」的算式因此只有一份；写侧对应的封装是 [`framed`]。
fn sliced(all: &[u8], at: &mut usize, what: &str) -> Result<std::ops::Range<usize>> {
    let len = word(all, at, what)? as usize;
    let end = at
        .checked_add(len)
        .filter(|&e| e <= all.len())
        .ok_or_else(|| {
            bad(&format!(
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
        return Err(bad(&format!(
            "因子字节 {} != 期望 {}",
            bytes.len(),
            want * 8
        )));
    }
    let (chunks, _) = bytes.as_chunks::<8>();
    Ok(chunks.iter().map(|c| f64::from_le_bytes(*c)).collect())
}

#[cfg(test)]
#[path = "../tests/axes.rs"]
mod tests;
