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
use crate::{ledger, rans};

/// artifact 魔数（v2：`rank` 载荷自带证书，v1 的纯秩数字不再可读）。
const MAGIC: [u8; 4] = *b"AX02";

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
/// 不必先落地一张张量。两边读的都是 [`rank_head`]，因此“证书长什么样”只有一种解析。
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
fn area((m, n): (usize, usize)) -> usize {
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

/// 标记位：0 = store（原样），1 = rans 帧。
const STORE_MARK: u8 = 0;
const RANS_MARK: u8 = 1;

fn pack_bytes(src: &[u8]) -> Vec<u8> {
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
fn payload_cap(want: usize) -> usize {
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

fn unpack_bytes(cur: &mut Cursor<'_>, cap: usize) -> Result<Vec<u8>> {
    let mark = cur.read_u8()?;
    unpack_body(mark, cur.remaining(), cap)
}

/// 对一段带标记的完整帧解码（用于 rank 里已知边界的 low/rest）。
fn unpack_marked(frame: &[u8], cap: usize) -> Result<Vec<u8>> {
    let (&mark, body) = frame.split_first().ok_or_else(|| bad("空熵帧"))?;
    unpack_body(mark, body, cap)
}

/// 唯一值排序表 + 等长索引序列：排序去重后二分定位，零关联容器依赖。
fn dict_payload(w: &[f32]) -> Option<Vec<u8>> {
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

fn decode_dict(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
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

fn quant_payload(w: &[f32]) -> Option<Vec<u8>> {
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

fn decode_quant(cur: &mut Cursor<'_>, want: usize) -> Result<Vec<f32>> {
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

/// 低秩载荷：秩由证书给出，证书不通过就不压，证书本身随载荷落盘。
///
/// 这里不再自己挑秩（旧版取 `min(m,n)/2`，与谱隙无关）：[`ledger::certify`] 用一次分解同时给出
/// 受认证的秩与 `L`、`C`，`None` 就是「拒压」——秩轴对这张矩阵不适用，让给别的轴。
///
/// 布局：`note_len | note | low_len | low | rest`。秩不再是一个裸数字：证书自带秩，并且多带
/// 「凭什么」——模式、稳定性、谱隙、扰动、截断误差、逐位往返。伪造一个更大的秩就得同时伪造
/// 谱隙，而谱隙那一条在读端会被载荷重算（[`ledger::audit`]）。
fn rank_payload(w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    let cert = ledger::certify(w, m, n).ok().flatten()?;
    let note = cert.note.to_bytes();
    let low = pack_bytes(&f32_bytes(&cert.low));
    let mut p = u32::try_from(note.len()).ok()?.to_le_bytes().to_vec();
    p.extend(note);
    p.extend_from_slice(&(low.len() as u32).to_le_bytes());
    p.extend(low);
    p.extend(pack_bytes(&f32_bytes(&cert.rest)));
    Some(p)
}

fn decode_rank(cur: &mut Cursor<'_>, (m, n): (usize, usize)) -> Result<Vec<f32>> {
    let all = cur.remaining();
    let head = rank_head(all)?;
    let (low, rest) = rank_frames(all, head.at, head.low_len)?;
    let want = area((m, n));
    let cap = want.saturating_mul(4);
    let low = rank_part(low, cap, want)?;
    let rest = rank_part(rest, cap, want)?;
    // 证书与它认证的两段对质：这一行才是「可证」——读到的载荷自己能把它重算一遍，
    // 而不是要读者相信写端曾经的声称。
    ledger::audit(&head.note, m, n, &low, &rest)?;
    Ok(low.iter().zip(&rest).map(|(a, b)| a + b).collect())
}

/// 秩载荷的头：证书本体、`low` 段长度，以及 `low` 段在载荷内的起点。
struct RankHead {
    note: ledger::Note,
    low_len: usize,
    /// `low` 段起点（证书长度前缀与证书本体之后）。
    at: usize,
}

/// rank 头判据：证书长度前缀、证书本体（必须恰好读完）、`low` 段长度。
fn rank_head(all: &[u8]) -> Result<RankHead> {
    let mut at = 0;
    let len = word(all, &mut at, "rank 证书长度截断")? as usize;
    let bytes = all
        .get(at..)
        .and_then(|tail| tail.get(..len))
        .ok_or_else(|| bad("rank 证书截断"))?;
    at += len;
    let mut cur = Cursor::new(bytes);
    let note = ledger::Note::decode(&mut cur)?;
    if !cur.remaining().is_empty() {
        return Err(bad("rank 证书有多余字节"));
    }
    let low_len = word(all, &mut at, "rank 头截断")? as usize;
    Ok(RankHead { note, low_len, at })
}

/// 低秩两段：`low` 由头部的长度界定，`rest` 一直取到末尾。
fn rank_frames(all: &[u8], at: usize, low_len: usize) -> Result<(&[u8], &[u8])> {
    all.get(at..)
        .and_then(|tail| tail.split_at_checked(low_len))
        .ok_or_else(|| bad("rank low 截断"))
}

/// 解一段带标记的低秩分量。
fn rank_part(frame: &[u8], cap: usize, want: usize) -> Result<Vec<f32>> {
    let bytes = unpack_marked(frame, cap)?;
    bytes_to_f32(&bytes, want)
}

/// 从 `buf` 的 `at` 处取 `N` 字节并推进偏移；越界用 `missing` 文案报错。
fn take<const N: usize>(buf: &[u8], at: &mut usize, missing: &str) -> Result<[u8; N]> {
    let chunk = buf.get(*at..).and_then(<[u8]>::first_chunk::<N>);
    let chunk = chunk.ok_or_else(|| bad(missing))?;
    *at += N;
    Ok(*chunk)
}

/// 读一个小端 `u32`。
fn word(buf: &[u8], at: &mut usize, missing: &str) -> Result<u32> {
    take::<4>(buf, at, missing).map(u32::from_le_bytes)
}

fn f32_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8], want: usize) -> Result<Vec<f32>> {
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

    /// 有谱隙的矩阵：秩轴产出载荷，载荷自带秩与残差、能逐位还原。
    ///
    /// 这里不假设"秩轴一定在五轴里胜出"——显式残差是每个元素 4 字节，只有在矩阵**精确**低秩
    /// （残差全零）时才比原字节便宜，那是结构性质而不是门控性质。门控性质只保证一件事：证书
    /// 不过就没有这个候选。
    #[test]
    fn a_certified_matrix_yields_a_rank_payload_that_roundtrips() {
        // `w[i][j] = 64·i + j`：秩 2、值两两不同，且 f32 精确。
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();
        let payload = rank_payload(&w, 64, 64).expect("谱隙存在，秩轴必须给出候选");
        let pick = frame(Axis::Rank, 64, 64, payload);
        let got = decode(&pick.artifact).expect("解码");
        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// 没有谱隙的矩阵不存在可信的秩：证书拒压，秩轴整轴不参与（不是压出一个近似值）。
    #[test]
    fn a_matrix_without_a_spectral_gap_skips_the_rank_axis() {
        let zeros = vec![0.0f32; 4096];
        assert!(
            rank_payload(&zeros, 64, 64).is_none(),
            "σ 全相等就没有可判秩"
        );
        assert_ne!(search(&zeros, 64, 64).expect("搜索").axis, Axis::Rank);
        // 单行/单列同理：`1 <= r < min(m,n)` 无解。
        let row: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        assert!(rank_payload(&row, 1, 64).is_none());
        assert_ne!(search(&row, 1, 64).expect("搜索").axis, Axis::Rank);
        let got = decode(&search(&zeros, 64, 64).expect("搜索").artifact).expect("解码");
        assert!(got.iter().all(|x| x.to_bits() == 0.0f32.to_bits()));
    }

    /// 证书随载荷落盘：不解码权重也能读到它，读到的就是写端那一张（不是重新算的）。
    #[test]
    fn the_rank_payload_carries_its_certificate() {
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();
        let cert = ledger::certify(&w, 64, 64)
            .expect("认证")
            .expect("谱隙存在");
        let artifact = frame(Axis::Rank, 64, 64, rank_payload(&w, 64, 64).expect("候选")).artifact;
        let note = certificate(&artifact).expect("解析").expect("秩轴");
        assert_eq!(note, cert.note, "落盘的是认证那一次，不是读端重算的另一张");
        // 其余轴没有证书这一说：返回 `None` 而不是编一张空证书。
        let raw = frame(Axis::Raw, 64, 64, pack_bytes(&f32_bytes(&w))).artifact;
        assert!(certificate(&raw).expect("解析").is_none());
    }

    /// 证书被改写就解不出来：读端把截断误差当断言，而不是相信写端的声称。
    ///
    /// 伪造的两处都落在「证书 + 载荷」两方对质能查出、而单看证书自身 `check` 查不出的位置：
    /// 改写 `decomp`（`check` 只看它有限且与 `lossless` 一致），把 `stable` 翻成 `Fail`
    /// （`check` 允许 `Fail`，但一条 `Fail` 的证书根本不该被落盘）。
    #[test]
    fn a_tampered_certificate_is_rejected_on_read() {
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();
        let cert = ledger::certify(&w, 64, 64)
            .expect("认证")
            .expect("谱隙存在");
        let low = pack_bytes(&f32_bytes(&cert.low));
        let rest = pack_bytes(&f32_bytes(&cert.rest));
        let framed = |note: Vec<u8>| {
            let mut p = u32::try_from(note.len())
                .expect("证书长度")
                .to_le_bytes()
                .to_vec();
            p.extend(note);
            p.extend_from_slice(&(low.len() as u32).to_le_bytes());
            p.extend(low.clone());
            p.extend(rest.clone());
            frame(Axis::Rank, 64, 64, p).artifact
        };

        let honest = framed(cert.note.to_bytes());
        let got = decode(&honest).expect("原证书可读");
        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));

        // 按字段伪造而不是改字节：`to_bytes` 的长度前缀是变长编码，偏移量不该在测试里写死。
        // 这张矩阵精确低秩，`decomp` 本就是 0（乘任何系数都不动），所以按加法伪造。
        let mut wrong_error = cert.note.clone();
        wrong_error.decomp += 1.0;
        assert!(
            decode(&framed(wrong_error.to_bytes())).is_err(),
            "截断误差被改写必须被拒"
        );

        // 稳定性翻成 `Fail`：`Fail` 本身合法，但一条没放行的证书不该出现在载荷里。
        let mut not_passed = cert.note.clone();
        not_passed.stable = ledger::Stable::Fail;
        assert!(
            decode(&framed(not_passed.to_bytes())).is_err(),
            "没放行的证书不能落盘"
        );
    }

    /// 旧布局（秩裸数字在最前）不再是合法载荷：第一段被读成证书长度，随后的字节读不成证书。
    #[test]
    fn the_rank_payload_requires_a_certificate() {
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();
        let cert = ledger::certify(&w, 64, 64)
            .expect("认证")
            .expect("谱隙存在");
        let low = pack_bytes(&f32_bytes(&cert.low));
        let mut old = (cert.note.rank).to_le_bytes().to_vec();
        old.extend_from_slice(&(low.len() as u32).to_le_bytes());
        old.extend(low);
        old.extend(pack_bytes(&f32_bytes(&cert.rest)));
        assert!(decode(&frame(Axis::Rank, 64, 64, old).artifact).is_err());
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
