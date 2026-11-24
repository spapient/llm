//! 五轴自适应选码（配置化）：`raw / dict / sparse / quant / rank` 五个候选，每个候选只有在解码后
//! 与原始位模式逐元素一致（`to_bits` 相等）时才被采纳，最终在通过验证的候选里取字节最小者。

use crate::bad;
use diagnostic::{Error, Result};

use crate::ledger;
use crate::svd;
use storage::leb::Cursor;

use crate::codec::{
    bytes_to_f32, decode_dict, decode_quant, decode_sparse, dict_payload, f32_bytes, from_le_bytes,
    pack_bytes, payload_cap, quant_payload, sparse_payload, to_le_bytes, unpack_bytes,
    unpack_marked, word,
};
use storage::leb::u32_at;

/// artifact 魔数（v4：`rank` 载荷改存低秩项的两个因子而非整张 `L`，v1–v3 不再可读）。
const MAGIC: [u8; 4] = *b"AX04";

/// 固定头长度：magic(4) + axis(1) + m(4) + n(4)。
const HEAD: usize = 13;

closed_enum! { @coded
    /// 候选轴（闭集，码位稳定）。
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
}

/// 五轴搜索的候选集合。`Raw` 始终作为逐位基线参与计算，集合控制的是其余四种候选；
/// 因此 [`AxisSet::empty`] 表示"只做 rANS 基线"，而不是连基线也跳过。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisSet(u8);

impl Default for AxisSet {
    fn default() -> Self {
        Self::ALL
    }
}

impl AxisSet {
    /// 全部候选（默认，与既有五轴搜索一致）。
    pub const ALL: Self = Self(0b1_1111);
    /// 不跑 SVD 的快速集合：字典、稀疏与精确网格量化。
    pub const WITHOUT_RANK: Self = Self::ALL.without(Axis::Rank);

    /// 只有 `Raw` 基线，不尝试其他候选。
    pub const fn empty() -> Self {
        Self(0)
    }

    /// 加入一个候选轴。
    #[must_use]
    pub const fn with(mut self, axis: Axis) -> Self {
        self.0 |= 1 << axis.code();
        self
    }

    /// 排除一个候选轴。
    #[must_use]
    pub const fn without(mut self, axis: Axis) -> Self {
        self.0 &= !(1 << axis.code());
        self
    }

    /// 本集合是否启用某个轴。
    #[must_use]
    pub const fn contains(self, axis: Axis) -> bool {
        self.0 & (1 << axis.code()) != 0
    }
}

/// 一轴的声明式能力表：编码与解码同源注册，调用点不需要逐个感知具体轴。
#[derive(Clone, Copy)]
struct AxisSpec {
    axis: Axis,
    encode: fn(&[f32], usize, usize) -> Option<Vec<u8>>,
    decode: fn(&mut Cursor<'_>, (usize, usize)) -> Result<Vec<f32>>,
}

/// 轴闭集：下标即 [`Axis::code`]；新增轴只在这里登记一次。
const AXES: [AxisSpec; 5] = [
    AxisSpec {
        axis: Axis::Raw,
        encode: |w, _, _| Some(pack_bytes(&f32_bytes(w))),
        decode: |cur, shape| {
            let want = area(shape);
            bytes_to_f32(&unpack_bytes(cur, payload_cap(want))?, want)
        },
    },
    AxisSpec {
        axis: Axis::Dict,
        encode: |w, _, _| dict_payload(w),
        decode: |cur, shape| decode_dict(cur, area(shape)),
    },
    AxisSpec {
        axis: Axis::Sparse,
        encode: |w, _, _| sparse_payload(w),
        decode: |cur, shape| decode_sparse(cur, area(shape)),
    },
    AxisSpec {
        axis: Axis::Quant,
        encode: |w, _, _| quant_payload(w),
        decode: |cur, shape| decode_quant(cur, area(shape)),
    },
    AxisSpec {
        axis: Axis::Rank,
        encode: rank_payload,
        decode: decode_rank,
    },
];

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
    search_with_axes(w, m, n, AxisSet::ALL)
}

/// 按候选集合搜索：与 [`search`] 同一判据，但只尝试 `axes` 启用的轴。
/// `Raw` 基线始终计算并验证，因此禁用 `Rank` 可以完全跳过 SVD。
pub fn search_with_axes(w: &[f32], m: usize, n: usize, axes: AxisSet) -> Result<Pick> {
    check_shape(w, m, n)?;
    let shape = (m, n);
    // `Raw` 恒可用，是基线；基线必须先自身逐位成立，破了就是压缩器实现 bug，报错而不是 panic。
    let base_payload = pack_bytes(&f32_bytes(w));
    if !sound(&AXES[0], &base_payload, shape, w) {
        return Err(bad("raw 基线没能逐位还原，压缩器自身不自洽"));
    }
    let base = frame(Axis::Raw, m, n, base_payload);
    // 其余四轴只在逐位成立且严格小于基线时才有胜出可能。
    Ok(AXES[1..]
        .iter()
        .filter(|spec| axes.contains(spec.axis))
        .filter_map(|spec| {
            let payload = (spec.encode)(w, m, n)?;
            sound(spec, &payload, shape, w).then(|| frame(spec.axis, m, n, payload))
        })
        .filter(|pick| pick.len() < base.len())
        .min_by_key(Pick::len)
        .unwrap_or(base))
}

/// 字节级无损压缩入口：`data` 是行主序 f32 小端字节（`m*n` 个元素）。仅当五轴 artifact
/// 严格小于原字节时才返回 `Some`，供 `.u` 容器写端在 rANS 之外择优。
pub fn compress(data: &[u8], m: usize, n: usize) -> Result<Option<Vec<u8>>> {
    compress_with_axes(data, m, n, AxisSet::ALL)
}

/// 按候选集合压缩；与 [`compress`] 同一逐位验证和体积判据。
pub fn compress_with_axes(
    data: &[u8],
    m: usize,
    n: usize,
    axes: AxisSet,
) -> Result<Option<Vec<u8>>> {
    let w = bytes_to_f32(data, m * n)?;
    let pick = search_with_axes(&w, m, n, axes)?;
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
    let spec = &AXES[axis as usize];
    let mut cur = Cursor::new(&artifact[HEAD..]);
    let out = (spec.decode)(&mut cur, (m, n))?;
    if out.len() != want {
        return Err(bad("解码长度与形状不符"));
    }
    Ok(out)
}

/// 只读证书：不解码权重也能拿到秩轴那张证书（其余轴返回 `None`）。
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
        return Err(bad(format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    Ok(())
}

/// 头判据：长度至少 `HEAD`、魔数匹配；返回轴与形状（乘积溢出即报错，不回绕）。
fn head(artifact: &[u8]) -> Result<(Axis, usize, usize)> {
    check_head(artifact)?;
    let axis = Axis::from_code(artifact[4])?;
    let (m, n) = shape_of(artifact)?;
    Ok((axis, m, n))
}

/// 头长与魔数：长度不足时不切片，避免在坏数据上 panic。
fn check_head(artifact: &[u8]) -> Result<()> {
    if artifact.len() < HEAD || artifact[..4] != MAGIC {
        return Err(bad("artifact 头无效"));
    }
    Ok(())
}

/// 形状两维：乘积溢出即报错，不回绕。
fn shape_of(artifact: &[u8]) -> Result<(usize, usize)> {
    let m = u32_at(artifact, 5)? as usize;
    let n = u32_at(artifact, 9)? as usize;
    m.checked_mul(n).ok_or_else(shape_overflow)?;
    Ok((m, n))
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
    artifact.push(axis.code());
    artifact.extend_from_slice(&(m as u32).to_le_bytes());
    artifact.extend_from_slice(&(n as u32).to_le_bytes());
    artifact.extend(payload);
    Pick { axis, artifact }
}

/// 载荷逐位还原判据：编码后的载荷直接反解并对质，跳过 `frame → head` 的再解析——
/// 帧头本就是写端刚构造的，读端重解析它并不会让结果更可信，只是多一次分配与越界检查。
fn sound(cfg: &AxisSpec, payload: &[u8], shape: (usize, usize), w: &[f32]) -> bool {
    (cfg.decode)(&mut Cursor::new(payload), shape).is_ok_and(|got| {
        got.len() == w.len() && got.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits())
    })
}

// —— 秩轴载荷：证书 + 两个低秩因子 + 残差的落盘与读回，读端自证（原 `rank` 平铺进 `axes`）——

/// 低秩载荷：秩由证书给出，证书不通过就不压，证书本身随载荷落盘。
pub(crate) fn rank_payload(w: &[f32], m: usize, n: usize) -> Option<Vec<u8>> {
    let cert = ledger::certify(w, m, n).ok().flatten()?;
    let mut p = framed(&cert.note.to_bytes()).ok()?;
    p.extend(framed(&pack_bytes(&f64_bytes(&cert.left))).ok()?);
    p.extend(framed(&pack_bytes(&f64_bytes(&cert.right))).ok()?);
    p.extend(pack_bytes(&f32_bytes(&cert.rest)));
    Some(p)
}

/// 长度前缀 + 本体：两段的边界由长度给出，`rest` 一直取到末尾故不带前缀。
pub(crate) fn framed(body: &[u8]) -> Result<Vec<u8>> {
    let len = u32::try_from(body.len()).map_err(|_| bad("rank 段超长"))?;
    let mut out = len.to_le_bytes().to_vec();
    out.extend_from_slice(body);
    Ok(out)
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

/// 因子 → 低秩项（`f32`）：与写端同一个 [`svd::factor_low`]，读端不另算一遍。
fn low_of(left: &[f64], right: &[f64], (m, n): (usize, usize), rank: usize) -> Vec<f32> {
    svd::factor_low(left, right, (m, n), rank)
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
    to_le_bytes(values)
}

/// 解一段带标记的因子（`f64`）。
fn factor_part(frame: &[u8], want: usize) -> Result<Vec<f64>> {
    let bytes = unpack_marked(frame, payload_cap(want * 2))?;
    from_le_bytes(&bytes, want)
}

#[cfg(test)]
#[path = "../tests/axes.rs"]
mod tests;
