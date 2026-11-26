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
    let w = bytes_to_f32(data, checked_area(m, n)?)?;
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

/// 形状判据：维数可写入 `u32`、均非零、乘积不溢出且与元素数一致。
fn check_shape(w: &[f32], m: usize, n: usize) -> Result<()> {
    let want = checked_area(m, n)?;
    check_dim_bounds(m, n)?;
    if [m, n].contains(&0) || w.len() != want {
        return Err(bad(format!("形状不合法 m={m} n={n} len={}", w.len())));
    }
    Ok(())
}

/// 两个维度都必须落在线上 `u32` 范围内。
fn check_dim_bounds(m: usize, n: usize) -> Result<()> {
    if u32::try_from(m).is_err() || u32::try_from(n).is_err() {
        return Err(bad("形状维度超过线上 u32 范围"));
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
    checked_area(m, n)?;
    Ok((m, n))
}

/// 安全计算形状面积，供字节压缩与头部解析共用。
fn checked_area(m: usize, n: usize) -> Result<usize> {
    m.checked_mul(n).ok_or_else(shape_overflow)
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
fn sound(spec: &AxisSpec, payload: &[u8], shape: (usize, usize), w: &[f32]) -> bool {
    (spec.decode)(&mut Cursor::new(payload), shape).is_ok_and(|got| {
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
mod tests {
    //! `axes` 模块的单元测试。
    use super::*;
    use crate::axes::{f64_bytes, framed};

    testkit::test_cases! {
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

    fn dict_wins_on_low_cardinality() {
        // 200 个互不相同、尾数随机的 f32 周期排布：同一尺度网格上量化不着、64×64 上秩轴要给

        // 两万个字节的因子，只有字典帧（200 个数 + 每元素 1 字节）真正更小。

        let vals: Vec<f32> = (0..200u32)
            .map(|i| f32::from_bits(0x3f80_0000 | (i.wrapping_mul(2_654_435_761) & 0x007f_ffff)))
            .collect();

        let w: Vec<f32> = (0..4096).map(|i| vals[(i * 37) % 200]).collect();

        let pick = search(&w, 64, 64).expect("搜索");

        assert_eq!(pick.axis, Axis::Dict);

        assert!(pick.len() < w.len() * 4);
    }

    fn dict_payload_has_stable_wire_bytes() {
        let weights = [1.0, -0.0, 2.5, 1.0, -0.0, 2.5, 1.0, 2.5];
        let payload = dict_payload(&weights).expect("低基数输入必须可用字典");
        assert_eq!(
            payload,
            [
                0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00,
                0x00, 0x20, 0x40, 0x00, 0x00, 0x00, 0x80, 0x00, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00,
                0x01,
            ]
        );
    }

    fn dictionary_idxs_roundtrip_at_every_width_boundary() {
        for unique in [1usize, 256, 257, 65_536, 65_537] {
            let weights: Vec<f32> = (0..unique * 2)
                .map(|idx| f32::from_bits((idx % unique) as u32))
                .collect();
            let payload = dict_payload(&weights).expect("重复输入必须可用字典");
            let decoded = decode_dict(&mut Cursor::new(&payload), weights.len()).expect("字典解码");
            assert!(
                decoded
                    .iter()
                    .zip(&weights)
                    .all(|(left, right)| left.to_bits() == right.to_bits()),
                "唯一值数量 {unique} 的索引宽度往返失败"
            );
        }
    }

    fn dictionary_rejects_an_unknown_idx_width() {
        let mut bytes = vec![0]; // store 标记
        for word in [3u32, 1, 0, 0] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let error = decode_dict(&mut Cursor::new(&bytes), 1).expect_err("宽度 3 不合法");
        assert!(error.to_string().contains("索引宽度 3"), "{error}");
    }

    fn overflowing_or_unrepresentable_shapes_are_errors() {
        let error = search(&[], usize::MAX, 2).expect_err("形状乘法不得回绕");
        assert!(error.to_string().contains("形状溢出"), "{error}");
        let error = compress(&[], usize::MAX, 2).expect_err("字节入口同样不得回绕");
        assert!(error.to_string().contains("形状溢出"), "{error}");

        #[cfg(target_pointer_width = "64")]
        {
            let error = search(&[], u32::MAX as usize + 1, 1).expect_err("维度必须能写入 u32");
            assert!(error.to_string().contains("u32"), "{error}");
        }
    }

    fn sparse_roundtrips_zero_heavy() {
        let mut w = vec![0.0f32; 4096];

        for i in (0..4096).step_by(97) {
            w[i] = (i as f32).sqrt();
        }

        let pick = search(&w, 64, 64).expect("搜索");

        let got = decode(&pick.artifact).expect("解码");

        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

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
    fn a_certified_matrix_yields_a_rank_payload_that_roundtrips() {
        // `w[i][j] = 64·i + j`：秩 2、值两两不同，且 f32 精确。

        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

        let payload = rank_payload(&w, 64, 64).expect("谱隙存在，秩轴必须给出候选");

        let pick = frame(Axis::Rank, 64, 64, payload);

        let got = decode(&pick.artifact).expect("解码");

        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// 没有谱隙的矩阵不存在可信的秩：证书拒压，秩轴整轴不参与（不是压出一个近似值）。
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
    fn a_tampered_certificate_is_rejected_on_read() {
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

        let cert = ledger::certify(&w, 64, 64)
            .expect("认证")
            .expect("谱隙存在");

        let left = pack_bytes(&f64_bytes(&cert.left));

        let right = pack_bytes(&f64_bytes(&cert.right));

        let rest = pack_bytes(&f32_bytes(&cert.rest));

        let framed = |note: Vec<u8>| {
            let mut p = framed(&note).expect("证书段");

            p.extend(framed(&left).expect("左因子段"));

            p.extend(framed(&right).expect("右因子段"));

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

        let mut not_passed = cert.note;

        not_passed.stable = ledger::Stable::Fail;

        assert!(
            decode(&framed(not_passed.to_bytes())).is_err(),
            "没放行的证书不能落盘"
        );
    }

    /// 旧布局（秩裸数字在最前）不再是合法载荷：第一段被读成证书长度，随后的字节读不成证书。
    fn the_rank_payload_requires_a_certificate() {
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

        let cert = ledger::certify(&w, 64, 64)
            .expect("认证")
            .expect("谱隙存在");

        // v3 布局：证书之后是一整张 `low`，它的长度前缀此刻被读成左因子长度。

        let low = pack_bytes(&f32_bytes(&cert.low));

        let mut old = framed(&cert.note.to_bytes()).expect("证书段");

        old.extend_from_slice(&(low.len() as u32).to_le_bytes());

        old.extend(low);

        old.extend(pack_bytes(&f32_bytes(&cert.rest)));

        assert!(decode(&frame(Axis::Rank, 64, 64, old).artifact).is_err());
    }

    fn malformed_artifacts_are_errors() {
        assert!(decode(b"").is_err());

        assert!(decode(b"AX01").is_err());

        assert!(decode(b"BAD!000000000").is_err());
    }

    fn bad_shape_is_an_error() {
        assert!(search(&[1.0], 3, 1).is_err());
    }

    fn an_empty_axis_set_keeps_only_the_verified_raw_baseline() {
        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();

        let pick = search_with_axes(&w, 64, 64, AxisSet::empty()).expect("搜索");

        assert_eq!(pick.axis, Axis::Raw);
        let got = decode(&pick.artifact).expect("解码");
        assert!(got.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    fn axis_sets_are_composable_and_rank_can_be_disabled() {
        assert_eq!(AxisSet::default(), AxisSet::ALL);
        assert!(AxisSet::WITHOUT_RANK.contains(Axis::Dict));
        assert!(!AxisSet::WITHOUT_RANK.contains(Axis::Rank));

        let w: Vec<f32> = (0..4096).map(|k| k as f32).collect();
        let pick = search_with_axes(&w, 64, 64, AxisSet::WITHOUT_RANK).expect("搜索");
        assert_ne!(pick.axis, Axis::Rank);

        let grid: Vec<f32> = (0..4096)
            .map(|i| 1.0 + ((i * 37) % 16) as f32 * 0.25)
            .collect();
        let only_quant = AxisSet::empty().with(Axis::Quant);
        let pick = search_with_axes(&grid, 64, 64, only_quant).expect("搜索");
        assert_eq!(pick.axis, Axis::Quant);
    }
    }
}
