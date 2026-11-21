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

use crate::axes_codec::{
    bytes_to_f32, decode_dict, decode_quant, decode_sparse, dict_payload, f32_bytes, pack_bytes,
    payload_cap, quant_payload, sparse_payload, unpack_bytes,
};
use crate::axes_rank::{decode_rank, rank_head, rank_payload};
use crate::leb::Cursor;
use crate::ledger;

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

#[cfg(test)]
#[path = "../tests/axes.rs"]
mod tests;
