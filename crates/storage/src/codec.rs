//! 快照正文的读取：游标与错误词汇、条目表编解码（名字前后缀共享、布局重复判据、偏移增量）。

use std::ops::Range;

use diagnostic::{Code, Error, Result};
use tensor::DType;
use wire::leb::Cursor as WireCursor;

use crate::ManifestEntry;
use crate::snapshot::PREALLOC;

/// 解码失败的上下文：错误只由这一层产生，排查时不必再猜是哪一段。
pub(crate) const CTX: &str = "storage::snapshot::Snapshot::decode";

/// 快照格式的元素类型标签：闭集 5 种（`tag` 与 `dtype` 共用同一张表）。
pub(crate) const TYPES: [(DType, u8); 5] = [
    (DType::F32, 0),
    (DType::F16, 1),
    (DType::BF16, 2),
    (DType::I8, 3),
    (DType::U4, 4),
];

/// 取一段字节；不足时按起始偏移报截断。
pub(crate) fn need(bytes: &[u8], span: Range<usize>, offset: usize) -> Result<&[u8]> {
    match bytes.get(span) {
        Some(slice) => Ok(slice),
        None => Err(truncated(offset)),
    }
}

/// 检查点在给定偏移处被截断。
pub(crate) fn truncated(offset: usize) -> Error {
    Error::new(Code::Invalid, format!("检查点在偏移 {offset} 处截断")).with(CTX)
}

/// 一条解码期非法输入；上下文固定在快照层。
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(CTX)
}

/// 检查点与手头的条目表不一致：只由 [`super::Snapshot::check_weights`] 产生。
pub(crate) fn mismatch(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("storage::snapshot::Snapshot::check_weights")
}

/// 第 `idx` 个条目的错误帧：一个几千条目的检查点里，"第 3 个条目"比"某个偏移"有用得多。
pub(crate) fn frame(idx: usize) -> impl Fn(Error) -> Error {
    move |error| error.with(format!("第 {idx} 个权重条目"))
}

/// 第 `idx` 个条目的非法输入：`frame` 与 `invalid` 的现成组合，省掉调用处的双层括号。
pub(crate) fn invalid_at(idx: usize, message: impl Into<String>) -> Error {
    frame(idx)(invalid(message))
}

/// 元素类型在快照字节里的标签：闭集外的类型显式报错。
pub(crate) fn tag(dtype: DType) -> Result<u8> {
    TYPES
        .iter()
        .find(|(known, _)| *known == dtype)
        .map(|(_, byte)| *byte)
        .ok_or_else(|| invalid(format!("快照格式不支持 {dtype:?} 元素类型")))
}

/// 标签 → 元素类型。
fn dtype(tag: u8) -> Result<DType> {
    TYPES
        .iter()
        .find(|(_, byte)| *byte == tag)
        .map(|(known, _)| *known)
        .ok_or_else(|| invalid(format!("未知的元素类型标签 {tag}")))
}

/// 逐字节读取器：在 `wire::Cursor` 之上保留快照侧的字段上下文与错误词汇。
pub(crate) struct Cursor<'a> {
    inner: WireCursor<'a>,
    /// 当前消费偏移；测试用它给名字块切片区间定位。
    pub(crate) offset: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], offset: usize) -> Result<Self> {
        let mut inner = WireCursor::new(bytes);
        inner.take(offset).map_err(|_| truncated(offset))?;
        Ok(Self { inner, offset })
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let slice = self
            .inner
            .take(len)
            .map_err(|_| truncated(self.inner.pos()))?;
        self.offset = self.inner.pos();
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        let byte = self
            .inner
            .read_u8()
            .map_err(|_| truncated(self.inner.pos()))?;
        self.offset = self.inner.pos();
        Ok(byte)
    }

    /// 读一个 LEB128 变长整数；非规范写法（如 `0x87 0x00`）或超出 `u64` 宽度都拒绝。
    pub(crate) fn varint(&mut self) -> Result<u64> {
        let value = self.inner.read::<u64>().map_err(|error| {
            if error.message().starts_with("字节不足") {
                truncated(self.inner.pos())
            } else {
                invalid("检查点里的整数非规范写法（含前导零字节）或超出 u64 上限")
            }
        })?;
        self.offset = self.inner.pos();
        Ok(value)
    }

    /// 读一个不能超过 `usize` 的计数；`what` 是字段名，用来把错误指到具体位置。
    pub(crate) fn count(&mut self, what: &str) -> Result<usize> {
        let value = self.varint()?;
        usize::try_from(value)
            .map_err(|_| invalid(format!("检查点的{what} {value} 超出 usize 上限")))
    }

    pub(crate) fn remaining(&self) -> usize {
        self.inner.remaining().len()
    }
}

/// 条目表：先按最小体积对账（伪造的计数撑不大内存），再逐条重建。
pub(crate) fn read_entries<E: ManifestEntry>(
    cursor: &mut Cursor<'_>,
    count: usize,
    min_entry: usize,
    prealloc: usize,
) -> Result<Vec<E>> {
    if count > cursor.remaining() / min_entry {
        return Err(invalid(format!(
            "检查点声明了 {count} 个权重条目，但剩余 {} 字节最多容纳 {} 个",
            cursor.remaining(),
            cursor.remaining() / min_entry
        )));
    }
    let mut entries = Vec::with_capacity(count.min(prealloc));
    let mut rebuild = Vec::new();
    for idx in 0..count {
        let entry = entry(cursor, idx, entries.last(), &mut rebuild)?;
        entries.push(entry);
    }
    // 尾部余量归正文对账：条目表之后还有矩状态段，不能在这里把余量判成多余字节。
    Ok(entries)
}

/// 一个条目：名字、布局与偏移；上一条同时提供名字共享与布局重复的基准。
fn entry<E: ManifestEntry>(
    cursor: &mut Cursor<'_>,
    idx: usize,
    previous: Option<&E>,
    rebuild: &mut Vec<u8>,
) -> Result<E> {
    let previous_name = previous.map_or(&[][..], |entry| entry.name().as_bytes());
    let name = rebuild_name(cursor, idx, previous_name, rebuild)?;
    let (dtype, dims) = layout(cursor, idx, previous)?;
    let offset = offset(cursor, idx, previous)?;
    Ok(E::from_parts(name.to_string(), dtype, dims, offset))
}

/// 按上一条名字拼回整串，并拒掉"共享不是最长"的写法（否则同一份快照有多种字节形态）。
fn rebuild_name<'b>(
    cursor: &mut Cursor<'_>,
    idx: usize,
    previous: &[u8],
    out: &'b mut Vec<u8>,
) -> Result<&'b str> {
    let (front, back, len) = name_lengths(cursor, idx, previous.len())?;
    let middle = cursor.take(len).map_err(frame(idx))?;
    out.clear();
    out.extend_from_slice(&previous[..front]);
    out.extend_from_slice(middle);
    out.extend_from_slice(&previous[previous.len() - back..]);
    let want_front = common_prefix(previous, out);
    let want_back = common_suffix(&previous[want_front..], &out[want_front..]);
    if (front, back) != (want_front, want_back) {
        let message = format!(
            "条目名的共享不是最长写法（写了前 {front} 后 {back}，\
             应为前 {want_front} 后 {want_back}）"
        );
        return Err(invalid_at(idx, message));
    }
    std::str::from_utf8(out).map_err(|error| invalid_at(idx, format!("条目名不是 UTF-8：{error}")))
}

/// 共享长度得分得开：前后两段不能重叠，也不能超过上一条名字。
fn fits_share(shared: usize, front: usize, back: usize, idx: usize) -> Result<()> {
    if front > shared || back > shared - front {
        let message = format!("上一条目名只有 {shared} 字节，却要共享前 {front} 后 {back} 字节");
        return Err(invalid_at(idx, message));
    }
    Ok(())
}

/// 名字的三段长度：共享前后缀必须分得开，中段长度随后读出。
fn name_lengths(
    cursor: &mut Cursor<'_>,
    idx: usize,
    shared: usize,
) -> Result<(usize, usize, usize)> {
    let front = cursor.count("名字的共享前缀").map_err(frame(idx))?;
    let back = cursor.count("名字的共享后缀").map_err(frame(idx))?;
    fits_share(shared, front, back, idx)?;
    let len = cursor.count("条目名中段长度").map_err(frame(idx))?;
    Ok((front, back, len))
}

/// 布局标签：`0` 表示与上一条同型同形，否则是 `秩 + 1` 加类型与各维度。
fn layout<E: ManifestEntry>(
    cursor: &mut Cursor<'_>,
    idx: usize,
    previous: Option<&E>,
) -> Result<(DType, Vec<usize>)> {
    let label = cursor.count("布局标签").map_err(frame(idx))?;
    if label == 0 {
        return repeated(previous, idx);
    }
    fresh(cursor, idx, label - 1, previous)
}

/// 与上一条同型同形；第一个条目没有上一条。
fn repeated<E: ManifestEntry>(previous: Option<&E>, idx: usize) -> Result<(DType, Vec<usize>)> {
    let Some(previous) = previous else {
        return Err(invalid_at(
            idx,
            "第一个条目不能与上一条共享布局（还没有上一条）",
        ));
    };
    Ok((previous.dtype(), previous.dims().to_vec()))
}

/// 重写布局：类型标签与各维度。
fn fresh<E: ManifestEntry>(
    cursor: &mut Cursor<'_>,
    idx: usize,
    rank: usize,
    previous: Option<&E>,
) -> Result<(DType, Vec<usize>)> {
    let dtype = dtype(cursor.u8().map_err(frame(idx))?)?;
    let dims = dims(cursor, idx, rank)?;
    canonical(previous, dtype, &dims, idx).map(|()| (dtype, dims))
}

/// 与上一条同型同形却重写了布局是非规范写法：同一份快照不能既有"重复"又有"重写"。
fn canonical<E: ManifestEntry>(
    previous: Option<&E>,
    dtype: DType,
    dims: &[usize],
    idx: usize,
) -> Result<()> {
    if let Some(previous) = previous
        && previous.dtype() == dtype
        && previous.dims() == dims
    {
        return Err(invalid_at(idx, "布局与上一条相同却没有写成重复（标签 0）"));
    }
    Ok(())
}

/// 形状的各维度：预分到预分配上限为止，伪造的秩也只能骗到一块固定大小的内存。
fn dims(cursor: &mut Cursor<'_>, idx: usize, rank: usize) -> Result<Vec<usize>> {
    let mut dims = Vec::with_capacity(rank.min(PREALLOC));
    for _ in 0..rank {
        dims.push(cursor.count("维度").map_err(frame(idx))?);
    }
    Ok(dims)
}

/// 偏移按增量存：上一条的偏移加增量，溢出就是坏数据。
fn offset<E: ManifestEntry>(
    cursor: &mut Cursor<'_>,
    idx: usize,
    previous: Option<&E>,
) -> Result<usize> {
    let previous = previous.map_or(0, ManifestEntry::offset);
    let delta = cursor.count("偏移增量").map_err(frame(idx))?;
    let overflow = || invalid_at(idx, format!("偏移 {previous} + {delta} 溢出"));
    previous.checked_add(delta).ok_or_else(overflow)
}

/// 两条名字共有的最大字节前缀（按字节比：共享长度最终要能直接切出字节区间）。
pub(crate) fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut at = 0;
    while at < limit && left[at] == right[at] {
        at += 1;
    }
    at
}

/// 两条名字共有的最大后缀；调用方传进来的总是已经被前缀吃掉的那两段尾巴。
pub(crate) fn common_suffix(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut at = 0;
    while at < limit && left[left.len() - 1 - at] == right[right.len() - 1 - at] {
        at += 1;
    }
    at
}

/// 第 `idx` 个条目的第一处不同；两者一致时为 `None`。
pub(crate) fn difference<E: ManifestEntry>(idx: usize, want: &E, have: &E) -> Option<String> {
    let name = want.name();
    if name != have.name() {
        let other = have.name();
        return Some(format!(
            "第 {idx} 个条目：检查点里是 `{name}`，权重文件里是 `{other}`"
        ));
    }
    if want.dtype() != have.dtype() {
        let (a, b) = (want.dtype(), have.dtype());
        return Some(format!(
            "条目 `{name}` 的元素类型：检查点里是 {a:?}，权重文件里是 {b:?}"
        ));
    }
    if want.dims() != have.dims() {
        let (a, b) = (want.dims(), have.dims());
        return Some(format!(
            "条目 `{name}` 的形状：检查点里是 {a:?}，权重文件里是 {b:?}"
        ));
    }
    if want.offset() != have.offset() {
        let (a, b) = (want.offset(), have.offset());
        return Some(format!(
            "条目 `{name}` 的偏移：检查点里是 {a}，权重文件里是 {b}"
        ));
    }
    None
}
