//! 布局算法：(形状, 步长, 起点) → 元素偏移的**唯一**实现，也是全部形状代数的唯一实现。
//!
//! 同一个 [`Tensor`](crate::Tensor) 的秩可以来自编译期（`[usize; D]`）或运行期（`Vec<usize>`），
//! 存储可以是借来的或自持的。下标、可达性、连续性与取行这些算式与秩的来源、元素的类型都无关，
//! 因此收口在这里一份——两套各自实现意味着任何一处溢出 / 越界口径的修改都要记得改两遍，
//! 那正是漂移的来源。
//!
//! 统一的调用形状是 `(dims, steps)` 两个切片；`&[usize; D]` 自动退化成 `&[usize]`。起点偏移
//! `base` 单独传：它是子视图零拷贝的全部秘密——缓冲不动，只把起点前移。
//!
//! # 形状代数
//!
//! 除构造期 [`check`] 与查询 [`offset`] / [`is_contiguous`] / [`row`] 外，这里有六个变换，全是
//! O(轴数) 的元数据重排，一个元素都不搬：
//!
//! | 变换 | 秩 | 手法 |
//! |---|---|---|
//! | [`narrow`] | 不变 | 起点前移，轴长收窄 |
//! | [`select`] | -1 | 起点前移，该轴去掉 |
//! | [`squeeze`] | -1 | 该轴长为 1 时去掉，起点不动 |
//! | [`unsqueeze`] | +1 | 插入长为 1、步长为 0 的轴 |
//! | [`flatten`] | 变短 | 段内行主序紧凑的一段压成一个轴 |
//!
//! 后四个共用 [`spliced`] 一个原语：把某段轴区间换成"一项"或"没有"，于是"头段照抄 / 尾段平移"
//! 只写一遍。
//!
//! # 不变量
//!
//! 构造期 [`check`] 证明 `起点 + 最高偏移 < 缓冲长度`。这些变换只会**缩小**从起点到最高偏移的
//! 跨度（轴长变短、去掉轴、插入步长 0 的轴、压平紧凑段），于是不变量自动保持——派生视图因此
//! 不必再跑一遍校验，这是它们 O(1) 且能容忍零长度段的原因。

use diagnostic::Result;

use crate::invalid;
use crate::shape::{Order, accumulate};

/// 秩变化的栈上容量：形状变换在栈上重排 `(dims, steps)`，超出这个轴数返回 `None`。
///
/// 16 轴以上的张量在实践中不存在；[`reshape`](crate::Tensor::reshape) 是换见证而不是重排，不受此限。
const MAX_RANK: usize = 16;

/// 构造期校验：秩对应、零元素口径、可达性。`marker` 是构造点标记，文案由调用点给出。
///
/// `numel` 由调用方算好传进来——一次构造只该扫一遍各轴长度。构造总是从起点 0 开始，所以这里
/// 不必传 `base`；派生视图靠"不变量自动保持"（见模块文档）免检。
pub(crate) fn check(
    dims: &[usize],
    steps: &[usize],
    numel: usize,
    len: usize,
    marker: &str,
) -> Result<()> {
    check_rank(dims, steps, marker)?;
    check_empty(numel, len, marker)?;
    check_reach(dims, steps, numel, len, marker)
}

/// 轴数：步长必须与形状逐轴对应，否则下面的偏移计算没有意义。
fn check_rank(dims: &[usize], steps: &[usize], marker: &str) -> Result<()> {
    if dims.len() == steps.len() {
        return Ok(());
    }
    Err(invalid(
        format!(
            "步长有 {} 个轴，形状 {dims:?} 有 {} 个轴",
            steps.len(),
            dims.len()
        ),
        marker,
    ))
}

/// 零元素口径：形状里有 0 个元素时只接受空缓冲。
fn check_empty(numel: usize, len: usize, marker: &str) -> Result<()> {
    if numel != 0 || len == 0 {
        return Ok(());
    }
    Err(invalid(
        format!("零元素视图不该借用 {len} 个元素的缓冲"),
        marker,
    ))
}

/// 可达性：每轴最后一位走出的总偏移必须算得出、且落在缓冲内。
fn check_reach(
    dims: &[usize],
    steps: &[usize],
    numel: usize,
    len: usize,
    marker: &str,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let highest = highest(dims, steps)
        .ok_or_else(|| invalid(format!("形状 {dims:?} 与步长 {steps:?} 算出溢出"), marker))?;
    if highest < len {
        return Ok(());
    }
    Err(invalid(
        format!("最高偏移 {highest} 超出缓冲（{len} 个元素）：步长与形状对不上"),
        marker,
    ))
}

/// 最高下标（每轴最后一位）走出的总偏移；任一轴为空或累加溢出则 `None`。
fn highest(dims: &[usize], steps: &[usize]) -> Option<usize> {
    dims.iter()
        .zip(steps)
        .try_fold(0usize, |acc, (&dim, &step)| {
            acc.checked_add(dim.checked_sub(1)?.checked_mul(step)?)
        })
}

/// 一组下标 → 元素偏移：秩不符、任一轴越界或累加溢出都是 `None`。
pub(crate) fn offset(dims: &[usize], steps: &[usize], base: usize, idx: &[usize]) -> Option<usize> {
    if idx.len() != dims.len() {
        return None;
    }
    let span = idx
        .iter()
        .enumerate()
        .try_fold(0usize, |acc, (axis, &position)| {
            acc.checked_add(axis_offset(dims[axis], steps[axis], position)?)
        })?;
    base.checked_add(span)
}

/// 单轴的偏移增量：下标必须落在该轴长度内，`position * step` 溢出则 `None`。
fn axis_offset(dim: usize, step: usize, position: usize) -> Option<usize> {
    if position >= dim {
        return None;
    }
    position.checked_mul(step)
}

/// 是否为行主序连续布局：与连续步长逐轴核对，无临时分配。
///
/// 长度为 1 的轴不参与核对：它的步长不贡献任何偏移，`unsqueeze` 与广播造出来的 0 步长轴因此
/// 不会被误判成非连续。
pub(crate) fn is_contiguous(dims: &[usize], steps: &[usize]) -> bool {
    if dims.len() != steps.len() {
        return false;
    }
    // 连续 = 整段压平后步长为 1；全单例（或零轴）只有一个元素，任何步长都算连续。
    match collapse(dims, steps, 0, dims.len()) {
        Some((_, 1)) => true,
        Some((dim, _)) => dim <= 1,
        None => false,
    }
}

/// 二维视图的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
pub(crate) fn row<'a, E>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    data: &'a [E],
    row: usize,
) -> Option<&'a [E]> {
    let (start, end) = row_span(dims, steps, base, row)?;
    data.get(start..end)
}

/// 第 `row` 行在存储里的半开区间：列必须是单位步长，否则"一行"不是连续的一段。
fn row_span(dims: &[usize], steps: &[usize], base: usize, row: usize) -> Option<(usize, usize)> {
    let columns = *dims.get(1)?;
    if steps.get(1) != Some(&1) {
        return None;
    }
    let start = offset(dims, steps, base, &[row, 0])?;
    Some((start, start.checked_add(columns)?))
}

/// 重排后的布局：起点偏移 + 各轴长度 / 步长，前 `len` 项有效。
///
/// 定长栈数组，所以形状变换在编译期秩下也不碰堆。
pub(crate) struct Layout {
    base: usize,
    len: usize,
    dims: [usize; MAX_RANK],
    steps: [usize; MAX_RANK],
}

impl Layout {
    /// 起点偏移。
    pub(crate) fn base(&self) -> usize {
        self.base
    }

    /// 各轴长度。
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims[..self.len]
    }

    /// 各轴步长。
    pub(crate) fn steps(&self) -> &[usize] {
        &self.steps[..self.len]
    }

    /// 待填的空壳；轴数超出栈上容量返回 `None`。
    fn blank(base: usize, len: usize) -> Option<Self> {
        if len > MAX_RANK {
            return None;
        }
        Some(Self {
            base,
            len,
            dims: [0; MAX_RANK],
            steps: [0; MAX_RANK],
        })
    }
}

/// 沿 `axis` 取 `[start, start + len)`：返回 (新起点偏移, 新轴长)；越界返回 `None`。
///
/// 秩不变，所以调用方直接改写手里那份元数据，连一次分配都不需要。
pub(crate) fn narrow(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    axis: usize,
    start: usize,
    len: usize,
) -> Option<(usize, usize)> {
    let room = dims.get(axis)?.checked_sub(len)?;
    if start > room {
        return None;
    }
    Some((forward(steps, base, axis, start)?, len))
}

/// 沿 `axis` 前移 `count` 格后的起点偏移；轴越界或溢出返回 `None`。
fn forward(steps: &[usize], base: usize, axis: usize, count: usize) -> Option<usize> {
    base.checked_add(count.checked_mul(*steps.get(axis)?)?)
}

/// 沿 `axis` 取第 `index` 个：起点前移、该轴去掉。越界返回 `None`。
pub(crate) fn select(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    axis: usize,
    idx: usize,
) -> Option<Layout> {
    let step = *steps.get(axis)?;
    let base = base.checked_add(axis_offset(*dims.get(axis)?, step, idx)?)?;
    spliced(dims, steps, base, axis, axis + 1, None)
}

/// 去掉长为 1 的第 `axis` 轴；该轴不是单例返回 `None`。
pub(crate) fn squeeze(dims: &[usize], steps: &[usize], base: usize, axis: usize) -> Option<Layout> {
    if dims.get(axis) != Some(&1) {
        return None;
    }
    spliced(dims, steps, base, axis, axis + 1, None)
}

/// 在第 `axis` 位插入一个长为 1 的轴；轴越界返回 `None`。
///
/// 单例轴的步长不参与连续性核对，所以插入项写成步长 0——它不贡献偏移，也不必猜周围是什么布局。
pub(crate) fn unsqueeze(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    axis: usize,
) -> Option<Layout> {
    spliced(dims, steps, base, axis, axis, Some((1, 0)))
}

/// 把 `[start, end)` 各轴压成一个轴；该段不是行主序紧凑排布返回 `None`。
pub(crate) fn flatten(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    start: usize,
    end: usize,
) -> Option<Layout> {
    let (dim, step) = collapse(dims, steps, start, end)?;
    spliced(dims, steps, base, start, end, Some((dim, step)))
}

/// 形状代数的唯一原语：把 `[start, end)` 换成 `item`（`None` 是删掉，`Some` 是并成一项）。
///
/// `start == end` 即插入，`end == start + 1` 且 `item` 为 `None` 即删除——三种秩变化共用一套
/// "头段照抄 / 尾段平移"的搬动。
fn spliced(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    start: usize,
    end: usize,
    item: Option<(usize, usize)>,
) -> Option<Layout> {
    if start > end || end > dims.len() {
        return None;
    }
    let added = usize::from(item.is_some());
    let mut out = Layout::blank(base, dims.len() - (end - start) + added)?;
    let tail = start + added;
    let (dst_dims, dst_steps) = (&mut out.dims[..out.len], &mut out.steps[..out.len]);
    dst_dims[..start].copy_from_slice(&dims[..start]);
    dst_steps[..start].copy_from_slice(&steps[..start]);
    if let Some((dim, step)) = item {
        dst_dims[start] = dim;
        dst_steps[start] = step;
    }
    dst_dims[tail..].copy_from_slice(&dims[end..]);
    dst_steps[tail..].copy_from_slice(&steps[end..]);
    Some(out)
}

/// `[start, end)` 压成一轴的 (轴长, 步长)；该段不是行主序紧凑排布返回 `None`。
///
/// 期望步长由 [`accumulate`] 给出——它与 [`Shape::strides`](crate::Shape::strides)、
/// [`Rank::contiguous`](crate::Rank::contiguous) 共用同一份累积规则，本函数只做逐轴比对，
/// 于是"连续"在本 crate 里始终只有一条定律。轴长不超过 1 时不必核对步长：这种段里没有
/// 第二个位置，步长无人读到。
fn collapse(dims: &[usize], steps: &[usize], start: usize, end: usize) -> Option<(usize, usize)> {
    let seg = dims.get(start..end)?;
    let dim = run_product(seg)?;
    if dim <= 1 {
        return Some((dim, 0));
    }
    Some((dim, unit_step(seg, steps, start)?))
}

/// 段内轴长之积；溢出返回 `None`。
fn run_product(seg: &[usize]) -> Option<usize> {
    seg.iter()
        .try_fold(1usize, |acc, &axis| acc.checked_mul(axis))
}

/// 段内压平后的单位步长（最内侧非单例轴的步长）；步长与行主序对不上返回 `None`。
fn unit_step(seg: &[usize], steps: &[usize], start: usize) -> Option<usize> {
    let mut want = [0usize; MAX_RANK];
    let want = want.get_mut(..seg.len())?;
    accumulate(seg, want, Order::RowMajor);
    // 单位步长取自最内侧非单例轴：它内侧全是单例，期望步长必然是 1。
    let unit = seg
        .iter()
        .rposition(|&axis| axis > 1)
        .and_then(|at| steps.get(start + at).copied())?;
    agree(seg, steps, start, (unit, want))?;
    Some(unit)
}

/// 逐轴核对实际步长等于 `unit × 期望步长`；任一轴对不上返回 `None`。
fn agree(
    seg: &[usize],
    steps: &[usize],
    start: usize,
    (unit, want): (usize, &[usize]),
) -> Option<()> {
    seg.iter()
        .copied()
        .enumerate()
        .filter(|(_, axis)| *axis > 1)
        .try_for_each(|(at, _)| {
            let reached = steps.get(start + at).copied()?;
            (reached == unit.checked_mul(want[at])?).then_some(())
        })
}
