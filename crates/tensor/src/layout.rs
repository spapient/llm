//! 布局代数：`(轴长, 步长, 起点)` 的校验、变换与逻辑序遍历。
//!
//! 所有公开形状算子共用这里的偏移规则。秩容器由 [`Rank`] 决定，因此定长秩无需堆分配，
//! 动态秩也没有固定的最大轴数。构造期证明最高偏移可达，派生变换只保持或缩小可达范围。

use diagnostic::Result;

use crate::invalid;
use crate::shape::{Rank, Slice, product};

/// 校验秩对应、零元素口径、整数溢出和缓冲可达性。
pub(crate) fn check(
    dims: &[usize],
    steps: &[usize],
    numel: usize,
    len: usize,
    marker: &str,
) -> Result<()> {
    if dims.len() != steps.len() {
        return Err(invalid(
            format!(
                "步长有 {} 个轴，形状 {dims:?} 有 {} 个轴",
                steps.len(),
                dims.len()
            ),
            marker,
        ));
    }
    if numel == 0 && len != 0 {
        return Err(invalid(
            format!("零元素视图不该借用 {len} 个元素的缓冲"),
            marker,
        ));
    }
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

/// 每轴最后一位的总偏移；空轴或算术溢出返回 `None`。
fn highest(dims: &[usize], steps: &[usize]) -> Option<usize> {
    dims.iter()
        .zip(steps)
        .try_fold(0usize, |acc, (&dim, &step)| {
            acc.checked_add(dim.checked_sub(1)?.checked_mul(step)?)
        })
}

/// 多下标转元素偏移；秩不符、越界或溢出返回 `None`。
pub(crate) fn offset(dims: &[usize], steps: &[usize], base: usize, idx: &[usize]) -> Option<usize> {
    if idx.len() != dims.len() {
        return None;
    }
    let span = idx
        .iter()
        .enumerate()
        .try_fold(0usize, |acc, (axis, &position)| {
            let dim = *dims.get(axis)?;
            let step = *steps.get(axis)?;
            if position >= dim {
                return None;
            }
            acc.checked_add(position.checked_mul(step)?)
        })?;
    base.checked_add(span)
}

/// 判断是否为行主序紧凑布局；单例轴和空轴不产生额外偏移约束。
pub(crate) fn is_contiguous(dims: &[usize], steps: &[usize]) -> bool {
    if dims.len() != steps.len() {
        return false;
    }
    match collapse(dims, steps, 0, dims.len()) {
        Some((_, 1)) => true,
        Some((dim, _)) => dim <= 1,
        None => false,
    }
}

/// 二维视图的连续行；秩、行号或列步长不合法返回 `None`。
pub(crate) fn row<'a, E>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    data: &'a [E],
    row: usize,
) -> Option<&'a [E]> {
    let columns = *dims.get(1)?;
    if steps.get(1) != Some(&1) {
        return None;
    }
    let start = offset(dims, steps, base, &[row, 0])?;
    data.get(start..start.checked_add(columns)?)
}

/// 一组同秩轴长与步长及其起点。
#[derive(Debug, Clone)]
pub(crate) struct Layout<R> {
    /// 起点偏移。
    pub(crate) base: usize,
    dims: R,
    steps: R,
}

impl<R: Rank> Layout<R> {
    /// 各轴长度。
    pub(crate) fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// 各轴步长。
    pub(crate) fn steps(&self) -> &[usize] {
        self.steps.as_slice()
    }

    /// 拆出起点、轴长和步长，供零拷贝组装使用。
    pub(crate) fn into_parts(self) -> (usize, R, R) {
        (self.base, self.dims, self.steps)
    }
}

/// 沿 `axis` 取 `[start, start + len)`，返回新起点和轴长。
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
    let base = base.checked_add(start.checked_mul(*steps.get(axis)?)?)?;
    Some((base, len))
}

/// 多轴切片；未给出的尾部轴保持不变。
pub(crate) fn window<R: Rank>(
    dims: &R,
    steps: &R,
    base: usize,
    slices: &[Slice],
) -> Option<Layout<R>> {
    if slices.len() > dims.as_slice().len() {
        return None;
    }
    let mut out = Layout {
        base,
        dims: dims.clone(),
        steps: steps.clone(),
    };
    for (axis, slice) in slices.iter().enumerate() {
        place(&mut out, axis, *slice)?;
    }
    Some(out)
}

/// 把一个切片落到轴元数据上，所有乘法都受溢出检查。
fn place<R: Rank>(out: &mut Layout<R>, axis: usize, slice: Slice) -> Option<()> {
    if slice.step == 0 {
        return None;
    }
    let len = slice.end.checked_sub(slice.start)?;
    let (base, dim) = narrow(
        out.dims(),
        out.steps(),
        out.base,
        axis,
        slice.start,
        len,
    )?;
    let step = out
        .steps
        .as_slice()
        .get(axis)?
        .checked_mul(slice.step)?;
    out.base = base;
    out.dims.as_slice_mut()[axis] = dim.div_ceil(slice.step);
    out.steps.as_slice_mut()[axis] = step;
    Some(())
}

/// 固定一轴并删除它。
pub(crate) fn select<R: Rank>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    axis: usize,
    idx: usize,
) -> Option<Layout<R>> {
    let dim = *dims.get(axis)?;
    let step = *steps.get(axis)?;
    if idx >= dim {
        return None;
    }
    let base = base.checked_add(idx.checked_mul(step)?)?;
    spliced(dims, steps, base, axis, axis + 1, None)
}

/// 删除长为 1 的一轴。
pub(crate) fn squeeze<R: Rank>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    axis: usize,
) -> Option<Layout<R>> {
    if dims.get(axis) != Some(&1) {
        return None;
    }
    spliced(dims, steps, base, axis, axis + 1, None)
}

/// 两轴取对角：保留轴长度取较小值，步长取两轴之和。
pub(crate) fn diagonal<R: Rank>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    a: usize,
    b: usize,
) -> Option<Layout<R>> {
    if a == b {
        return None;
    }
    let (keep, drop) = (a.min(b), a.max(b));
    let len = (*dims.get(keep)?).min(*dims.get(drop)?);
    let diagonal_step = steps.get(keep)?.checked_add(*steps.get(drop)?)?;
    let out_dims = R::from_iter(dims.iter().copied().enumerate().filter_map(|(axis, dim)| {
        (axis != drop).then_some(if axis == keep { len } else { dim })
    }))?;
    let out_steps = R::from_iter(steps.iter().copied().enumerate().filter_map(|(axis, step)| {
        (axis != drop).then_some(if axis == keep { diagonal_step } else { step })
    }))?;
    Some(Layout {
        base,
        dims: out_dims,
        steps: out_steps,
    })
}

/// 插入一个长为 1、步长为 0 的轴。
pub(crate) fn unsqueeze<R: Rank>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    axis: usize,
) -> Option<Layout<R>> {
    spliced(dims, steps, base, axis, axis, Some((1, 0)))
}

/// 把行主序紧凑的 `[start, end)` 轴段压成一轴。
pub(crate) fn flatten<R: Rank>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    start: usize,
    end: usize,
) -> Option<Layout<R>> {
    let (dim, step) = collapse(dims, steps, start, end)?;
    spliced(dims, steps, base, start, end, Some((dim, step)))
}

/// 将轴段替换为一项或删除，直接构造目标秩容器。
fn spliced<R: Rank>(
    dims: &[usize],
    steps: &[usize],
    base: usize,
    start: usize,
    end: usize,
    item: Option<(usize, usize)>,
) -> Option<Layout<R>> {
    if start > end || end > dims.len() || dims.len() != steps.len() {
        return None;
    }
    let out_dims = R::from_iter(
        dims[..start]
            .iter()
            .copied()
            .chain(item.map(|(dim, _)| dim))
            .chain(dims[end..].iter().copied()),
    )?;
    let out_steps = R::from_iter(
        steps[..start]
            .iter()
            .copied()
            .chain(item.map(|(_, step)| step))
            .chain(steps[end..].iter().copied()),
    )?;
    Some(Layout {
        base,
        dims: out_dims,
        steps: out_steps,
    })
}

/// 计算轴段压平后的轴长和步长；非紧凑段返回 `None`。
fn collapse(dims: &[usize], steps: &[usize], start: usize, end: usize) -> Option<(usize, usize)> {
    let segment = dims.get(start..end)?;
    let dim = segment
        .iter()
        .try_fold(1usize, |acc, &axis| acc.checked_mul(axis))?;
    if dim <= 1 {
        return Some((dim, 0));
    }
    let inner = segment.iter().rposition(|&axis| axis > 1)?;
    let unit = *steps.get(start + inner)?;
    let mut expected = unit;
    for (offset, &axis) in segment.iter().enumerate().rev() {
        if axis <= 1 {
            continue;
        }
        if steps.get(start + offset) != Some(&expected) {
            return None;
        }
        expected = expected.checked_mul(axis)?;
    }
    Some((dim, unit))
}

/// 逻辑行主序偏移迭代器，可选择让指定轴反向遍历。
///
/// 状态使用泛型秩容器；连续切片由调用方走整段拷贝快路径，其余布局每个元素只做一次进位，
/// 不为读取临时物化完整张量。
pub(crate) struct Offsets<'a, R> {
    dims: &'a [usize],
    steps: &'a [usize],
    turn: &'a [bool],
    indices: Option<R>,
    at: isize,
    remaining: usize,
}

impl<'a, R: Rank> Offsets<'a, R> {
    /// 创建偏移序列。无效见证或翻转起点溢出会产生空序列而非 panic。
    pub(crate) fn new(
        dims: &'a [usize],
        steps: &'a [usize],
        base: usize,
        turn: &'a [bool],
    ) -> Self {
        let indices = R::filled(dims.len(), 0);
        let mut remaining = if dims.len() == steps.len() && indices.is_some() {
            product(dims)
        } else {
            0
        };
        let mut at = base;
        for (axis, &dim) in dims.iter().enumerate() {
            if !turn.get(axis).copied().unwrap_or(false) {
                continue;
            }
            let Some(step) = steps.get(axis) else {
                remaining = 0;
                break;
            };
            match step
                .checked_mul(dim.saturating_sub(1))
                .and_then(|span| at.checked_add(span))
            {
                Some(start) => at = start,
                None => {
                    remaining = 0;
                    break;
                }
            }
        }
        Self {
            dims,
            steps,
            turn,
            indices,
            at: at as isize,
            remaining,
        }
    }

    /// 行进到下一个逻辑位置；最后一个元素之后不再做无效偏移运算。
    fn advance(&mut self) {
        let Some(indices) = self.indices.as_mut() else {
            self.remaining = 0;
            return;
        };
        let indices = indices.as_slice_mut();
        let mut at = self.at;
        for axis in (0..indices.len()).rev() {
            indices[axis] += 1;
            let step = self.steps[axis] as isize;
            let turned = self.turn.get(axis).copied().unwrap_or(false);
            at += if turned { -step } else { step };
            if indices[axis] < self.dims[axis] {
                break;
            }
            indices[axis] = 0;
            let span = step * self.dims[axis] as isize;
            at += if turned { span } else { -span };
        }
        self.at = at;
    }
}

impl<R: Rank> Iterator for Offsets<'_, R> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let current = self.at as usize;
        self.remaining -= 1;
        if self.remaining != 0 {
            self.advance();
        }
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<R: Rank> ExactSizeIterator for Offsets<'_, R> {
    fn len(&self) -> usize {
        self.remaining
    }
}
