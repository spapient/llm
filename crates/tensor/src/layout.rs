//! 布局算法：形状 + 步长 → 元素偏移的**唯一**实现。
//!
//! 同一个 [`Tensor`](crate::Tensor) 可以是运行期秩（`Vec<usize>`）或编译期秩（`[usize; D]`）。
//! 下标、可达性、连续性与取行这些算式与秩的来源、元素的类型都无关，因此收口在这里一份——两套各自
//! 实现意味着任何一处溢出 / 越界口径的修改都要记得改两遍，那正是漂移的来源。
//!
//! 统一的调用形状是 `(dims, steps)` 两个切片；`&[usize; D]` 自动退化成 `&[usize]`。

use diagnostic::Result;

use crate::invalid;

/// 构造期校验：秩对应、零元素口径、可达性——三条判据写在一个函数里，省掉三层转调。
///
/// `numel` 由调用方算好传进来：一次构造只该扫一遍各轴长度。
pub(crate) fn check(
    dims: &[usize],
    steps: &[usize],
    numel: usize,
    len: usize,
    marker: &str,
) -> Result<()> {
    if dims.len() != steps.len() {
        let axes = (steps.len(), dims.len());
        return Err(invalid(
            format!("步长有 {} 个轴，形状 {dims:?} 有 {} 个轴", axes.0, axes.1),
            marker,
        ));
    }
    if numel == 0 {
        return match len {
            0 => Ok(()),
            len => Err(invalid(
                format!("零元素视图不该借用 {len} 个元素的缓冲"),
                marker,
            )),
        };
    }
    // 最高下标（每轴最后一位）走出的总偏移必须算得出、且落在缓冲内。
    let highest = dims
        .iter()
        .zip(steps)
        .try_fold(0usize, |acc, (&dim, &step)| {
            acc.checked_add(dim.checked_sub(1)?.checked_mul(step)?)
        });
    match highest {
        Some(highest) if highest < len => Ok(()),
        Some(highest) => Err(invalid(
            format!("最高偏移 {highest} 超出缓冲（{len} 个元素）：步长与形状对不上"),
            marker,
        )),
        None => Err(invalid(
            format!("形状 {dims:?} 与步长 {steps:?} 算出溢出"),
            marker,
        )),
    }
}

/// 一组下标 → 元素偏移：秩不符、任一轴越界或累加溢出都是 `None`。
pub(crate) fn offset(dims: &[usize], steps: &[usize], idx: &[usize]) -> Option<usize> {
    if idx.len() != dims.len() {
        return None;
    }
    idx.iter()
        .enumerate()
        .try_fold(0usize, |acc, (axis, &position)| {
            acc.checked_add(axis_offset(dims[axis], steps[axis], position)?)
        })
}

/// 单轴的偏移增量：下标必须落在该轴长度内，`position * step` 溢出则 `None`。
fn axis_offset(dim: usize, step: usize, position: usize) -> Option<usize> {
    if position >= dim {
        return None;
    }
    position.checked_mul(step)
}

/// 是否为行主序连续布局：与连续步长逐轴核对，无临时分配。
pub(crate) fn is_contiguous(dims: &[usize], steps: &[usize]) -> bool {
    if dims.len() != steps.len() {
        return false;
    }
    let mut acc = 1usize;
    // 反向核对期望累积值；串行依赖，保留循环。
    for axis in (0..dims.len()).rev() {
        if steps[axis] != acc {
            return false;
        }
        acc = acc.saturating_mul(dims[axis]);
    }
    true
}

/// 二维视图的第 `row` 行；秩不是 2、行越界或列不连续时返回 `None`。
pub(crate) fn row<'a, E>(
    dims: &[usize],
    steps: &[usize],
    data: &'a [E],
    row: usize,
) -> Option<&'a [E]> {
    let [_, columns] = <[usize; 2]>::try_from(dims).ok()?;
    if steps.get(1) != Some(&1) {
        return None;
    }
    let start = axis_offset(dims[0], steps[0], row)?;
    data.get(start..start.checked_add(columns)?)
}
