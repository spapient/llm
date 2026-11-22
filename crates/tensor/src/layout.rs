//! 布局算法：形状 + 步长 → 元素偏移的**唯一**实现。
//!
//! 同一个 [`Tensor`](crate::Tensor) 可以是运行期秩（`Vec<usize>`）或编译期秩（`[usize; D]`），
//! [`Raw`](crate::Raw) 的字节视图也走这里。下标、可达性、连续性与取行这些算式与秩的来源、元素的
//! 类型都无关，因此收口在这里一份——两套各自实现意味着任何一处溢出 / 越界口径的修改都要记得改两遍，
//! 那正是漂移的来源。
//!
//! 统一的调用形状是 `(dims, steps)` 两个切片；`&[usize; D]` 自动退化成 `&[usize]`。

use diagnostic::{Code, Error, Result};

/// 构造期校验：秩对应、零元素口径、可达性。`marker` 是构造点标记，文案由调用点给出。
pub(crate) fn check(dims: &[usize], steps: &[usize], len: usize, marker: &str) -> Result<()> {
    check_rank(dims, steps, marker)?;
    check_empty(dims, len, marker)?;
    check_reach(dims, steps, len, marker)
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
fn check_empty(dims: &[usize], len: usize, marker: &str) -> Result<()> {
    if crate::product(dims) != 0 || len == 0 {
        return Ok(());
    }
    Err(invalid(
        format!("零元素视图不该借用 {len} 个元素的缓冲"),
        marker,
    ))
}

/// 可达性：每轴最后一位走出的总偏移必须算得出、且落在缓冲内。
fn check_reach(dims: &[usize], steps: &[usize], len: usize, marker: &str) -> Result<()> {
    if crate::product(dims) == 0 {
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

/// 构造期 `Invalid` 错误：本模块所有校验共用一条构造路径，标记由调用点给出。
fn invalid(message: String, marker: &str) -> Error {
    Error::new(Code::Invalid, message).with(marker)
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
