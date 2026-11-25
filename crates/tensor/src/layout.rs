//! 布局代数：`(轴长, 步长, 起点)` 的校验、变换与逻辑序遍历。
//!
//! 所有公开形状算子共用这里的偏移规则。秩容器由 [`Rank`] 决定，因此定长秩无需堆分配，
//! 动态秩也没有固定的最大轴数。构造期证明最高偏移可达，派生变换只保持或缩小可达范围。

use diagnostic::Result;

use crate::element::Element;
use crate::tensor::{Order, Rank, Slice, accumulate, assemble, product};
use crate::{DynTensor, Owned, Store, StoreMut, Tensor, View, ViewMut, invalid};

/// 校验秩对应、零元素口径、整数溢出和缓冲可达性。
pub(crate) fn check(
    dims: &[usize],
    strides: &[usize],
    numel: usize,
    len: usize,
    marker: &str,
) -> Result<()> {
    check_rank(dims, strides, marker)?;
    check_empty(numel, len, marker)?;
    if numel == 0 {
        return Ok(());
    }
    check_reachable(dims, strides, len, marker)
}

/// 轴长与步长必须来自同一个秩见证。
fn check_rank(dims: &[usize], strides: &[usize], marker: &str) -> Result<()> {
    if dims.len() == strides.len() {
        return Ok(());
    }
    Err(invalid(
        format!(
            "步长有 {} 个轴，形状 {dims:?} 有 {} 个轴",
            strides.len(),
            dims.len()
        ),
        marker,
    ))
}

/// 空视图只能借用空缓冲，非空视图交给可达性校验。
fn check_empty(numel: usize, len: usize, marker: &str) -> Result<()> {
    if numel != 0 || len == 0 {
        return Ok(());
    }
    Err(invalid(
        format!("零元素视图不该借用 {len} 个元素的缓冲"),
        marker,
    ))
}

/// 最高逻辑位置必须落在实际缓冲内。
fn check_reachable(dims: &[usize], strides: &[usize], len: usize, marker: &str) -> Result<()> {
    let highest = highest(dims, strides)
        .ok_or_else(|| invalid(format!("形状 {dims:?} 与步长 {strides:?} 算出溢出"), marker))?;
    if highest < len {
        return Ok(());
    }
    Err(invalid(
        format!("最高偏移 {highest} 超出缓冲（{len} 个元素）：步长与形状对不上"),
        marker,
    ))
}

/// 每轴最后一位的总偏移；空轴或算术溢出返回 `None`。
fn highest(dims: &[usize], strides: &[usize]) -> Option<usize> {
    dims.iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &step)| {
            acc.checked_add(dim.checked_sub(1)?.checked_mul(step)?)
        })
}

/// 读取同一轴的长度与步长，避免各变换分别处理缺失轴。
fn axis_spec(dims: &[usize], strides: &[usize], axis: usize) -> Option<(usize, usize)> {
    Some((*dims.get(axis)?, *strides.get(axis)?))
}

/// 从 `base` 沿 `step` 行进 `at` 步，乘法与加法都受溢出保护。
fn shifted(base: usize, step: usize, at: usize) -> Option<usize> {
    at.checked_mul(step).and_then(|span| base.checked_add(span))
}

/// 多下标转元素偏移；秩不符、越界或溢出返回 `None`。
pub(crate) fn offset(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    idx: &[usize],
) -> Option<usize> {
    if idx.len() != dims.len() {
        return None;
    }
    idx.iter()
        .enumerate()
        .try_fold(base, |base, (axis, &position)| {
            let (dim, step) = axis_spec(dims, strides, axis)?;
            (position < dim)
                .then_some(())
                .and_then(|()| shifted(base, step, position))
        })
}

/// 判断是否为行主序紧凑布局；单例轴和空轴不产生额外偏移约束。
pub(crate) fn is_contiguous(dims: &[usize], strides: &[usize]) -> bool {
    if dims.len() != strides.len() {
        return false;
    }
    match collapse(dims, strides, 0, dims.len()) {
        Some((_, 1)) => true,
        Some((dim, _)) => dim <= 1,
        None => false,
    }
}

/// 二维视图的连续行；秩、行号或列步长不合法返回 `None`。
pub(crate) fn row<'a, E>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    data: &'a [E],
    row: usize,
) -> Option<&'a [E]> {
    let columns = *dims.get(1)?;
    if strides.get(1) != Some(&1) {
        return None;
    }
    let start = offset(dims, strides, base, &[row, 0])?;
    data.get(start..start.checked_add(columns)?)
}

/// 一组同秩轴长与步长及其起点。
#[derive(Debug, Clone)]
pub(crate) struct Layout<R> {
    /// 起点偏移。
    pub(crate) base: usize,
    dims: R,
    strides: R,
}

impl<R: Rank> Layout<R> {
    /// 各轴长度。
    pub(crate) fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// 各轴步长。
    pub(crate) fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    /// 拆出起点、轴长和步长，供零拷贝组装使用。
    pub(crate) fn into_parts(self) -> (usize, R, R) {
        (self.base, self.dims, self.strides)
    }
}

/// 沿 `axis` 取 `[start, start + len)`，返回新起点和轴长。
pub(crate) fn narrow(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    axis: usize,
    start: usize,
    len: usize,
) -> Option<(usize, usize)> {
    let (dim, step) = axis_spec(dims, strides, axis)?;
    dim.checked_sub(len)?.checked_sub(start)?;
    shifted(base, step, start).map(|base| (base, len))
}

/// 多轴切片；未给出的尾部轴保持不变。
pub(crate) fn window<R: Rank>(
    dims: &R,
    strides: &R,
    base: usize,
    slices: impl IntoIterator<Item = Slice>,
) -> Option<Layout<R>> {
    let mut out = Layout {
        base,
        dims: dims.clone(),
        strides: strides.clone(),
    };
    for (axis, slice) in slices.into_iter().enumerate() {
        place(&mut out, axis, slice)?;
    }
    Some(out)
}

/// 把一个切片落到轴元数据上，所有乘法都受溢出检查。
fn place<R: Rank>(out: &mut Layout<R>, axis: usize, slice: Slice) -> Option<()> {
    if slice.step == 0 {
        return None;
    }
    let len = slice.end.checked_sub(slice.start)?;
    let (base, dim) = narrow(out.dims(), out.strides(), out.base, axis, slice.start, len)?;
    let step = out
        .strides
        .as_slice()
        .get(axis)
        .and_then(|step| step.checked_mul(slice.step))?;
    out.base = base;
    out.dims.as_slice_mut()[axis] = dim.div_ceil(slice.step);
    out.strides.as_slice_mut()[axis] = step;
    Some(())
}

/// 固定一轴并删除它。
pub(crate) fn select<R: Rank>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    axis: usize,
    idx: usize,
) -> Option<Layout<R>> {
    let (base, _) = narrow(dims, strides, base, axis, idx, 1)?;
    spliced(dims, strides, base, axis, axis + 1, None)
}

/// 删除长为 1 的一轴。
pub(crate) fn squeeze<R: Rank>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    axis: usize,
) -> Option<Layout<R>> {
    if dims.get(axis) != Some(&1) {
        return None;
    }
    spliced(dims, strides, base, axis, axis + 1, None)
}

/// 两轴取对角：保留轴长度取较小值，步长取两轴之和。
pub(crate) fn diagonal<R: Rank>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    a: usize,
    b: usize,
) -> Option<Layout<R>> {
    let (keep, drop, len, diagonal_step) = diagonal_geometry(dims, strides, a, b)?;
    let out_dims = R::from_iter(
        dims.iter()
            .copied()
            .enumerate()
            .filter_map(|(axis, dim)| diagonal_axis(axis, keep, drop, dim, len)),
    )?;
    let out_strides = R::from_iter(
        strides
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(axis, step)| diagonal_axis(axis, keep, drop, step, diagonal_step)),
    )?;
    Some(Layout {
        base,
        dims: out_dims,
        strides: out_strides,
    })
}

/// 计算对角视图的四元几何，集中处理同轴、缺轴与步长溢出。
fn diagonal_geometry(
    dims: &[usize],
    strides: &[usize],
    a: usize,
    b: usize,
) -> Option<(usize, usize, usize, usize)> {
    if a == b {
        return None;
    }
    let (keep, drop) = (a.min(b), a.max(b));
    let (keep_dim, keep_step) = axis_spec(dims, strides, keep)?;
    let (drop_dim, drop_step) = axis_spec(dims, strides, drop)?;
    let step = keep_step.checked_add(drop_step)?;
    Some((keep, drop, keep_dim.min(drop_dim), step))
}

/// 删除对角源轴，并把保留轴替换为对角长度或合成步长。
fn diagonal_axis(
    axis: usize,
    keep: usize,
    drop: usize,
    value: usize,
    replacement: usize,
) -> Option<usize> {
    (axis != drop).then_some(if axis == keep { replacement } else { value })
}

/// 插入一个长为 1、步长为 0 的轴。
pub(crate) fn unsqueeze<R: Rank>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    axis: usize,
) -> Option<Layout<R>> {
    spliced(dims, strides, base, axis, axis, Some((1, 0)))
}

/// 把行主序紧凑的 `[start, end)` 轴段压成一轴。
pub(crate) fn flatten<R: Rank>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    start: usize,
    end: usize,
) -> Option<Layout<R>> {
    let (dim, step) = collapse(dims, strides, start, end)?;
    spliced(dims, strides, base, start, end, Some((dim, step)))
}

/// 将轴段替换为一项或删除，直接构造目标秩容器。
fn spliced<R: Rank>(
    dims: &[usize],
    strides: &[usize],
    base: usize,
    start: usize,
    end: usize,
    item: Option<(usize, usize)>,
) -> Option<Layout<R>> {
    valid_range(dims, strides, start, end)?;
    let out_dims = R::from_iter(
        dims[..start]
            .iter()
            .copied()
            .chain(item.map(|(dim, _)| dim))
            .chain(dims[end..].iter().copied()),
    )?;
    let out_strides = R::from_iter(
        strides[..start]
            .iter()
            .copied()
            .chain(item.map(|(_, step)| step))
            .chain(strides[end..].iter().copied()),
    )?;
    Some(Layout {
        base,
        dims: out_dims,
        strides: out_strides,
    })
}

/// 轴段必须在同秩元数据内，切片下标才可安全使用。
fn valid_range(dims: &[usize], strides: &[usize], start: usize, end: usize) -> Option<()> {
    (start <= end && end <= dims.len() && dims.len() == strides.len()).then_some(())
}

/// 计算轴段压平后的轴长和步长；非紧凑段返回 `None`。
fn collapse(dims: &[usize], strides: &[usize], start: usize, end: usize) -> Option<(usize, usize)> {
    let segment = dims.get(start..end)?;
    let dim = collapsed_dim(segment)?;
    let unit = unit_step(segment, strides, start, dim)?;
    check_compact(segment, strides, start, dim, unit).map(|()| (dim, unit))
}

/// 轴段总长度，任何乘法溢出都使变换失败。
fn collapsed_dim(segment: &[usize]) -> Option<usize> {
    segment
        .iter()
        .try_fold(1usize, |acc, &axis| acc.checked_mul(axis))
}

/// 空轴和单例轴不产生偏移；其余布局取最内非单例轴的步长。
fn unit_step(segment: &[usize], strides: &[usize], start: usize, dim: usize) -> Option<usize> {
    if dim <= 1 {
        return Some(0);
    }
    let inner = segment.iter().rposition(|&axis| axis > 1)?;
    strides.get(start + inner).copied()
}

/// 从内向外线性验证期望步长，单例轴不参与约束。
fn check_compact(
    segment: &[usize],
    strides: &[usize],
    start: usize,
    dim: usize,
    unit: usize,
) -> Option<()> {
    if dim <= 1 {
        return Some(());
    }
    segment
        .iter()
        .enumerate()
        .rev()
        .try_fold(unit, |expected, (offset, &axis)| {
            compact_step(strides, start, offset, axis, expected)
        })
        .map(|_| ())
}

/// 验证一个非单例轴并推出下一轴的期望步长。
fn compact_step(
    strides: &[usize],
    start: usize,
    offset: usize,
    axis: usize,
    expected: usize,
) -> Option<usize> {
    if axis <= 1 {
        return Some(expected);
    }
    (strides.get(start + offset) == Some(&expected)).then_some(())?;
    expected.checked_mul(axis)
}

/// 逻辑行主序偏移迭代器，可选择让指定轴反向遍历。
///
/// 状态使用泛型秩容器；连续切片由调用方走整段拷贝快路径，其余布局每个元素只做一次进位，
/// 不为读取临时物化完整张量。
pub(crate) struct Offsets<'a, R> {
    dims: &'a [usize],
    strides: &'a [usize],
    reverse: &'a [bool],
    idxs: Option<R>,
    offset: isize,
    remaining: usize,
}

impl<'a, R: Rank> Offsets<'a, R> {
    /// 创建偏移序列。无效见证或翻转起点溢出会产生空序列而非 panic。
    // 基址由布局校验约束在缓冲范围内，宽度不超 isize；clippy 看不到该不变量，允许构造转换。
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn new(
        dims: &'a [usize],
        strides: &'a [usize],
        base: usize,
        reverse: &'a [bool],
    ) -> Self {
        let idxs = R::filled(dims.len(), 0);
        let offset = Self::reversed_base(dims, strides, base, reverse);
        let remaining = if offset.is_some() && dims.len() == strides.len() && idxs.is_some() {
            product(dims)
        } else {
            0
        };
        Self {
            dims,
            strides,
            reverse,
            idxs,
            offset: offset.unwrap_or(0) as isize,
            remaining,
        }
    }

    /// 每个反向轴从逻辑末位起步；任一轴缺失或溢出都使序列为空。
    fn reversed_base(
        dims: &[usize],
        strides: &[usize],
        base: usize,
        reverse: &[bool],
    ) -> Option<usize> {
        dims.iter()
            .enumerate()
            .try_fold(base, |offset, (axis, &dim)| {
                if !reverse.get(axis).copied().unwrap_or(false) {
                    return Some(offset);
                }
                let stride = *strides.get(axis)?;
                shifted(offset, stride, dim.saturating_sub(1))
            })
    }

    /// 行进到下一个逻辑位置；最后一个元素之后不再做无效偏移运算。
    fn advance(&mut self) {
        let Some(idxs) = self.idxs.as_mut() else {
            self.remaining = 0;
            return;
        };
        let idxs = idxs.as_slice_mut();
        let mut offset = self.offset;
        for axis in (0..idxs.len()).rev() {
            idxs[axis] += 1;
            let reversed = self.reverse.get(axis).copied().unwrap_or(false);
            let (step, reset) = Self::directed_step(self.strides[axis], self.dims[axis], reversed);
            offset += step;
            if idxs[axis] < self.dims[axis] {
                break;
            }
            idxs[axis] = 0;
            offset += reset;
        }
        self.offset = offset;
    }

    /// 返回单次行进与轴回绕的有向位移，正反方向共用同一套进位。
    // stride 与轴长 dim 均经布局校验约束在缓冲范围内，宽度不超 isize；允许形状整数到有向位移的换算。
    #[allow(clippy::cast_possible_wrap)]
    fn directed_step(stride: usize, dim: usize, reversed: bool) -> (isize, isize) {
        let stride = stride as isize;
        let span = stride * dim as isize;
        if reversed {
            (-stride, span)
        } else {
            (stride, -span)
        }
    }
}

impl<R: Rank> Iterator for Offsets<'_, R> {
    type Item = usize;

    // 当前落点由基址与有向进位约束在缓冲内、下标非负；clippy 看不到反向方向的非负不变量。
    #[allow(clippy::cast_sign_loss)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let current = self.offset as usize;
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

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 交换两轴，不移动元素。
    #[doc(alias = "swap_dims")]
    pub fn transpose(self, first: usize, second: usize) -> Option<Self> {
        self.map_axes(|axes| axes.swapped(first, second))
    }

    /// 把一轴搬到另一位置，其余轴的相对顺序不变。
    #[doc(alias = "moveaxis")]
    pub fn movedim(self, from: usize, to: usize) -> Option<Self> {
        self.map_axes(|axes| axes.moved(from, to))
    }

    /// 按完整轴排列重排元数据。
    pub fn permute(self, order: impl AsRef<[usize]>) -> Option<Self> {
        let order = order.as_ref();
        if !is_permutation(order, self.rank()) {
            return None;
        }
        self.map_axes(|axes| axes.permuted(order))
    }

    /// 对轴长和步长施加同一个变换，避免两套元数据漂移。
    fn map_axes(self, transform: impl Fn(&R) -> Option<R>) -> Option<Self> {
        Some(assemble(
            transform(&self.dims)?,
            transform(&self.strides)?,
            self.base,
            self.storage,
        ))
    }

    /// 广播到目标形状；新增或扩展轴使用步长 0。
    #[doc(alias = "expand")]
    pub fn broadcast(self, dims: impl Into<R>) -> Option<Self> {
        let dims = dims.into();
        let leading = dims.as_slice().len().checked_sub(self.rank())?;
        let mut strides = dims.clone();
        for (axis, (&wanted, stride)) in dims
            .as_slice()
            .iter()
            .zip(strides.as_slice_mut())
            .enumerate()
        {
            *stride = broadcast_stride(leading, self.dims(), self.strides(), axis, wanted)?;
        }
        Some(assemble(dims, strides, self.base, self.storage))
    }

    /// 等元素数地改变形状；仅行主序紧凑布局可直接重解释。
    pub fn reshape<R2: Rank>(self, dims: impl Into<R2>) -> Option<Tensor<B, R2, K>> {
        if !self.is_contiguous() {
            return None;
        }
        let dims = dims.into();
        if product(dims.as_slice()) != self.numel() {
            return None;
        }
        let strides = R2::contiguous(dims.as_slice())?;
        Some(assemble(dims, strides, self.base, self.storage))
    }

    /// 沿一轴取 `[start, start + len)`，只前移起点并收窄轴长。
    pub fn narrow(mut self, axis: usize, start: usize, len: usize) -> Option<Self> {
        let (base, dim) = narrow(self.dims(), self.strides(), self.base, axis, start, len)?;
        self.base = base;
        self.dims.as_slice_mut()[axis] = dim;
        Some(self)
    }

    /// 多轴切片，支持 [`Slice::stepped`]；所有切片只修改元数据。
    pub fn slice<S: Into<Slice> + Clone>(self, ranges: impl AsRef<[S]>) -> Option<Self> {
        let layout = self.window(ranges.as_ref())?;
        Some(self.with_layout(layout))
    }

    /// 计算切片窗口，供只读切片和可写窗口共用。
    pub(crate) fn window<S: Into<Slice> + Clone>(&self, ranges: &[S]) -> Option<Layout<R>> {
        if ranges.len() > self.rank() {
            return None;
        }
        window(
            &self.dims,
            &self.strides,
            self.base,
            ranges.iter().cloned().map(Into::into),
        )
    }

    /// 沿 `axis` 等分为至多 `pieces` 个非空借用视图，分段按需惰性构造。
    pub fn chunks(&self, axis: usize, pieces: usize) -> impl Iterator<Item = Tensor<&[K], R, K>> {
        let dim = self.dims().get(axis).copied().unwrap_or(0);
        let size = dim.div_ceil(pieces.max(1));
        self.segments(axis, (0..pieces).map(move |_| size))
    }

    /// 按给定段长依次切分；越界段夹到剩余长度，请求耗尽后停止。
    pub fn split(
        &self,
        axis: usize,
        sizes: impl IntoIterator<Item = usize>,
    ) -> Vec<Tensor<&[K], R, K>> {
        self.segments(axis, sizes).collect()
    }

    /// 惰性分段，调用方选择收集或直接消费。
    fn segments(
        &self,
        axis: usize,
        sizes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = Tensor<&[K], R, K>> {
        let dim = self.dims().get(axis).copied().unwrap_or(0);
        let mut start = 0usize;
        sizes.into_iter().map_while(move |len| {
            let len = len.min(dim.saturating_sub(start));
            if len == 0 {
                return None;
            }
            let piece = self.segment(axis, start, len)?;
            start += len;
            Some(piece)
        })
    }

    /// 构造一个借用分段。
    fn segment(&self, axis: usize, start: usize, len: usize) -> Option<Tensor<&[K], R, K>> {
        assemble(
            self.dims.clone(),
            self.strides.clone(),
            self.base,
            self.storage.slice(),
        )
        .narrow(axis, start, len)
    }

    /// 固定一轴的每个位置，返回全部秩减一借用视图。
    pub fn unbind<R2: Rank>(&self, axis: usize) -> Vec<Tensor<&[K], R2, K>> {
        let dim = self.dims().get(axis).copied().unwrap_or(0);
        (0..dim)
            .filter_map(|idx| {
                let layout = select::<R2>(self.dims(), self.strides(), self.base, axis, idx)?;
                Some(assemble_layout(layout, self.storage.slice()))
            })
            .collect()
    }

    /// 固定一轴并删除它。
    pub fn select<R2: Rank>(self, axis: usize, idx: usize) -> Option<Tensor<B, R2, K>> {
        let layout = select(self.dims(), self.strides(), self.base, axis, idx)?;
        Some(self.with_layout(layout))
    }

    /// 删除长为 1 的一轴。
    pub fn squeeze<R2: Rank>(self, axis: usize) -> Option<Tensor<B, R2, K>> {
        let layout = squeeze(self.dims(), self.strides(), self.base, axis)?;
        Some(self.with_layout(layout))
    }

    /// 两轴取对角，长度取较小值、步长相加，不移动元素。
    #[doc(alias = "diag")]
    pub fn diagonal<R2: Rank>(self, first: usize, second: usize) -> Option<Tensor<B, R2, K>> {
        let layout = diagonal(self.dims(), self.strides(), self.base, first, second)?;
        Some(self.with_layout(layout))
    }

    /// 在指定位置插入一个单例轴。
    pub fn unsqueeze<R2: Rank>(self, axis: usize) -> Option<Tensor<B, R2, K>> {
        let layout = unsqueeze(self.dims(), self.strides(), self.base, axis)?;
        Some(self.with_layout(layout))
    }

    /// 压平行主序紧凑的 `[start, end)` 轴段。
    pub fn flatten<R2: Rank>(self, start: usize, end: usize) -> Option<Tensor<B, R2, K>> {
        let layout = flatten(self.dims(), self.strides(), self.base, start, end)?;
        Some(self.with_layout(layout))
    }

    /// 沿指定轴反转逻辑顺序并物化；无效轴忽略，重复轴只翻一次。
    #[must_use]
    pub fn flip(&self, axes: impl AsRef<[usize]>) -> Tensor<Owned<K>, R, K> {
        let mut reversed_axes = vec![false; self.rank()];
        for &axis in axes.as_ref() {
            if let Some(slot) = reversed_axes.get_mut(axis) {
                *slot = true;
            }
        }
        let source = self.storage.slice();
        let values = Offsets::<R>::new(self.dims(), self.strides(), self.base, &reversed_axes)
            .map(|offset| source[offset])
            .collect();
        self.owned_with_values(values)
    }

    /// 把已经计算好的布局装回张量并保留原存储。
    fn with_layout<R2: Rank>(self, layout: Layout<R2>) -> Tensor<B, R2, K> {
        assemble_layout(layout, self.storage)
    }

    /// 按逻辑序收集元素；连续布局直接整段复制，非连续布局零临时迭代。
    pub fn to_vec(&self) -> Vec<K> {
        if self.is_contiguous() {
            let end = self.base.saturating_add(self.numel());
            return self
                .storage
                .slice()
                .get(self.base..end)
                .unwrap_or_default()
                .to_vec();
        }
        self.iter().collect()
    }

    /// 物化为行主序连续自持张量。
    #[must_use]
    pub fn to_contiguous(&self) -> Tensor<Owned<K>, R, K> {
        self.owned_with_values(self.to_vec())
    }

    /// 借出运行期秩只读视图，不复制元素。
    pub fn as_view(&self) -> View<'_, K> {
        assemble(
            self.dims().to_vec(),
            self.strides().to_vec(),
            self.base,
            self.storage.slice(),
        )
    }

    /// 消费为自持连续张量；已有连续存储直接复用，非连续布局按逻辑序物化。
    #[must_use]
    pub fn into_owned(self) -> Tensor<Owned<K>, R, K> {
        if !self.is_contiguous() {
            return self.to_contiguous();
        }
        let end = self.base.saturating_add(self.numel());
        let mut strides = self.dims.clone();
        accumulate(self.dims(), strides.as_slice_mut(), Order::RowMajor);
        let values = self.storage.into_vec_range(self.base..end);
        assemble(self.dims, strides, 0, Owned::from(values))
    }

    /// 擦除存储与编译期秩，返回后端统一拥有形态。
    pub fn into_dyn(self) -> DynTensor<K> {
        let owned = self.into_owned();
        assemble(
            owned.dims().to_vec(),
            owned.strides().to_vec(),
            owned.base,
            owned.storage,
        )
    }
}

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: StoreMut<K>,
{
    /// 返回底层全部元素的可写切片。
    pub fn as_slice_mut(&mut self) -> &mut [K] {
        self.storage.slice_mut()
    }

    /// 借出运行期秩可写视图，不复制元素。
    pub fn as_view_mut(&mut self) -> ViewMut<'_, K> {
        assemble(
            self.dims().to_vec(),
            self.strides().to_vec(),
            self.base,
            self.storage.slice_mut(),
        )
    }

    /// 将窗口内每个逻辑位置写为 `value`。
    pub fn slice_fill<S: Into<Slice> + Clone>(
        &mut self,
        ranges: impl AsRef<[S]>,
        value: K,
    ) -> Option<()> {
        let layout = self.window(ranges.as_ref())?;
        self.write_window(&layout, core::iter::repeat(value));
        Some(())
    }

    /// 按行主序逻辑顺序给窗口赋值，元素数必须完全一致。
    pub fn slice_assign<S: Into<Slice> + Clone>(
        &mut self,
        ranges: impl AsRef<[S]>,
        values: impl AsRef<[K]>,
    ) -> Option<()> {
        let layout = self.window(ranges.as_ref())?;
        let values = values.as_ref();
        if values.len() != product(layout.dims()) {
            return None;
        }
        self.write_window(&layout, values.iter().copied());
        Some(())
    }

    /// 共用窗口写入循环；调用方负责校验填充值或源元素数量。
    fn write_window(&mut self, layout: &Layout<R>, values: impl IntoIterator<Item = K>) {
        let offsets = Offsets::<R>::new(layout.dims(), layout.strides(), layout.base, &[]);
        let target = self.storage.slice_mut();
        for (offset, value) in offsets.zip(values) {
            target[offset] = value;
        }
    }
}

/// 广播单轴的步长规则。
fn broadcast_stride(
    leading: usize,
    dims: &[usize],
    strides: &[usize],
    axis: usize,
    wanted: usize,
) -> Option<usize> {
    let Some(source) = axis.checked_sub(leading) else {
        return Some(0);
    };
    let existing = *dims.get(source)?;
    if existing == 1 {
        return Some(0);
    }
    if existing != wanted {
        return None;
    }
    strides.get(source).copied()
}

/// 将布局的三个组成部分装回张量。
fn assemble_layout<B, R: Rank, K>(layout: Layout<R>, storage: B) -> Tensor<B, R, K> {
    let (base, dims, strides) = layout.into_parts();
    assemble(dims, strides, base, storage)
}

/// 判断轴序列是否恰好是 `0..rank` 的排列。
fn is_permutation(order: &[usize], rank: usize) -> bool {
    order.len() == rank && (0..rank).all(|axis| order.contains(&axis))
}
