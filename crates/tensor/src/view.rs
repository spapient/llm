//! 零拷贝形状算子、逻辑序物化与可写窗口。

use crate::element::Element;
use crate::shape::Slice;
use crate::tensor::assemble;
use crate::{DynTensor, Owned, Rank, Store, StoreMut, Tensor, View, ViewMut, layout, product};

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 交换两轴，不移动元素。
    #[doc(alias = "swap_dims")]
    pub fn transpose(self, a: usize, b: usize) -> Option<Self> {
        self.reaxis(|axes| axes.swapped(a, b))
    }

    /// 把一轴搬到另一位置，其余轴的相对顺序不变。
    #[doc(alias = "moveaxis")]
    pub fn movedim(self, from: usize, to: usize) -> Option<Self> {
        self.reaxis(|axes| axes.moved(from, to))
    }

    /// 按完整轴排列重排元数据。
    pub fn permute(self, order: impl AsRef<[usize]>) -> Option<Self> {
        let order = order.as_ref();
        if !is_permutation(order, self.rank()) {
            return None;
        }
        self.reaxis(|axes| axes.permuted(order))
    }

    /// 对轴长和步长施加同一个变换，避免两套元数据漂移。
    fn reaxis(self, transform: impl Fn(&R) -> Option<R>) -> Option<Self> {
        Some(assemble(
            transform(&self.dims)?,
            transform(&self.steps)?,
            self.base,
            self.buf,
        ))
    }

    /// 广播到目标形状；新增或扩展轴使用步长 0。
    #[doc(alias = "expand")]
    pub fn broadcast(self, dims: impl Into<R>) -> Option<Self> {
        let dims = dims.into();
        let lead = dims.as_slice().len().checked_sub(self.rank())?;
        let mut steps = dims.clone();
        for (axis, (&want, step)) in dims.as_slice().iter().zip(steps.as_slice_mut()).enumerate() {
            *step = broadcast_step((lead, self.dims(), self.strides()), (axis, want))?;
        }
        Some(assemble(dims, steps, self.base, self.buf))
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
        let steps = R2::contiguous(dims.as_slice())?;
        Some(assemble(dims, steps, self.base, self.buf))
    }

    /// 沿一轴取 `[start, start + len)`，只前移起点并收窄轴长。
    pub fn narrow(mut self, axis: usize, start: usize, len: usize) -> Option<Self> {
        let (base, dim) = layout::narrow(self.dims(), self.strides(), self.base, axis, start, len)?;
        self.base = base;
        self.dims.as_slice_mut()[axis] = dim;
        Some(self)
    }

    /// 多轴切片，支持 [`Slice::stepped`]；所有切片只修改元数据。
    pub fn slice<S: Into<Slice> + Clone>(self, ranges: impl AsRef<[S]>) -> Option<Self> {
        let out = self.window(ranges.as_ref())?;
        Some(self.respine(out))
    }

    /// 计算切片窗口，供只读切片和可写窗口共用。
    pub(crate) fn window<S: Into<Slice> + Clone>(&self, ranges: &[S]) -> Option<layout::Layout<R>> {
        if ranges.len() > self.rank() {
            return None;
        }
        layout::window(
            &self.dims,
            &self.steps,
            self.base,
            ranges.iter().cloned().map(Into::into),
        )
    }

    /// 沿 `axis` 等分为至多 `pieces` 个非空借用视图。
    pub fn chunks(&self, axis: usize, pieces: usize) -> impl Iterator<Item = Tensor<&[K], R, K>> {
        let dim = self.dims().get(axis).copied().unwrap_or(0);
        let size = dim.div_ceil(pieces.max(1));
        self.split(axis, (0..pieces).map(move |_| size)).into_iter()
    }

    /// 按给定段长依次切分；越界段夹到剩余长度，请求耗尽后停止。
    pub fn split(
        &self,
        axis: usize,
        sizes: impl IntoIterator<Item = usize>,
    ) -> Vec<Tensor<&[K], R, K>> {
        let dim = self.dims().get(axis).copied().unwrap_or(0);
        let mut start = 0usize;
        let mut parts = Vec::new();
        for len in sizes {
            let len = len.min(dim.saturating_sub(start));
            if len == 0 {
                break;
            }
            let Some(piece) = self.segment(axis, start, len) else {
                break;
            };
            parts.push(piece);
            start += len;
        }
        parts
    }

    /// 构造一个借用分段。
    fn segment(&self, axis: usize, start: usize, len: usize) -> Option<Tensor<&[K], R, K>> {
        assemble(
            self.dims.clone(),
            self.steps.clone(),
            self.base,
            self.buf.slice(),
        )
        .narrow(axis, start, len)
    }

    /// 固定一轴的每个位置，返回全部秩减一借用视图。
    pub fn unbind<R2: Rank>(&self, axis: usize) -> Vec<Tensor<&[K], R2, K>> {
        let dim = self.dims().get(axis).copied().unwrap_or(0);
        (0..dim)
            .filter_map(|idx| {
                let out = layout::select::<R2>(self.dims(), self.strides(), self.base, axis, idx)?;
                Some(seat(out, self.buf.slice()))
            })
            .collect()
    }

    /// 固定一轴并删除它。
    pub fn select<R2: Rank>(self, axis: usize, idx: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::select(self.dims(), self.strides(), self.base, axis, idx)?;
        Some(self.respine(out))
    }

    /// 删除长为 1 的一轴。
    pub fn squeeze<R2: Rank>(self, axis: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::squeeze(self.dims(), self.strides(), self.base, axis)?;
        Some(self.respine(out))
    }

    /// 两轴取对角，长度取较小值、步长相加，不移动元素。
    #[doc(alias = "diag")]
    pub fn diagonal<R2: Rank>(self, a: usize, b: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::diagonal(self.dims(), self.strides(), self.base, a, b)?;
        Some(self.respine(out))
    }

    /// 在指定位置插入一个单例轴。
    pub fn unsqueeze<R2: Rank>(self, axis: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::unsqueeze(self.dims(), self.strides(), self.base, axis)?;
        Some(self.respine(out))
    }

    /// 压平行主序紧凑的 `[start, end)` 轴段。
    pub fn flatten<R2: Rank>(self, start: usize, end: usize) -> Option<Tensor<B, R2, K>> {
        let out = layout::flatten(self.dims(), self.strides(), self.base, start, end)?;
        Some(self.respine(out))
    }

    /// 沿指定轴反转逻辑顺序并物化；无效轴忽略，重复轴只翻一次。
    #[must_use]
    pub fn flip(&self, axes: impl AsRef<[usize]>) -> Tensor<Owned<K>, R, K> {
        let mut turn = vec![false; self.rank()];
        for &axis in axes.as_ref() {
            if let Some(slot) = turn.get_mut(axis) {
                *slot = true;
            }
        }
        let source = self.buf.slice();
        let values = layout::Offsets::<R>::new(self.dims(), self.strides(), self.base, &turn)
            .map(|offset| source[offset])
            .collect();
        self.respell(values)
    }

    /// 把已经计算好的布局装回张量并保留原存储。
    fn respine<R2: Rank>(self, out: layout::Layout<R2>) -> Tensor<B, R2, K> {
        seat(out, self.buf)
    }

    /// 按逻辑序收集元素；连续布局直接整段复制，非连续布局零临时迭代。
    pub fn to_vec(&self) -> Vec<K> {
        if self.is_contiguous() {
            let end = self.base.saturating_add(self.numel());
            return self
                .buf
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
        self.respell(self.to_vec())
    }

    /// 借出运行期秩只读视图，不复制元素。
    pub fn as_view(&self) -> View<'_, K> {
        assemble(
            self.dims().to_vec(),
            self.strides().to_vec(),
            self.base,
            self.buf.slice(),
        )
    }

    /// 物化为自持连续张量。
    #[must_use]
    pub fn into_owned(self) -> Tensor<Owned<K>, R, K> {
        self.to_contiguous()
    }

    /// 擦除存储与编译期秩，返回后端统一拥有形态。
    pub fn into_dyn(self) -> DynTensor<K> {
        let owned = self.to_contiguous();
        assemble(
            owned.dims().to_vec(),
            owned.strides().to_vec(),
            owned.base,
            owned.buf,
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
        self.buf.slice_mut()
    }

    /// 借出运行期秩可写视图，不复制元素。
    pub fn as_view_mut(&mut self) -> ViewMut<'_, K> {
        assemble(
            self.dims().to_vec(),
            self.strides().to_vec(),
            self.base,
            self.buf.slice_mut(),
        )
    }

    /// 将窗口内每个逻辑位置写为 `value`。
    pub fn slice_fill<S: Into<Slice> + Clone>(
        &mut self,
        ranges: impl AsRef<[S]>,
        value: K,
    ) -> Option<()> {
        let out = self.window(ranges.as_ref())?;
        let offsets = layout::Offsets::<R>::new(out.dims(), out.steps(), out.base, &[]);
        let target = self.buf.slice_mut();
        for offset in offsets {
            target[offset] = value;
        }
        Some(())
    }

    /// 按行主序逻辑顺序给窗口赋值，元素数必须完全一致。
    pub fn slice_assign<S: Into<Slice> + Clone>(
        &mut self,
        ranges: impl AsRef<[S]>,
        values: impl AsRef<[K]>,
    ) -> Option<()> {
        let out = self.window(ranges.as_ref())?;
        let values = values.as_ref();
        if values.len() != product(out.dims()) {
            return None;
        }
        let offsets = layout::Offsets::<R>::new(out.dims(), out.steps(), out.base, &[]);
        let target = self.buf.slice_mut();
        for (offset, value) in offsets.zip(values) {
            target[offset] = *value;
        }
        Some(())
    }
}

/// 广播单轴的步长规则。
fn broadcast_step(
    (lead, dims, steps): (usize, &[usize], &[usize]),
    (axis, want): (usize, usize),
) -> Option<usize> {
    let Some(source) = axis.checked_sub(lead) else {
        return Some(0);
    };
    let have = *dims.get(source)?;
    if have == 1 {
        return Some(0);
    }
    if have != want {
        return None;
    }
    steps.get(source).copied()
}

/// 将布局的三个组成部分装回张量。
fn seat<B, R: Rank, K>(out: layout::Layout<R>, buf: B) -> Tensor<B, R, K> {
    let (base, dims, steps) = out.into_parts();
    assemble(dims, steps, base, buf)
}

/// 判断轴序列是否恰好是 `0..rank` 的排列。
fn is_permutation(order: &[usize], rank: usize) -> bool {
    order.len() == rank && (0..rank).all(|axis| order.contains(&axis))
}
