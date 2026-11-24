//! 张量本体、存储抽象、构造与只读访问。

use core::{marker::PhantomData, ops::Range};
use diagnostic::Result;

use crate::element::Element;
use crate::{DType, layout};

/// 逻辑张量：存储 `B`、秩见证 `R`、元素类型 `K` 各占一个泛型位。
#[derive(Debug, Clone)]
pub struct Tensor<B, R = Vec<usize>, K = f32> {
    pub(crate) dims: R,
    pub(crate) strides: R,
    /// 下标零对应的存储位置，子视图只前移它而不复制缓冲。
    pub(crate) base: usize,
    pub(crate) storage: B,
    /// 函数返回位置避免 `K` 的自动 trait 泄漏到 `Tensor`。
    marker: PhantomData<fn() -> K>,
}

/// 运行期秩的只读借用视图。
pub type View<'a, K> = Tensor<&'a [K], Vec<usize>, K>;

/// 运行期秩的可写借用视图。
pub type ViewMut<'a, K> = Tensor<&'a mut [K], Vec<usize>, K>;

/// 运行期秩的自持张量，是后端边界的统一拥有形态。
pub type DynTensor<K> = Tensor<Owned<K>, Vec<usize>, K>;

/// 只读元素存储。
pub trait Store<E> {
    /// 返回底层全部元素，而非当前子视图的逻辑元素。
    fn slice(&self) -> &[E];

    /// 消费存储并复制或复用指定范围；越界范围按可得元素截断。
    fn into_vec_range(self, range: Range<usize>) -> Vec<E>
    where
        Self: Sized,
        E: Clone,
    {
        self.slice().get(range).unwrap_or_default().to_vec()
    }
}

/// 可写元素存储。
pub trait StoreMut<E>: Store<E> {
    /// 返回底层全部元素的可写切片。
    fn slice_mut(&mut self) -> &mut [E];
}

impl<E> Store<E> for &[E] {
    fn slice(&self) -> &[E] {
        self
    }
}

impl<E> Store<E> for &mut [E] {
    fn slice(&self) -> &[E] {
        self
    }
}

impl<E> StoreMut<E> for &mut [E] {
    fn slice_mut(&mut self) -> &mut [E] {
        self
    }
}

/// 自己持有的连续元素缓冲。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owned<E>(Vec<E>);

impl<E> Owned<E> {
    /// 分配 `len` 个默认元素；容量不可用时返回错误而不是中止进程。
    pub fn zeros(len: usize) -> Result<Self>
    where
        E: Copy + Default,
    {
        let mut elements = Vec::new();
        elements.try_reserve_exact(len).map_err(|_| {
            crate::invalid(
                format!("{len} 个元素超出可用内存或地址空间"),
                "tensor::Owned::zeros",
            )
        })?;
        elements.resize(len, E::default());
        Ok(Self(elements))
    }

    /// 底层只读切片。
    pub fn as_slice(&self) -> &[E] {
        &self.0
    }

    /// 拆出底层 `Vec`。
    pub fn into_vec(self) -> Vec<E> {
        self.0
    }
}

impl<E> From<Vec<E>> for Owned<E> {
    fn from(elements: Vec<E>) -> Self {
        Self(elements)
    }
}

impl<E> Store<E> for Owned<E> {
    fn slice(&self) -> &[E] {
        &self.0
    }

    fn into_vec(self) -> Vec<E>
    where
        E: Clone,
    {
        self.0
    }
}

impl<E> StoreMut<E> for Owned<E> {
    fn slice_mut(&mut self) -> &mut [E] {
        &mut self.0
    }
}

impl<'a, K: Element> View<'a, K> {
    /// 从切片构造一维连续只读视图，轴长由切片长度给出。
    pub fn one(slice: &'a [K]) -> Self {
        assemble(vec![slice.len()], vec![1], 0, slice)
    }

    /// 构造 `rows × cols` 行主序只读视图。
    pub fn matrix(rows: usize, cols: usize, slice: &'a [K]) -> Result<Self> {
        Self::new(vec![rows, cols], slice)
    }

    /// 构造三条同形二维只读视图，供注意力等算子复用装配逻辑。
    pub fn matrix_triple(
        rows: usize,
        cols: usize,
        q: &'a [K],
        k: &'a [K],
        v: &'a [K],
    ) -> Result<(Self, Self, Self)> {
        Ok((
            Self::matrix(rows, cols, q)?,
            Self::matrix(rows, cols, k)?,
            Self::matrix(rows, cols, v)?,
        ))
    }

    /// 保留原始生命周期地取出借用切片。
    pub fn borrowed_slice(&self) -> &'a [K] {
        self.storage
    }
}

impl<'a, K: Element> ViewMut<'a, K> {
    /// 从切片构造一维连续可写视图。
    pub fn one(slice: &'a mut [K]) -> Self {
        assemble(vec![slice.len()], vec![1], 0, slice)
    }

    /// 构造 `rows × cols` 行主序可写视图。
    pub fn matrix(rows: usize, cols: usize, slice: &'a mut [K]) -> Result<Self> {
        Self::new(vec![rows, cols], slice)
    }

    /// 消费视图并取出原始可写借用。
    pub fn into_slice_mut(self) -> &'a mut [K] {
        self.storage
    }
}

impl<B, R, K, B2, R2> PartialEq<Tensor<B2, R2, K>> for Tensor<B, R, K>
where
    R: Rank,
    R2: Rank,
    K: Element + PartialEq,
    B: Store<K>,
    B2: Store<K>,
{
    /// 比较形状和逻辑序元素，存储与秩见证可以不同。
    fn eq(&self, other: &Tensor<B2, R2, K>) -> bool {
        self.dims() == other.dims() && self.iter().eq(other.iter())
    }
}

impl<B, R, K> Eq for Tensor<B, R, K>
where
    R: Rank,
    K: Element + Eq,
    B: Store<K>,
{
}

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 构造行主序连续张量，元素数必须与形状一致。
    pub fn new(dims: impl Into<R>, storage: B) -> Result<Self> {
        Self::with_order(dims, storage, Order::RowMajor)
    }

    /// 按配置的 `order` 构造连续张量。
    pub fn with_order(dims: impl Into<R>, storage: B, order: Order) -> Result<Self> {
        Self::build(dims.into(), storage, order, "tensor::Tensor::with_order")
    }

    /// 构造任意合法步长的张量，支持转置、广播和非连续子视图。
    pub fn strided(dims: impl Into<R>, strides: impl Into<R>, storage: B) -> Result<Self> {
        let dims = dims.into();
        let strides = strides.into();
        layout::check(
            dims.as_slice(),
            strides.as_slice(),
            product(dims.as_slice()),
            storage.slice().len(),
            "tensor::Tensor::strided",
        )?;
        Ok(assemble(dims, strides, 0, storage))
    }

    /// 校验连续缓冲并推导步长。
    fn build(dims: R, storage: B, order: Order, marker: &str) -> Result<Self> {
        let numel = product(dims.as_slice());
        let len = storage.slice().len();
        if len != numel {
            return Err(element_count_error::<K, _>(&dims, numel, len, marker));
        }
        let strides = R::ordered(dims.as_slice(), order).ok_or_else(|| {
            crate::invalid(
                format!("形状 {:?} 的轴数与秩见证不符", dims.as_slice()),
                marker,
            )
        })?;
        Ok(assemble(dims, strides, 0, storage))
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// 各轴步长。
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    /// 下标零在底层存储里的偏移。
    pub fn base(&self) -> usize {
        self.base
    }

    /// 轴数；标量为 0。
    pub fn rank(&self) -> usize {
        self.dims.as_slice().len()
    }

    /// 逻辑元素个数；标量为 1。
    pub fn numel(&self) -> usize {
        product(self.dims.as_slice())
    }

    /// 元素类型常量。
    pub fn dtype(&self) -> DType {
        K::DTYPE
    }

    /// 底层全部存储；子视图逻辑元素请使用 [`iter`](Self::iter)。
    pub fn as_slice(&self) -> &[K] {
        self.storage.slice()
    }

    /// 是否为行主序紧凑布局。
    pub fn is_contiguous(&self) -> bool {
        layout::is_contiguous(self.dims(), self.strides())
    }

    /// 按下标读取；秩不符、越界或偏移溢出返回 `None`。
    pub fn at(&self, idx: impl AsRef<[usize]>) -> Option<K> {
        let offset = layout::offset(self.dims(), self.strides(), self.base, idx.as_ref())?;
        self.storage.slice().get(offset).copied()
    }

    /// 返回二维张量的连续行。
    pub fn row(&self, row: usize) -> Option<&[K]> {
        layout::row(
            self.dims(),
            self.strides(),
            self.base,
            self.storage.slice(),
            row,
        )
    }

    /// 按行主序逻辑顺序读取元素，不创建中间 `Vec`。
    ///
    /// 连续布局直接迭代原切片；转置、广播和带步长切片使用步长迭代器。
    pub fn iter(&self) -> impl ExactSizeIterator<Item = K> + '_ {
        LogicalValues::<R, K>::new(
            self.dims(),
            self.strides(),
            self.base,
            self.numel(),
            self.is_contiguous(),
            self.storage.slice(),
        )
    }

    /// 用同一形状和连续新存储重建张量。
    pub(crate) fn owned_with_values<O: Element>(&self, storage: Vec<O>) -> Tensor<Owned<O>, R, O> {
        let mut strides = self.dims.clone();
        accumulate(self.dims(), strides.as_slice_mut(), Order::RowMajor);
        assemble(self.dims.clone(), strides, 0, Owned::from(storage))
    }
}

impl<R, K> Tensor<Owned<K>, R, K>
where
    R: Rank,
    K: Element,
{
    /// 按形状分配行主序连续零值张量。
    pub fn zeros(dims: impl Into<R>) -> Result<Self> {
        Self::zeros_with_order(dims, Order::RowMajor)
    }

    /// 按形状和排列顺序分配连续零值张量。
    pub fn zeros_with_order(dims: impl Into<R>, order: Order) -> Result<Self> {
        let dims = dims.into();
        let storage = Owned::zeros(product(dims.as_slice()))?;
        Self::build(dims, storage, order, "tensor::Tensor::zeros_with_order")
    }

    /// 消费张量并返回逻辑序元素；连续布局复用原分配，非连续布局按逻辑序物化。
    pub fn into_vec(self) -> Vec<K> {
        if !self.is_contiguous() {
            return self.to_vec();
        }
        let base = self.base;
        let numel = self.numel();
        let mut values = self.storage.into_vec();
        if base != 0 {
            values.drain(..base);
        }
        values.truncate(numel);
        values
    }
}

/// 连续构造错误的统一文案。
fn element_count_error<K: Element, R: Rank>(
    dims: &R,
    numel: usize,
    len: usize,
    marker: &str,
) -> diagnostic::Error {
    crate::invalid(
        format!(
            "缓冲 {len} 个元素，形状 {:?} 需要 {numel} 个（{:?} 共 {} 字节）",
            dims.as_slice(),
            K::DTYPE,
            element_bytes(numel, K::DTYPE)
        ),
        marker,
    )
}

/// 无校验组装，仅用于已经建立布局不变量的构造与变换。
pub(crate) fn assemble<B, R, K>(dims: R, strides: R, base: usize, storage: B) -> Tensor<B, R, K> {
    Tensor {
        dims,
        strides,
        base,
        storage,
        marker: PhantomData,
    }
}

/// 元素在扁平缓冲里的排列顺序。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// 行主序：最后一轴相邻（C / NumPy 默认）。
    RowMajor,
    /// 列主序：第一轴相邻（Fortran / 列主序矩阵接口）。
    ColMajor,
}

/// 轴长度支持的整数运算。
pub trait AxisLen: Copy {
    /// 空轴序列（标量）的元素数。
    const ONE: Self;

    /// 饱和乘法，避免不受信任的形状回绕成小元素数。
    fn mul_saturated(self, rhs: Self) -> Self;
}

macro_rules! axis_len {
    ($($integer:ty),+ $(,)?) => {
        $(
            impl AxisLen for $integer {
                const ONE: Self = 1;

                fn mul_saturated(self, rhs: Self) -> Self {
                    self.saturating_mul(rhs)
                }
            }
        )+
    };
}

axis_len!(usize, u64);

/// 各轴长度的饱和乘积；空切片为 1（标量）。
#[must_use]
pub fn product<A: AxisLen>(dims: &[A]) -> A {
    dims.iter().fold(A::ONE, |acc, dim| acc.mul_saturated(*dim))
}

/// 按 `order` 原地推导连续步长，是所有构造与连续性判断的唯一规则。
pub(crate) fn accumulate(dims: &[usize], strides: &mut [usize], order: Order) {
    let mut stride = 1usize;
    for offset in 0..strides.len() {
        let axis = match order {
            Order::RowMajor => strides.len() - 1 - offset,
            Order::ColMajor => offset,
        };
        let Some(&dim) = dims.get(axis) else { break };
        strides[axis] = stride;
        stride = stride.saturating_mul(dim);
    }
}

/// 秩见证：编译期定长 `[usize; D]` 或运行期变长 `Vec<usize>`。
///
/// 实现只需提供切片访问和受检切片构造；重排、连续步长与填充均有统一默认实现。
pub trait Rank: Clone {
    /// 轴值切片。
    fn as_slice(&self) -> &[usize];

    /// 可写轴值切片。
    fn as_slice_mut(&mut self) -> &mut [usize];

    /// 从切片构造；长度与目标秩不符时返回 `None`。
    #[must_use]
    fn from_slice(values: &[usize]) -> Option<Self>;

    /// 从迭代器构造；内置秩容器覆盖为零临时集合的高效路径。
    #[must_use]
    fn from_iter(values: impl IntoIterator<Item = usize>) -> Option<Self> {
        let values = values.into_iter().collect::<Vec<_>>();
        Self::from_slice(&values)
    }

    /// 构造 `len` 个相同轴值。
    #[must_use]
    fn filled(len: usize, value: usize) -> Option<Self> {
        Self::from_iter(core::iter::repeat_n(value, len))
    }

    /// 按指定排列顺序推导连续步长。
    #[must_use]
    fn ordered(dims: &[usize], order: Order) -> Option<Self> {
        let mut strides = Self::filled(dims.len(), 0)?;
        accumulate(dims, strides.as_slice_mut(), order);
        Some(strides)
    }

    /// 行主序连续步长。
    #[must_use]
    fn contiguous(dims: &[usize]) -> Option<Self> {
        Self::ordered(dims, Order::RowMajor)
    }

    /// 交换两轴；轴越界返回 `None`。
    #[must_use]
    fn swapped(&self, a: usize, b: usize) -> Option<Self> {
        if a >= self.as_slice().len() || b >= self.as_slice().len() {
            return None;
        }
        let mut out = self.clone();
        out.as_slice_mut().swap(a, b);
        Some(out)
    }

    /// 搬动一轴并保持其余轴的相对顺序，不创建排列索引。
    #[must_use]
    fn moved(&self, from: usize, to: usize) -> Option<Self> {
        let mut out = self.clone();
        let axes = out.as_slice_mut();
        if from >= axes.len() || to >= axes.len() {
            return None;
        }
        let value = axes[from];
        if from < to {
            axes.copy_within(from + 1..=to, from);
        } else {
            axes.copy_within(to..from, to + 1);
        }
        axes[to] = value;
        Some(out)
    }

    /// 按排列重排；长度或轴值不合法返回 `None`。
    #[must_use]
    fn permuted(&self, order: &[usize]) -> Option<Self> {
        let source = self.as_slice();
        if order.len() != source.len() {
            return None;
        }
        let mut out = Self::from_slice(source)?;
        for (target, &source_axis) in out.as_slice_mut().iter_mut().zip(order) {
            *target = *source.get(source_axis)?;
        }
        Some(out)
    }
}

impl Rank for Vec<usize> {
    fn as_slice(&self) -> &[usize] {
        self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        self
    }

    fn from_slice(values: &[usize]) -> Option<Self> {
        Some(values.to_vec())
    }

    fn from_iter(values: impl IntoIterator<Item = usize>) -> Option<Self> {
        Some(values.into_iter().collect())
    }
}

impl<const D: usize> Rank for [usize; D] {
    fn as_slice(&self) -> &[usize] {
        self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        self
    }

    fn from_slice(values: &[usize]) -> Option<Self> {
        Self::try_from(values).ok()
    }

    fn from_iter(values: impl IntoIterator<Item = usize>) -> Option<Self> {
        let mut values = values.into_iter();
        let mut out = [0; D];
        for slot in &mut out {
            *slot = values.next()?;
        }
        values.next().is_none().then_some(out)
    }
}

/// 形状：每个轴的长度。空形状表示标量，`numel()` 为 1。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// 由各轴长度构造。
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// 轴数；标量为 0。
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// 某一轴的长度；越界返回 `None`。
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// 元素个数；空形状为 1。
    pub fn numel(&self) -> usize {
        product(&self.0)
    }

    /// 该元素类型下需要的字节数，亚字节精度按位向上取整。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        element_bytes(self.numel(), dtype)
    }

    /// 按 `order` 推导连续步长。
    pub fn strides(&self, order: Order) -> Vec<usize> {
        let mut strides = vec![0; self.0.len()];
        accumulate(&self.0, &mut strides, order);
        strides
    }
}

/// 元素个数转字节数；地址空间溢出按上限夹取。
pub(crate) fn element_bytes(numel: usize, dtype: DType) -> usize {
    let numel = u64::try_from(numel).unwrap_or(u64::MAX);
    usize::try_from(dtype.bytes_for(numel)).unwrap_or(usize::MAX)
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

/// 半开切片区间及读取步长。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// 起始下标（含）。
    pub start: usize,
    /// 结束下标（不含）。
    pub end: usize,
    /// 读取步长，必须大于 0。
    pub step: usize,
}

impl Slice {
    /// 连续段，等价于 `start..end`。
    pub const fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            step: 1,
        }
    }

    /// 每隔 `step` 个位置读取一次。
    pub const fn stepped(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }
}

impl From<Range<usize>> for Slice {
    fn from(range: Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}

/// 逻辑值迭代器：连续切片快路径与通用步长路径共享一个公开方法。
enum LogicalValues<'a, R, K> {
    Contiguous(core::slice::Iter<'a, K>),
    Strided {
        source: &'a [K],
        offsets: layout::Offsets<'a, R>,
    },
}

impl<'a, R: Rank, K> LogicalValues<'a, R, K> {
    fn new(
        dims: &'a [usize],
        strides: &'a [usize],
        base: usize,
        numel: usize,
        contiguous: bool,
        source: &'a [K],
    ) -> Self {
        if contiguous && let Some(values) = source.get(base..base.saturating_add(numel)) {
            return Self::Contiguous(values.iter());
        }
        Self::Strided {
            source,
            offsets: layout::Offsets::new(dims, strides, base, &[]),
        }
    }
}

impl<R: Rank, K: Copy> Iterator for LogicalValues<'_, R, K> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(values) => values.next().copied(),
            Self::Strided { source, offsets } => offsets.next().map(|offset| source[offset]),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Contiguous(values) => values.size_hint(),
            Self::Strided { offsets, .. } => offsets.size_hint(),
        }
    }
}

impl<R: Rank, K: Copy> ExactSizeIterator for LogicalValues<'_, R, K> {}
