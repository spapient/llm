//! 张量本体、存储抽象、构造与只读访问。

use core::marker::PhantomData;
use diagnostic::Result;

use crate::element::Element;
use crate::shape::{Order, accumulate, element_bytes};
use crate::{DType, Rank, Tensor, layout, product};

/// 逻辑张量：存储 `B`、秩见证 `R`、元素类型 `K` 各占一个泛型位。
#[derive(Debug, Clone)]
pub struct Tensor<B, R = Vec<usize>, K = f32> {
    pub(crate) dims: R,
    pub(crate) steps: R,
    /// 下标零对应的存储位置，子视图只前移它而不复制缓冲。
    pub(crate) base: usize,
    pub(crate) buf: B,
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
        self.buf
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
        self.buf
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
    pub fn new(dims: impl Into<R>, buf: B) -> Result<Self> {
        Self::with_order(dims, buf, Order::RowMajor)
    }

    /// 按配置的 `order` 构造连续张量。
    pub fn with_order(dims: impl Into<R>, buf: B, order: Order) -> Result<Self> {
        Self::build(dims.into(), buf, order, "tensor::Tensor::with_order")
    }

    /// 构造任意合法步长的张量，支持转置、广播和非连续子视图。
    pub fn strided(dims: impl Into<R>, steps: impl Into<R>, buf: B) -> Result<Self> {
        let dims = dims.into();
        let steps = steps.into();
        layout::check(
            dims.as_slice(),
            steps.as_slice(),
            product(dims.as_slice()),
            buf.slice().len(),
            "tensor::Tensor::strided",
        )?;
        Ok(assemble(dims, steps, 0, buf))
    }

    /// 校验连续缓冲并推导步长。
    fn build(dims: R, buf: B, order: Order, marker: &str) -> Result<Self> {
        let numel = product(dims.as_slice());
        let len = buf.slice().len();
        if len != numel {
            return Err(invalid_message(&dims, numel, len, marker));
        }
        let steps = R::ordered(dims.as_slice(), order).ok_or_else(|| {
            crate::invalid(
                format!("形状 {:?} 的轴数与秩见证不符", dims.as_slice()),
                marker,
            )
        })?;
        Ok(assemble(dims, steps, 0, buf))
    }

    /// 各轴长度。
    pub fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// 各轴步长。
    pub fn strides(&self) -> &[usize] {
        self.steps.as_slice()
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
        self.buf.slice()
    }

    /// 是否为行主序紧凑布局。
    pub fn is_contiguous(&self) -> bool {
        layout::is_contiguous(self.dims(), self.strides())
    }

    /// 按下标读取；秩不符、越界或偏移溢出返回 `None`。
    pub fn at(&self, idx: impl AsRef<[usize]>) -> Option<K> {
        let offset = layout::offset(self.dims(), self.strides(), self.base, idx.as_ref())?;
        self.buf.slice().get(offset).copied()
    }

    /// 返回二维张量的连续行。
    pub fn row(&self, row: usize) -> Option<&[K]> {
        layout::row(
            self.dims(),
            self.strides(),
            self.base,
            self.buf.slice(),
            row,
        )
    }

    /// 按行主序逻辑顺序读取元素，不创建中间 `Vec`。
    ///
    /// 转置、广播和带步长切片都按各自元数据遍历；连续物化仍走整段拷贝快路径。
    pub fn iter(&self) -> impl ExactSizeIterator<Item = K> + '_ {
        let source = self.buf.slice();
        layout::Offsets::<R>::new(self.dims(), self.strides(), self.base, &[])
            .map(move |offset| source[offset])
    }

    /// 用同一形状和连续新存储重建张量。
    pub(crate) fn respell<O: Element>(&self, buf: Vec<O>) -> Tensor<Owned<O>, R, O> {
        let mut steps = self.dims.clone();
        accumulate(self.dims(), steps.as_slice_mut(), Order::RowMajor);
        assemble(self.dims.clone(), steps, 0, Owned::from(buf))
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
        let buf = Owned::zeros(product(dims.as_slice()))?;
        Self::build(dims, buf, order, "tensor::Tensor::zeros_with_order")
    }

    /// 消费张量并返回逻辑序元素；完整连续自持张量可直接交出原缓冲。
    pub fn into_vec(self) -> Vec<K> {
        if self.base == 0 && self.is_contiguous() && self.buf.slice().len() == self.numel() {
            self.buf.into_vec()
        } else {
            self.to_vec()
        }
    }
}

/// 连续构造错误的统一文案。
fn invalid_message<R: Rank>(dims: &R, numel: usize, len: usize, marker: &str) -> diagnostic::Error {
    crate::invalid(
        format!(
            "缓冲 {len} 个元素，形状 {:?} 需要 {numel} 个（{:?} 共 {} 字节）",
            dims.as_slice(),
            "元素",
            element_bytes(numel, DType::F32)
        ),
        marker,
    )
}

/// 无校验组装，仅用于已经建立布局不变量的构造与变换。
pub(crate) fn assemble<B, R, K>(dims: R, steps: R, base: usize, buf: B) -> Tensor<B, R, K> {
    Tensor {
        dims,
        steps,
        base,
        buf,
        marker: PhantomData,
    }
}
