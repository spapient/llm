//! 形状、切片与秩见证：轴长推导和轴容器只在这里定义。

use core::ops::Range;

use dtype::DType;

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
pub(crate) fn accumulate(dims: &[usize], steps: &mut [usize], order: Order) {
    let mut stride = 1usize;
    for offset in 0..steps.len() {
        let axis = match order {
            Order::RowMajor => steps.len() - 1 - offset,
            Order::ColMajor => offset,
        };
        let Some(&dim) = dims.get(axis) else { break };
        steps[axis] = stride;
        stride = stride.saturating_mul(dim);
    }
}

/// 秩见证：编译期定长 `[usize; D]` 或运行期变长 `Vec<usize>`。
///
/// 实现只需提供切片访问和受检迭代构造；重排、连续步长与填充均有统一默认实现。
pub trait Rank: Clone {
    /// 轴值切片。
    fn as_slice(&self) -> &[usize];

    /// 可写轴值切片。
    fn as_slice_mut(&mut self) -> &mut [usize];

    /// 从迭代器构造；元素数与目标秩不符时返回 `None`。
    #[must_use]
    fn from_iter(values: impl IntoIterator<Item = usize>) -> Option<Self>;

    /// 从切片构造；长度与目标秩不符时返回 `None`。
    #[must_use]
    fn from_slice(values: &[usize]) -> Option<Self> {
        Self::from_iter(values.iter().copied())
    }

    /// 构造 `len` 个相同轴值。
    #[must_use]
    fn filled(len: usize, value: usize) -> Option<Self> {
        Self::from_iter(core::iter::repeat_n(value, len))
    }

    /// 按指定排列顺序推导连续步长。
    #[must_use]
    fn ordered(dims: &[usize], order: Order) -> Option<Self> {
        let mut steps = Self::filled(dims.len(), 0)?;
        accumulate(dims, steps.as_slice_mut(), order);
        Some(steps)
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
        Self { start, end, step: 1 }
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
