//! 形状：每个轴的长度与元素数 / 字节数推导。

use core::ops::Range;

use dtype::DType;

pub use order::Order;

/// 轴长度的整数类型：元素数乘积 [`product`] 需要的最小接口。
pub trait AxisLen: Copy {
    /// 乘法单位元：空轴序列（标量）的元素个数。
    const ONE: Self;

    /// 饱和乘法。
    fn mul_saturated(self, rhs: Self) -> Self;
}

impl AxisLen for usize {
    const ONE: Self = 1;

    fn mul_saturated(self, rhs: Self) -> Self {
        usize::saturating_mul(self, rhs)
    }
}

impl AxisLen for u64 {
    const ONE: Self = 1;

    fn mul_saturated(self, rhs: Self) -> Self {
        u64::saturating_mul(self, rhs)
    }
}

/// 各轴长度的饱和乘积；空切片为 1（标量）。
#[must_use]
pub fn product<A: AxisLen>(dims: &[A]) -> A {
    dims.iter().fold(A::ONE, |acc, dim| acc.mul_saturated(*dim))
}

/// 按 `order` 把连续步长写进 `steps`：`steps[axis]` 是该侧全部轴长之积。
///
/// 累乘是串行依赖，保留循环。这是"连续"的**唯一**定义：[`Shape::strides`]、编译期秩与运行期秩的
/// [`Rank::contiguous`](crate::Rank::contiguous)、以及 [`Tensor::is_contiguous`](crate::Tensor)
/// 的期望值全部经由本函数，任何一处口径变更不会漏改第二遍。
pub(crate) fn accumulate(dims: &[usize], steps: &mut [usize], order: Order) {
    let mut acc = 1usize;
    for offset in 0..steps.len() {
        // 行主序从最后一轴起累积，列主序从第一轴起——同一段循环，只有轴向不同。
        let axis = match order {
            Order::RowMajor => steps.len() - 1 - offset,
            Order::ColMajor => offset,
        };
        let Some(&dim) = dims.get(axis) else { break };
        steps[axis] = acc;
        acc = acc.saturating_mul(dim);
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

    /// 轴数（rank）。标量为 0。
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// 某一轴的长度；越界返回 `None`。
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// 元素个数：各轴长度之积，空形状为 1。
    pub fn numel(&self) -> usize {
        product(&self.0)
    }

    /// 该元素类型下需要的字节数（按位向上取整，`U4` 会小于 `numel`）。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        element_bytes(self.numel(), dtype)
    }

    /// 按 `order` 推出的连续步长：与 [`Tensor`](crate::Tensor) 构造期同一套累积规则。
    ///
    /// 步长不是另一种形状，就是"每轴一个 `usize`"的同一类数，因此直接给 [`Vec`]——不必再引入
    /// 一个平行容器，再写一份 `new` / `as_slice` / `From`。
    pub fn strides(&self, order: Order) -> Vec<usize> {
        let mut strides = vec![0; self.0.len()];
        accumulate(&self.0, &mut strides, order);
        strides
    }
}

/// 元素个数 → 字节数：`usize` 溢出按上限夹取（`U4` 按位向上取整，会小于 `numel`）。
pub(crate) fn element_bytes(numel: usize, dtype: DType) -> usize {
    let numel = u64::try_from(numel).unwrap_or(u64::MAX);
    usize::try_from(dtype.bytes_for(numel)).unwrap_or(usize::MAX)
}

/// 让运行期秩的 [`Tensor`](crate::Tensor) 直接吃 [`Shape`]，既有调用点不必改写法。
impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

mod order {
    //! 排列顺序：同一组元素在扁平缓冲里沿哪个轴相邻。

    /// 元素在扁平缓冲里的排列顺序。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Order {
        /// 行主序：最后一轴相邻（C / numpy 默认）。
        RowMajor,
        /// 列主序：第一轴相邻（Fortran / cuBLAS 的 `T` 位）。
        ColMajor,
    }
}

/// 一段切片：`start..end` 加上读取步长（Burn 的 `s![..;2]`）。
///
/// 步长只改元数据里的 `steps[axis]`，所以带步长的 [`slice`](crate::Tensor::slice) 依旧一个
/// 元素都不搬——降采样、隔行取列都是视图。`Range<usize>` 经 `From` 直接当连续段用。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// 起始下标（含）。
    pub start: usize,
    /// 结束下标（不含）。
    pub end: usize,
    /// 读取步长：`1` 即连续。
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

    /// 跳着读的段：该轴每 `step` 个位置取一个。
    pub const fn stepped(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }
}

impl From<Range<usize>> for Slice {
    fn from(range: Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}
