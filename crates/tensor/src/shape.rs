//! 形状：每个轴的长度与元素数 / 字节数推导。

use dtype::DType;

pub use order::Order;
pub use strides::Strides;

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
/// 累乘是串行依赖，保留循环。这是"连续"的**唯一**定义：`Strides::ordered`、编译期秩与运行期秩的
/// [`Rank::contiguous`](crate::Rank::contiguous)、以及 [`Tensor::is_contiguous`](crate::Tensor)
/// 的期望值全部经由本函数，任何一处口径变更不会漏改第二遍。
pub(crate) fn accumulate(dims: &[usize], steps: &mut [usize], order: Order) {
    let mut acc = 1usize;
    match order {
        Order::RowMajor => {
            for (step, &dim) in steps.iter_mut().zip(dims.iter()).rev() {
                *step = acc;
                acc = acc.saturating_mul(dim);
            }
        }
        Order::ColMajor => {
            for (step, &dim) in steps.iter_mut().zip(dims.iter()) {
                *step = acc;
                acc = acc.saturating_mul(dim);
            }
        }
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

mod strides {
    //! 步长：沿每个轴前进一个下标要走多少个元素。

    use super::{Order, Shape, accumulate};

    /// 步长：沿每个轴前进一个下标要走多少个元素。
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Strides(Vec<usize>);

    impl Strides {
        /// 由各轴步长构造（不做合法性校验：非连续布局也是合法布局）。
        pub fn new(strides: impl Into<Vec<usize>>) -> Self {
            Self(strides.into())
        }

        /// 按排列顺序推出的连续步长。
        pub fn ordered(shape: &Shape, order: Order) -> Self {
            let mut strides = vec![0; shape.rank()];
            accumulate(shape.dims(), &mut strides, order);
            Self(strides)
        }

        /// 行主序连续布局：最后一轴步长为 1，向前累乘。
        pub fn contiguous(shape: &Shape) -> Self {
            Self::ordered(shape, Order::RowMajor)
        }

        /// 各轴步长。
        pub fn as_slice(&self) -> &[usize] {
            &self.0
        }

        /// 某一轴的步长；越界返回 `None`。
        pub fn get(&self, axis: usize) -> Option<usize> {
            self.0.get(axis).copied()
        }
    }

    /// 让运行期秩的 [`Tensor`](crate::Tensor) 直接吃 [`Strides`]。
    impl From<Strides> for Vec<usize> {
        fn from(strides: Strides) -> Self {
            strides.0
        }
    }
}
