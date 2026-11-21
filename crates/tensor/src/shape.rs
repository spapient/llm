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
        let numel = u64::try_from(self.numel()).unwrap_or(u64::MAX);
        usize::try_from(dtype.bytes_for(numel)).unwrap_or(usize::MAX)
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
    //! 步长：沿每个轴前进一个下标要走多少个元素；连续性判定与顺序累积的唯一实现。

    use super::Order;
    use crate::shape::Shape;

    /// 步长：沿每个轴前进一个下标要走多少个元素。
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Strides(Vec<usize>);

    impl Strides {
        /// 由各轴步长构造（不做合法性校验：非连续布局也是合法布局）。
        pub fn new(strides: impl Into<Vec<usize>>) -> Self {
            Self(strides.into())
        }

        /// 按排列顺序推出的连续步长：行主序反向累乘，列主序正向累乘。
        pub fn ordered(shape: &Shape, order: Order) -> Self {
            let dims = shape.dims();
            let mut strides = vec![0; dims.len()];
            let mut acc = 1usize;
            match order {
                Order::RowMajor => {
                    for axis in (0..dims.len()).rev() {
                        strides[axis] = acc;
                        acc = acc.saturating_mul(dims[axis]);
                    }
                }
                Order::ColMajor => {
                    for axis in 0..dims.len() {
                        strides[axis] = acc;
                        acc = acc.saturating_mul(dims[axis]);
                    }
                }
            }
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

        /// 是否为行主序连续布局：与 [`contiguous`](Self::contiguous) 同一套期望累积值，
        /// 逐轴无分配核对，不再临时构造一条 `Vec` 再比较。层内收口实现，不进公开面。
        #[must_use]
        pub(crate) fn is_contiguous(&self, shape: &Shape) -> bool {
            if self.0.len() != shape.rank() {
                return false;
            }
            let dims = shape.dims();
            // 反向核对期望累积值；串行依赖同上，保留循环。
            let mut acc = 1usize;
            for axis in (0..dims.len()).rev() {
                if self.0[axis] != acc {
                    return false;
                }
                acc = acc.saturating_mul(dims[axis]);
            }
            true
        }
    }
}
