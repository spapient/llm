//! 形状：每个轴的长度与元素数 / 字节数推导。

use dtype::DType;

pub use order::Order;
pub use strides::Strides;

/// 轴长度的整数类型：元素数乘积 [`product`] 需要的最小接口。
///
/// 只有 `usize` 与 `u64` 两臂：前者是本层 [`Shape`] 的轴宽，后者是权重文件头的轴宽（线上格式
/// 就是 `u64`）。两臂共用同一条饱和乘积，既不在小位宽目标上因「先换算再乘」提前饱和，也不给
/// 第二份 `fold(saturating_mul)` 留下自己漂移的机会。
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
///
/// 全仓元素个数的唯一实现：`Shape::numel` 与权重段的 `numel` 都走这里。溢出时**饱和**而不
/// 回绕——形状可以来自权重文件头这类不受信任的输入，回绕会让一个巨大形状报出很小的元素数
/// （例如 `[2^32, 2^32]` 得到 0），于是「缓冲够长」的校验被绕过。
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
    ///
    /// 溢出的饱和口径由 [`product`] 唯一给出：回绕会让一个巨大形状报出很小的 `numel`（例如
    /// `[2^32, 2^32]` 得到 0），于是 [`crate::Tensor::new`] / [`crate::View::new`] 的"缓冲够长"判定
    /// 被绕过，得到一个形状与缓冲不符的张量。饱和之后下游只会按"缓冲不够"拒绝，而
    /// `checked_mul` 的调用点照旧能看见溢出。
    pub fn numel(&self) -> usize {
        product(&self.0)
    }

    /// 该元素类型下需要的字节数（按位向上取整，`U4` 会小于 `numel`）。
    ///
    /// 公式的唯一实现在 L1 的 [`DType::bytes_for`]：这里只把 `usize` 的元素数搬进它接受
    /// 的 `u64`，再把结果搬回来——两处各写一遍 `(numel * bits).div_ceil(8)` 迟早会漂移，
    /// 而装载方的缓冲校验正是按 `numel` 与字节数比对。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        let numel = u64::try_from(self.numel()).unwrap_or(u64::MAX);
        usize::try_from(dtype.bytes_for(numel)).unwrap_or(usize::MAX)
    }
}

mod order {
    //! 排列顺序：同一组元素在扁平缓冲里沿哪个轴相邻。
    //!
    //! 这是纯形状元数据，不涉及缓冲与分配，因此归 `tensor`；`storage` 把它随布局再导出，
    //! 不另持一份枚举。

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
        ///
        /// 累积存在串行依赖（下一轴 = 本轴长度 × 当前累积），保留循环；形状维度数据相关，
        /// 无法整块展开。饱和乘积与 [`Shape::numel`](crate::Shape::numel) 同一套溢出口径。
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
