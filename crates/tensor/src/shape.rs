//! 形状：每个轴的长度与元素数 / 字节数推导。

use dtype::DType;

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
///
/// 饱和而不是回绕：长度可能来自不受信任的权重文件头，溢出必须夹到上限让下游的容量检查报错，
/// 不能乘出一个偏小的数骗过它。
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
    ///
    /// 宽度算术只有 [`DType::bytes_for`] 一份：这里不重写「位宽 × 元素数 ÷ 8 再取整」，
    /// 否则亚字节精度会在两处各有一套进位规则。
    pub fn element_bytes(&self, dtype: DType) -> usize {
        let numel = u64::try_from(self.numel()).unwrap_or(u64::MAX);
        usize::try_from(dtype.bytes_for(numel)).unwrap_or(usize::MAX)
    }
}
