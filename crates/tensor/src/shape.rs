//! 形状：每个轴的长度与元素数 / 字节数推导。

use dtype::DType;

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
    /// 溢出时**饱和**到 `usize::MAX`，不回绕：形状可以来自权重文件头这类不受信任的输入，
    /// 回绕会让一个巨大形状报出很小的 `numel`（例如 `[2^32, 2^32]` 得到 0），于是
    /// [`crate::Tensor::new`] / [`crate::View::new`] 的"缓冲够长"判定被绕过，得到一个形状与缓冲不符的
    /// 张量。饱和之后下游只会按"缓冲不够"拒绝，而 `checked_mul` 的调用点照旧能看见溢出。
    pub fn numel(&self) -> usize {
        self.0
            .iter()
            .fold(1usize, |acc, dim| acc.saturating_mul(*dim))
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
