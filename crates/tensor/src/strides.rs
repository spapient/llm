//! 步长：沿每个轴前进一个下标要走多少个元素；连续性判定的唯一实现。

use crate::shape::Shape;

/// 步长：沿每个轴前进一个下标要走多少个元素。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<usize>);

impl Strides {
    /// 由各轴步长构造（不做合法性校验：非连续布局也是合法布局）。
    pub fn new(strides: impl Into<Vec<usize>>) -> Self {
        Self(strides.into())
    }

    /// 行主序连续布局：最后一轴步长为 1，向前累乘。
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![0; dims.len()];
        // 反向累积存在串行依赖（下一轴 = 本轴长度 × 当前累积），保留循环；
        // 形状维度数据相关，无法整块展开。
        let mut acc = 1usize;
        for axis in (0..dims.len()).rev() {
            strides[axis] = acc;
            acc = acc.saturating_mul(dims[axis]);
        }
        Self(strides)
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
