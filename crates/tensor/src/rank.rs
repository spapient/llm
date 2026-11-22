//! 秩见证：`Tensor` 的第二个泛型位。
//!
//! `[usize; D]` 是编译期秩：形状与步长是栈上定长数组，构造不碰堆，秩错在编译期拦下。
//! `Vec<usize>` 是运行期秩：batch / segment 几何要到运行期才知道（ADR 0046）。
//!
//! Burn 的 `Tensor<B, const D, K>` 只有前者；同一个 [`Tensor`](crate::Tensor) 同时接受两种是这里的
//! 超集。两种见证只差"秩从哪来"——偏移、可达性、连续性与取行收口在 [`crate::layout`]，
//! 连续步长收口在 [`crate::shape::accumulate`]，本模块只负责"往哪个容器里装"。

use crate::shape::{Order, accumulate};

/// 秩见证：一个长度等于秩的 `usize` 容器。
pub trait Rank: Clone {
    /// 各轴长度 / 各轴步长——本容器里的数。
    fn as_slice(&self) -> &[usize];

    /// 行主序连续步长；轴数与秩不符返回 `None`。
    #[must_use]
    fn contiguous(dims: &[usize]) -> Option<Self>;

    /// 交换两轴；轴越界返回 `None`。
    #[must_use]
    fn swapped(&self, a: usize, b: usize) -> Option<Self>;

    /// 按 `order` 重排；元素个数不符或取值越界返回 `None`（排列性由调用方先验）。
    #[must_use]
    fn permuted(&self, order: &[usize]) -> Option<Self>;
}

impl Rank for Vec<usize> {
    fn as_slice(&self) -> &[usize] {
        self
    }

    fn contiguous(dims: &[usize]) -> Option<Self> {
        let mut steps = vec![0; dims.len()];
        accumulate(dims, &mut steps, Order::RowMajor);
        Some(steps)
    }

    fn swapped(&self, a: usize, b: usize) -> Option<Self> {
        if a >= self.len() || b >= self.len() {
            return None;
        }
        let mut out = self.clone();
        out.swap(a, b);
        Some(out)
    }

    fn permuted(&self, order: &[usize]) -> Option<Self> {
        order.iter().map(|&from| self.get(from).copied()).collect()
    }
}

impl<const D: usize> Rank for [usize; D] {
    fn as_slice(&self) -> &[usize] {
        self
    }

    fn contiguous(dims: &[usize]) -> Option<Self> {
        if dims.len() != D {
            return None;
        }
        let mut steps = [0usize; D];
        accumulate(dims, &mut steps, Order::RowMajor);
        Some(steps)
    }

    fn swapped(&self, a: usize, b: usize) -> Option<Self> {
        if a >= D || b >= D {
            return None;
        }
        let mut out = *self;
        out.swap(a, b);
        Some(out)
    }

    fn permuted(&self, order: &[usize]) -> Option<Self> {
        let mut out = [0usize; D];
        for (axis, &from) in order.iter().enumerate() {
            out[axis] = *self.get(from)?;
        }
        Some(out)
    }
}
