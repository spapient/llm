//! 秩见证：`Tensor` 的第二个泛型位。
//!
//! `[usize; D]` 是编译期秩：形状与步长是栈上定长数组，构造不碰堆，秩错在编译期拦下。
//! `Vec<usize>` 是运行期秩：batch / segment 几何要到运行期才知道（ADR 0046）。
//!
//! Burn 的 `Tensor<B, const D, K>` 只有前者；同一个 [`Tensor`](crate::Tensor) 同时接受两种是这里的
//! 超集。两种见证只差"秩从哪来"——偏移、可达性、连续性与取行收口在 [`crate::layout`]，
//! 连续步长收口在 [`crate::shape::accumulate`]，本模块只负责"往哪个容器里装"。
//!
//! 于是三个重排动作（`contiguous` / `swapped` / `permuted`）写成**默认方法**：两种容器各只实现
//! `as_slice` / `as_slice_mut` / `from_slice` 三行，重排逻辑只有一份。

use crate::shape::{Order, accumulate};

/// 秩见证：一个长度等于秩的 `usize` 容器。
pub trait Rank: Clone {
    /// 各轴长度 / 各轴步长——本容器里的数。
    fn as_slice(&self) -> &[usize];

    /// 同一个容器，可写。
    fn as_slice_mut(&mut self) -> &mut [usize];

    /// 用给定的数构造同秩容器；长度与秩不符返回 `None`（这就是编译期秩的检查点）。
    #[must_use]
    fn from_slice(vals: &[usize]) -> Option<Self>;

    /// 行主序连续步长；轴数与秩不符返回 `None`。
    #[must_use]
    fn contiguous(dims: &[usize]) -> Option<Self> {
        let mut steps = Self::from_slice(dims)?;
        accumulate(dims, steps.as_slice_mut(), Order::RowMajor);
        Some(steps)
    }

    /// 交换两轴；轴越界返回 `None`。
    #[must_use]
    fn swapped(&self, a: usize, b: usize) -> Option<Self> {
        let len = self.as_slice().len();
        if a >= len || b >= len {
            return None;
        }
        let mut out = self.clone();
        out.as_slice_mut().swap(a, b);
        Some(out)
    }

    /// 按 `order` 重排；长度不符或取值越界返回 `None`（排列性由调用方先验）。
    #[must_use]
    fn permuted(&self, order: &[usize]) -> Option<Self> {
        let src = self.as_slice();
        if order.len() != src.len() {
            return None;
        }
        let mut out = Self::from_slice(src)?;
        let dst = out.as_slice_mut();
        for (axis, &from) in order.iter().enumerate() {
            dst[axis] = *src.get(from)?;
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

    fn from_slice(vals: &[usize]) -> Option<Self> {
        Some(vals.to_vec())
    }
}

impl<const D: usize> Rank for [usize; D] {
    fn as_slice(&self) -> &[usize] {
        self
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        self
    }

    fn from_slice(vals: &[usize]) -> Option<Self> {
        <[usize; D]>::try_from(vals).ok()
    }
}
