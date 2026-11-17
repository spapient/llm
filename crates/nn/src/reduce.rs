//! 归约层：把一维输入折叠成单个标量输出。
//!
//! 与单轴族（[`crate::pointwise`]）共享 [`crate::axis::AxisCore`]，差别只在输出形状：归约消掉
//! 输入轴 `n`，输出轴 `o` 恒为 1，因此层的声明形状是标量 [`Shape`]（空轴）。下界仍由归约算子
//! 的语义表推出，层不手写公式。

use crate::ParamRole;
use crate::axis::{AxisCore, OutputShape};
use crate::layer_delegate;

/// 求和归约：`y = Σ a[0..n]`，输出单个标量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceSum {
    inner: AxisCore,
}
layer_delegate!(
    ReduceSum,
    "reduce_sum_f32",
    "归约层输入",
    OutputShape::Scalar,
    ParamRole::Trainable
);

/// 最大值归约：`y = max(a[0..n])`，输出单个标量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceMax {
    inner: AxisCore,
}
layer_delegate!(
    ReduceMax,
    "reduce_max_f32",
    "归约层输入",
    OutputShape::Scalar,
    ParamRole::Trainable
);

#[cfg(test)]
#[path = "../tests/reduce_tests.rs"]
mod tests;
