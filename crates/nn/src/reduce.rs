//! 归约层：把一维输入折叠成单个标量输出。
//!
//! 与单轴族（[`crate::pointwise`]）共享 [`crate::axis::AxisCore`]，差别只在输出形状：归约消掉
//! 输入轴 `n`，输出轴 `o` 恒为 1，因此层的声明形状是标量 [`Shape`]（空轴）。下界仍由归约算子
//! 的语义表推出，层不手写公式。

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
    OutputShape::Scalar
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
    OutputShape::Scalar
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{Bounds, Contract};
    use diagnostic::Code;

    #[test]
    fn a_reduction_emits_a_scalar_shape() {
        let sum = ReduceSum::new("loss.sum", 128).expect("合法");
        assert_eq!(sum.output().dims(), [], "归约输出是标量");
        assert_eq!(sum.spca().ops(), [ReduceSum::OP]);
        let floor = sum.bounds().expect("合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_context(ReduceSum::OP, 128, 1).expect("同一份语义")
        );
        // 读侧覆盖整个输入，写侧只有一个标量。
        assert!(floor.read() > floor.write());
    }

    #[test]
    fn a_max_reduction_binds_its_operator() {
        let max = ReduceMax::new("pool.max", 64).expect("合法");
        assert_eq!(max.output().dims(), []);
        assert_eq!(max.spca().ops(), [ReduceMax::OP]);
        assert!(max.bounds().expect("合法").is_some());
    }

    #[test]
    fn a_zero_length_reduction_is_rejected() {
        assert_eq!(ReduceSum::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(ReduceMax::new("x", 0).unwrap_err().code(), Code::Invalid);
    }
}
