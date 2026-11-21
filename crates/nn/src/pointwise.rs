//! 单轴层：逐元素、归一化、激活、位置编码与逐元素二元运算。

use crate::ParamRole;
use crate::axis::{AxisCore, OutputShape};
use crate::layer_delegate;

/// RMSNorm 归一化：`y = γ · x / sqrt(mean(x²) + ε)`，沿最后一维 `len` 归一化。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNorm {
    inner: AxisCore,
}
layer_delegate!(
    RmsNorm,
    "rmsnorm_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Norm
);

/// Softmax：沿最后一维 `len` 做数值稳定的归一化（减最大值后取指数再除和）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Softmax {
    inner: AxisCore,
}
layer_delegate!(
    Softmax,
    "softmax_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// GELU 激活（精确 erf 版）：逐元素 `0.5·x·(1 + erf(x/√2))`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gelu {
    inner: AxisCore,
}
layer_delegate!(
    Gelu,
    "gelu_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// SiLU / Swish 激活：逐元素 `x · sigmoid(x)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Silu {
    inner: AxisCore,
}
layer_delegate!(
    Silu,
    "silu_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// LeakyReLU 激活：`y = x`（非负）或 `0.01·x`（负区）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeakyRelu {
    inner: AxisCore,
}
layer_delegate!(
    LeakyRelu,
    "leaky_relu_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// HardSigmoid 激活：`y = clamp(x/6 + 0.5, 0, 1)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardSigmoid {
    inner: AxisCore,
}
layer_delegate!(
    HardSigmoid,
    "hard_sigmoid_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// RoPE 旋转位置编码：成对分量按位置做二维旋转；`len` 为头宽，奇数末尾原样透传。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rope {
    inner: AxisCore,
}
layer_delegate!(
    Rope,
    "rope_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 标量缩放：逐元素 `α · x`，系数由算子样本固定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scale {
    inner: AxisCore,
}
layer_delegate!(
    Scale,
    "scale_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 逐元素加法：`y = a + b`，两路输入等长 `len`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add {
    inner: AxisCore,
}
layer_delegate!(
    Add,
    "add_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 逐元素乘法：`y = a · b`，两路输入等长 `len`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul {
    inner: AxisCore,
}
layer_delegate!(
    Mul,
    "mul_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 残差相加：`y = x + 残差`，两路输入等长 `len`；与 [`Add`](crate::Add) 同一段加法，独立成类型对应网络残差层。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualAdd {
    inner: AxisCore,
}
layer_delegate!(
    ResidualAdd,
    "residual_add_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);
mod reduce {
    //! 归约层：把一维输入折叠成单个标量输出。

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
}

pub use reduce::{ReduceMax, ReduceSum};

#[cfg(test)]
#[path = "../tests/support/pointwise_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/pointwise_tests.rs"]
mod pointwise_tests;
