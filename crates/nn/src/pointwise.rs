//! 单轴层：逐元素、归一化、激活、位置编码与逐元素二元运算。
//!
//! 这些层共享同一副形状口径：一条轴 `n`（元素数），输入与输出等长；下界由对应算子的单轴
//! 语义表推出（[`Bounds::of_context`]），层里不写流量常数。各具体层只固定自己的算子名，
//! 规格校验、访问器委托与下界路径由 [`crate::axis::AxisCore`] 与 [`layer_delegate`] 收口一份；
//! 类型本身在源码里显式声明，便于架构门禁按源码行核对所有权。

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
}

pub use reduce::{ReduceMax, ReduceSum};

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{Bounds, Contract, ParamRole};
    use diagnostic::Code;

    fn assert_layer<L: Contract>(layer: &L, name: &str, op: &str, len: usize) {
        assert_eq!(layer.name(), name);
        assert_eq!(layer.output().dims(), [len]);
        assert_eq!(layer.spca().ops(), [op]);
        let floor = layer.bounds().expect("单轴形状合法").expect("有算子");
        // 单轴算子不写第二份公式：下界必须与直接问语义表逐项相等。
        let same = Bounds::of_context(op, len, 1).expect("同一份语义");
        assert_eq!(floor, same, "{op} 的下界来自算子语义表");
        assert!(floor.read() > 0 && floor.write() > 0, "{op} 有读有写");
    }

    #[test]
    fn each_pointwise_layer_binds_its_declared_operator() {
        assert_layer(
            &RmsNorm::new("blk.0.norm", 128).expect("合法"),
            "blk.0.norm",
            RmsNorm::OP,
            128,
        );
        assert_layer(
            &Softmax::new("attn.softmax", 64).expect("合法"),
            "attn.softmax",
            Softmax::OP,
            64,
        );
        assert_layer(
            &Gelu::new("mlp.gelu", 128).expect("合法"),
            "mlp.gelu",
            Gelu::OP,
            128,
        );
        assert_layer(
            &Silu::new("mlp.silu", 128).expect("合法"),
            "mlp.silu",
            Silu::OP,
            128,
        );
        assert_layer(
            &Rope::new("attn.rope", 128).expect("合法"),
            "attn.rope",
            Rope::OP,
            128,
        );
        assert_layer(
            &Scale::new("proj.scale", 128).expect("合法"),
            "proj.scale",
            Scale::OP,
            128,
        );
        assert_layer(
            &Add::new("res.add", 128).expect("合法"),
            "res.add",
            Add::OP,
            128,
        );
        assert_layer(
            &Mul::new("gate.mul", 128).expect("合法"),
            "gate.mul",
            Mul::OP,
            128,
        );
        assert_layer(
            &ResidualAdd::new("residual.add", 128).expect("合法"),
            "residual.add",
            ResidualAdd::OP,
            128,
        );
    }

    #[test]
    fn normalization_declares_the_norm_role_and_other_axis_layers_the_trainable_role() {
        let norm = RmsNorm::new("blk.0.norm", 128).expect("合法");
        assert_eq!(norm.spca().role(), ParamRole::Norm);
        assert!(norm.spca().role().wd_exempt());

        let activation = Gelu::new("mlp.gelu", 128).expect("合法");
        assert_eq!(activation.spca().role(), ParamRole::Trainable);
    }

    #[test]
    fn a_zero_length_pointwise_layer_is_rejected() {
        assert_eq!(Gelu::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Silu::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(RmsNorm::new("x", 0).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn pointwise_layers_work_as_trait_objects() {
        let layer: &dyn Contract = &Silu::new("mlp.silu", 64).expect("合法");
        assert_eq!(layer.output().dims(), [64]);
        assert!(layer.bounds().expect("合法").is_some());
    }
}
