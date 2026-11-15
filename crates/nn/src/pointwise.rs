//! 单轴层：逐元素、归一化、激活、位置编码与逐元素二元运算。
//!
//! 这些层共享同一副形状口径：一条轴 `n`（元素数），输入与输出等长；下界由对应算子的单轴
//! 语义表推出（[`Bounds::of_context`]），层里不写流量常数。各具体层只固定自己的算子名，
//! 规格校验、访问器委托与下界路径只有一份实现（[`PointwiseCore`]）；类型本身在源码里显式
//! 声明，便于架构门禁按源码行核对所有权。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec};

/// 单轴层共享内核：一份已校验规格 + 元素数。
///
/// 具体层包裹它，只补自己的算子名与文档；名字、输出形状、算子列表与下界都从这里取，避免
/// 每一层再抄一遍委托。
#[derive(Debug, Clone, PartialEq, Eq)]
struct PointwiseCore {
    spec: ContractSpec,
    len: usize,
}

impl PointwiseCore {
    /// 构造：元素数必须非零；算子名由调用的具体层给出，构造期经 [`ContractSpec::new`] 校验。
    fn new(name: impl Into<String>, op: &'static str, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("单轴层元素数必须非零（len = {len}）"),
            )
            .with("nn::PointwiseCore::new"));
        }
        let spec = ContractSpec::new(name, Shape::new([len]), vec![op.to_string()])?;
        Ok(Self { spec, len })
    }

    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    /// 元素数（轴 `n` 长度）。
    fn len(&self) -> usize {
        self.len
    }

    /// 本层下界：单轴算子的第二形参被骨架忽略，补 `1` 只满足 `shape_ctx` 签名，不参与代价。
    fn bounds(&self) -> BoundsResult {
        let op = self
            .spec
            .ops()
            .first()
            .map(String::as_str)
            .unwrap_or_default();
        Bounds::of_context(op, self.len, 1).map(Some)
    }
}

/// 为一个已显式声明的单轴层绑定算子名、构造器、访问器与 [`Contract`] 委托。
///
/// 类型声明留在源码里（架构门禁按 `pub struct` 行核对）；本宏只展开构造与委托，算子语义
/// （表达式、位宽、样本、参考）仍在 `op` 一侧，这里不抄第二份。
macro_rules! pointwise_delegate {
    ($ty:ident, $op:literal) => {
        impl $ty {
            /// 本层绑定的算子名。
            pub const OP: &str = $op;

            /// 由层名与元素数构造；元素数为零、或算子未在 `op` 注册时返回说明。
            pub fn new(name: impl Into<String>, len: usize) -> Result<Self> {
                Ok(Self {
                    inner: PointwiseCore::new(name, Self::OP, len)?,
                })
            }

            /// 元素数（轴 `n` 长度）。
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            /// 是否为空：构造期已拒绝零长度，因此恒为 `false`。
            pub fn is_empty(&self) -> bool {
                self.inner.len() == 0
            }
        }

        impl Contract for $ty {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn output(&self) -> &Shape {
                self.inner.output()
            }

            fn spca(&self) -> &ContractSpec {
                self.inner.spca()
            }

            fn bounds(&self) -> BoundsResult {
                self.inner.bounds()
            }
        }
    };
}

/// RMSNorm 归一化：`y = γ · x / sqrt(mean(x²) + ε)`，沿最后一维 `len` 归一化。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNorm {
    inner: PointwiseCore,
}
pointwise_delegate!(RmsNorm, "rmsnorm_f32");

/// Softmax：沿最后一维 `len` 做数值稳定的归一化（减最大值后取指数再除和）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Softmax {
    inner: PointwiseCore,
}
pointwise_delegate!(Softmax, "softmax_f32");

/// GELU 激活（精确 erf 版）：逐元素 `0.5·x·(1 + erf(x/√2))`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gelu {
    inner: PointwiseCore,
}
pointwise_delegate!(Gelu, "gelu_f32");

/// SiLU / Swish 激活：逐元素 `x · sigmoid(x)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Silu {
    inner: PointwiseCore,
}
pointwise_delegate!(Silu, "silu_f32");

/// RoPE 旋转位置编码：成对分量按位置做二维旋转；`len` 为头宽，奇数末尾原样透传。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rope {
    inner: PointwiseCore,
}
pointwise_delegate!(Rope, "rope_f32");

/// 标量缩放：逐元素 `α · x`，系数由算子样本固定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scale {
    inner: PointwiseCore,
}
pointwise_delegate!(Scale, "scale_f32");

/// 逐元素加法：`y = a + b`，两路输入等长 `len`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add {
    inner: PointwiseCore,
}
pointwise_delegate!(Add, "add_f32");

/// 逐元素乘法：`y = a · b`，两路输入等长 `len`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul {
    inner: PointwiseCore,
}
pointwise_delegate!(Mul, "mul_f32");

/// 残差相加：`y = x + 残差`，两路输入等长 `len`；与 [`Add`](crate::Add) 同一段加法，独立成类型对应网络残差层。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualAdd {
    inner: PointwiseCore,
}
pointwise_delegate!(ResidualAdd, "residual_add_f32");

#[cfg(test)]
mod tests {
    use super::*;

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
