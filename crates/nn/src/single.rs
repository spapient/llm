//! 单算子层：一个层对应算子语义表里的一个算子，形状完全由构造参数给出、没有窗口/分组几何。
//!
//! 两族在这里同居，差别只在权重口径：逐元素/归一化/激活/位置编码/逐元素二元运算是
//! **无参数或对角参数**，查表/矩阵乘是**成形的权重矩阵**。共同的契约是：层不写第二份流量
//! 常数，下界一律由算子语义表推出（`layer_delegate!` 与 `spec_accessors!` 共同保证）。

use diagnostic::{Code, Error, Result};

use crate::declare::{AxisCore, OutputShape};
use crate::layer_delegate;
use crate::{Bounds, BoundsResult, Contract, ParamRole, spec_accessors};

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
    use crate::declare::{AxisCore, OutputShape};
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

/// 嵌入查表层。
#[crate::spec(name = $name, out = [$tokens, $dim], ops = [$op], role = $role)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embedding {
    vocab: usize,
    dim: usize,
    tokens: usize,
}

impl Embedding {
    /// 绑定的算子名。
    pub const OP: &str = "embed_f32";

    /// 由层名、词表大小、行宽与 token 数构造。
    ///
    /// # Errors
    ///
    /// `vocab` / `dim` / `tokens` 任一为零（[`Code::Invalid`]）；算子未在 `op` 注册或没有
    /// 语义表（[`crate::ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, vocab: usize, dim: usize, tokens: usize) -> Result<Self> {
        crate::nonzero_all(
            "nn::Embedding::new",
            "Embedding 的 ",
            &[("vocab", vocab), ("dim", dim), ("tokens", tokens)],
        )?;
        Ok(Self {
            spec: __layer_spec_Embedding!(name, tokens, dim, Self::OP, ParamRole::Embedding)?,
            vocab,
            dim,
            tokens,
        })
    }

    crate::accessors! {
        /// 词表大小（权重表行数）。
        vocab -> usize = vocab,
        /// 行宽（每个 token 向量的维度）。
        dim -> usize = dim,
        /// 一次查表的 token 数。
        tokens -> usize = tokens,
    }
}

impl Contract for Embedding {
    spec_accessors!();

    fn embedding(&self) -> Option<&Embedding> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        Bounds::of_ctx(Self::OP, self.tokens, 1).map(Some)
    }
}

/// 矩阵乘层：结果形状 `[rows, cols]`，收缩轴由算子固定。
#[crate::spec(name = $name, out = [$rows, $cols], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMul {
    rows: usize,
    cols: usize,
}

impl MatMul {
    /// 绑定的稠密矩阵乘算子名。
    pub const OP: &str = "gemm_f32";

    /// 由输出行数与列数构造；两者都必须非零。
    ///
    /// # Errors
    ///
    /// `rows` / `cols` 任一为零（[`Code::Invalid`]）；算子未在 `op` 注册或没有语义表
    /// （[`crate::ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("矩阵乘维度必须非零（rows = {rows}，cols = {cols}）"),
            )
            .with("nn::MatMul::new"));
        }
        let spec = __layer_spec_MatMul!(name, rows, cols, Self::OP)?;
        Ok(Self { spec, rows, cols })
    }

    crate::accessors! {
        /// 输出行数（轴 `m`）。
        rows -> usize = rows,
        /// 输出列数（轴 `n`）。
        cols -> usize = cols,
    }
}

impl Contract for MatMul {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 三轴骨架：收缩轴 `k` 由算子的 `project_shape` 固定，层只给 `(m, n)`，走通用形状入口，
        // 而不是投影入口——投影入口只填输入/输出两条角色轴，填不出独立的收缩轴。
        Bounds::of_ctx(Self::OP, self.rows, self.cols).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/support/pointwise_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/single_tests.rs"]
mod single_tests;
