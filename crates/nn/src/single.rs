//! 单算子层：一个层对应算子语义表里的一个算子，形状完全由构造参数给出、没有窗口/分组几何。
//!
//! 两族在这里同居，差别只在权重口径：逐元素/归一化/激活/位置编码/逐元素二元运算是
//! **无参数或对角参数**，查表/矩阵乘是**成形的权重矩阵**。共同的契约是：层不写第二份流量
//! 常数，下界一律由算子语义表推出（`layer_delegate!` 与 `spec_accessors!` 共同保证）。

use diagnostic::{Code, Error, Result};

use crate::declare::{AxisCore, OutputShape};
use crate::layer_delegate;
use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole, Shape, spec_accessors};

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

/// 逐元素指数：`y = exp(a)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exp {
    inner: AxisCore,
}
layer_delegate!(
    Exp,
    "exp_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 逐元素自然对数：`y = ln(a)`。定义域由算子语义表负责（层不替内核判值域）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Log {
    inner: AxisCore,
}
layer_delegate!(
    Log,
    "log_f32",
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

/// 逐元素减法：`y = a - b`，两路输入等长 `len`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sub {
    inner: AxisCore,
}
layer_delegate!(
    Sub,
    "sub_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 逐元素除法：`y = a / b`，两路输入等长 `len`；除零语义归算子，层不越权改写。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Div {
    inner: AxisCore,
}
layer_delegate!(
    Div,
    "div_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 逐元素取大：`y = max(a, b)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Max {
    inner: AxisCore,
}
layer_delegate!(
    Max,
    "max_f32",
    "单轴层",
    OutputShape::Single,
    ParamRole::Trainable
);

/// 逐元素取小：`y = min(a, b)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Min {
    inner: AxisCore,
}
layer_delegate!(
    Min,
    "min_f32",
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

/// 收缩轴档位：`K` 是**算子表声明的固定档**，不是层能编造的尺寸。
///
/// 矩阵乘在算子表里按 `K` 分成两档：`gemm_*` 用 8，`matmul_*` 用 4096（prefill 主链路代表值）。
/// 两档的下界相差 512 倍，所以档位必须由层**声明**——拿小档的算子给大档的层计价，
/// 报出来的就是一条偏小 512 倍的下界。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gauge {
    /// `K = 8`：小收缩轴（逐 token 的投影级矩阵乘）。
    Short,
    /// `K = 4096`：大收缩轴。
    Long,
}

/// 矩阵乘的元素精度：**对称量化、无分组标度**。
///
/// 它与投影侧的 [`WeightFormat`](crate::WeightFormat) 不是同一张表，差别的来源在算子表里：
/// `gemv_*` 声明了 `needs.quant.group`（每 `group` 个元素配一个标度），`gemm_*` / `matmul_*`
/// 没有这条声明。分组是投影侧的口径，矩阵乘侧只按精度选内核，所以这里是另一条轴，而不是
/// 把 `WeightFormat` 的 `group` 字段勉强塞进来。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 稠密 32 位浮点。
    F32,
    /// 对称 2 位整数。
    Int2,
    /// 对称 4 位整数。
    Int4,
    /// 对称 8 位整数。
    Int8,
}

impl Gauge {
    /// 本档位写进诊断时的说法。
    pub const fn name(self) -> &'static str {
        match self {
            Self::Short => "K = 8",
            Self::Long => "K = 4096",
        }
    }
}

impl Precision {
    /// 本精度写进诊断时的说法。
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int2 => "int2",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
        }
    }

    /// 本精度在给定档位下的算子名；算子表没有注册这个组合时返回 `None`。
    pub const fn operator(self, gauge: Gauge) -> Option<&'static str> {
        match (self, gauge) {
            (Self::F32, Gauge::Short) => Some(MatMul::OP),
            (Self::F32, Gauge::Long) => Some("matmul_f32"),
            (Self::Int8, Gauge::Short) => Some("gemm_i8"),
            (Self::Int8, Gauge::Long) => Some("matmul_i8"),
            // 大收缩轴只注册了 f32 与 int8：int2 / int4 的大 K 内核不存在。层对这种情况的回答是
            // 「没有这个组合」，而不是退回小 K 的兄弟算子凑一个数。
            (Self::Int2, Gauge::Short) => Some("gemm_i2"),
            (Self::Int4, Gauge::Short) => Some("gemm_i4"),
            (Self::Int2 | Self::Int4, Gauge::Long) => None,
        }
    }
}

/// 矩阵乘层：结果形状 `[rows, cols]`；收缩轴档位与元素精度共同决定用哪个算子。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMul {
    spec: ContractSpec,
    rows: usize,
    cols: usize,
    precision: Precision,
    gauge: Gauge,
}

impl MatMul {
    /// 稠密小收缩轴（`gemm_f32`）对应的算子名，也是 [`MatMul::new`] 的默认口径。
    pub const OP: &str = "gemm_f32";

    /// 由输出行数与列数构造，按稠密 f32 × `K = 8` 计价；两者都必须非零。
    ///
    /// # Errors
    ///
    /// `rows` / `cols` 任一为零（[`Code::Invalid`]）；算子未在 `op` 注册或没有语义表
    /// （[`crate::ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> Result<Self> {
        Self::with_precision(name, rows, cols, Precision::F32, Gauge::Short)
    }

    /// 按给定的元素精度与收缩轴档位构造。
    ///
    /// 算子表没有注册该组合时（目前只有 int2 / int4 × `K = 4096`）层的 `ops` 为空：它仍是一个
    /// 合法的矩阵乘声明，只是推不出下界，`bounds` 会给出「没有这个组合」的理由。做成 `Err`
    /// 会逼调用方随便挑一个精度来让构造成功，那正是下界开始说谎的起点。
    ///
    /// # Errors
    ///
    /// `rows` / `cols` 任一为零（[`Code::Invalid`]）；选中的算子未在 `op` 注册或没有语义表。
    pub fn with_precision(
        name: impl Into<String>,
        rows: usize,
        cols: usize,
        precision: Precision,
        gauge: Gauge,
    ) -> Result<Self> {
        crate::nonzero_all(
            "nn::MatMul::with_precision",
            "矩阵乘的 ",
            &[("rows", rows), ("cols", cols)],
        )?;
        let spec = ContractSpec::new(
            name,
            Shape::new([rows, cols]),
            precision.operator(gauge),
        )?;
        Ok(Self {
            spec,
            rows,
            cols,
            precision,
            gauge,
        })
    }

    crate::accessors! {
        /// 输出行数（轴 `m`）。
        rows -> usize = rows,
        /// 输出列数（轴 `n`）。
        cols -> usize = cols,
        /// 元素精度。它决定用哪个内核，也决定「这一层的下界是否可算」。
        precision -> Precision = precision,
        /// 收缩轴档位。
        gauge -> Gauge = gauge,
    }
}

impl Contract for MatMul {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 三轴骨架：收缩轴 `k` 由算子的 `project_shape` 固定，层只给 `(m, n)`，走通用形状入口，
        // 而不是投影入口——投影入口只填输入/输出两条角色轴，填不出独立的收缩轴。
        let Some(operator) = self.precision.operator(self.gauge) else {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "矩阵乘层 `{}` 声明了 {} 精度 × {} 的收缩轴，但算子表没有注册这个组合，无法推出下界",
                    self.spec.name(),
                    self.precision.name(),
                    self.gauge.name()
                ),
            )
            .with("nn::MatMul::bounds"));
        };
        Bounds::of_ctx(operator, self.rows, self.cols).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/support/pointwise_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/single_tests.rs"]
mod single_tests;
