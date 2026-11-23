//! 单算子层：一个层对应算子语义表里的一个算子，形状完全由构造参数给出、没有窗口/分组几何。
//!
//! 两族在这里同居，差别只在权重口径：逐元素/归一化/激活/位置编码/逐元素二元运算是
//! **无参数或对角参数**，查表/矩阵乘是**成形的权重矩阵**。共同的契约是：层不写第二份流量
//! 常数，下界一律由算子语义表推出（`layer_delegate!` 与 `spec_accessors!` 共同保证）。

use diagnostic::{Code, Error, Result};

use crate::axis;
use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole, Shape, spec_accessors};

// 单轴层清单：一层三行——`axis!(`、文档、`类型, 算子名[, 参数角色]);`。
//
// 这十九层原本是四行块（`);` 单独占一行），块里真正有信息量的只有「类型 / 算子 / 角色」三列，
// 其余逐字相同；把结尾的 `);` 收上来到参数行之后，一层就是三行、差异一眼可见。文档留在括号里
// （宏的 `$(#[$meta])*` 会把它挂到展开出的结构体上），而不是写在调用点外——写在外面 rustdoc
// 不认。角色省略即 `Trainable`（由 `axis!` 的默认臂决定）。
axis!(
    /// RMSNorm 归一化：`y = γ · x / sqrt(mean(x²) + ε)`，沿最后一维 `len` 归一化。
    RmsNorm, "rmsnorm_f32", Norm);
axis!(
    /// Softmax：沿最后一维 `len` 做数值稳定的归一化（减最大值后取指数再除和）。
    Softmax, "softmax_f32");
axis!(
    /// GELU 激活（精确 erf 版）：逐元素 `0.5·x·(1 + erf(x/√2))`。
    Gelu, "gelu_f32");
axis!(
    /// SiLU / Swish 激活：逐元素 `x · sigmoid(x)`。
    Silu, "silu_f32");
axis!(
    /// 逐元素指数：`y = exp(a)`。
    Exp, "exp_f32");
axis!(
    /// 逐元素自然对数：`y = ln(a)`。定义域由算子语义表负责（层不替内核判值域）。
    Log, "log_f32");
axis!(
    /// LeakyReLU 激活：`y = x`（非负）或 `0.01·x`（负区）。
    LeakyRelu, "leaky_relu_f32");
axis!(
    /// HardSigmoid 激活：`y = clamp(x/6 + 0.5, 0, 1)`。
    HardSigmoid, "hard_sigmoid_f32");
axis!(
    /// Sigmoid 激活：`y = 1 / (1 + exp(-x))`，输出落在 `(0, 1)`。
    Sigmoid, "sigmoid_f32");
axis!(
    /// Tanh 激活：`y = tanh(x)`，输出落在 `(-1, 1)`。
    Tanh, "tanh_f32");
axis!(
    /// ReLU 激活：`y = max(x, 0)`。
    Relu, "relu_f32");
axis!(
    /// RoPE 旋转位置编码：成对分量按位置做二维旋转；`len` 为头宽，奇数末尾原样透传。
    Rope, "rope_f32");
axis!(
    /// 标量缩放：逐元素 `α · x`，系数由算子样本固定。
    Scale, "scale_f32");
axis!(
    /// 逐元素加法：`y = a + b`，两路输入等长 `len`。
    Add, "add_f32");
axis!(
    /// 逐元素乘法：`y = a · b`，两路输入等长 `len`。
    Mul, "mul_f32");
axis!(
    /// 逐元素减法：`y = a - b`，两路输入等长 `len`。
    Sub, "sub_f32");
axis!(
    /// 逐元素除法：`y = a / b`，两路输入等长 `len`；除零语义归算子，层不越权改写。
    Div, "div_f32");
axis!(
    /// 逐元素取大：`y = max(a, b)`。
    Max, "max_f32");
axis!(
    /// 逐元素取小：`y = min(a, b)`。
    Min, "min_f32");

/// 嵌入查表层。
#[crate::spec(name = $name, out = [$tokens, $dim], ops = [$op], role = $role)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Gauge {
    /// `K = 8`：小收缩轴（逐 token 的投影级矩阵乘）。
    Short = 0,
    /// `K = 4096`：大收缩轴。
    Long = 1,
}

/// 矩阵乘的元素精度：**对称量化、无分组标度**。
///
/// 它与投影侧的 [`WeightFormat`](crate::WeightFormat) 不是同一张表，差别的来源在算子表里：
/// `gemv_*` 声明了 `needs.quant.group`（每 `group` 个元素配一个标度），`gemm_*` / `matmul_*`
/// 没有这条声明。分组是投影侧的口径，矩阵乘侧只按精度选内核，所以这里是另一条轴，而不是
/// 把 `WeightFormat` 的 `group` 字段勉强塞进来。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Precision {
    /// 稠密 32 位浮点。
    F32 = 0,
    /// 对称 2 位整数。
    Int2 = 1,
    /// 对称 4 位整数。
    Int4 = 2,
    /// 对称 8 位整数。
    Int8 = 3,
}

/// 「精度 × 档位 → 算子名」的全部已注册组合，唯一事实源。
///
/// 表的行数就是 `crates/op` 里 gemm 骨架注册的算子数（6 条）；台账 v3.0.74 记有它们的来历。
const RUNGS: [(Precision, Gauge, &str); 6] = [
    (Precision::F32, Gauge::Short, MatMul::OP),
    (Precision::F32, Gauge::Long, "matmul_f32"),
    (Precision::Int8, Gauge::Short, "gemm_i8"),
    (Precision::Int8, Gauge::Long, "matmul_i8"),
    (Precision::Int4, Gauge::Short, "gemm_i4"),
    (Precision::Int2, Gauge::Short, "gemm_i2"),
];

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
    ///
    /// 组合写成一张表而不是 `match` 分支：这里要表达的是「哪些格子里有内核」这个二维事实，
    /// 表里有就是有、没写就是没有，读者一眼能看出空着的那两格（int2 / int4 × `K = 4096`）。
    // `precision as u8 == self as u8` 与 `rung as u8 == gauge as u8` 比较的是**判别值**，
    // 因此两个枚举都声明了 `#[repr(u8)]` + 显式判别值：`RUNGS` 里的顺序不再隐含成契约，
    // 调整枚举成员顺序也不会让查表悄悄换行。
    pub const fn operator(self, gauge: Gauge) -> Option<&'static str> {
        let mut idx = 0;
        while idx < RUNGS.len() {
            let (precision, rung, operator) = RUNGS[idx];
            if precision as u8 == self as u8 && rung as u8 == gauge as u8 {
                return Some(operator);
            }
            idx += 1;
        }
        None
    }
}

axis!(
    /// 前缀扫描层：`s_t = a·s_{t-1} + b·x_t`，沿单一时间轴串行递推。
    ///
    /// 与逐元素层的差别不在形状而在**依赖**：每个输出都依赖前一个输出，因此算子表给了独立的
    /// `scan` 骨架与分块声明。层侧仍只声明长度，读写字节与算术量全部由算子表推出。
    Scan,
    "scan_f32"
);

/// 矩阵乘层：结果形状 `[rows, cols]`；收缩轴档位与元素精度共同决定用哪个算子。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
        let spec = ContractSpec::new(name, Shape::new([rows, cols]), precision.operator(gauge))?;
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
#[path = "../tests/single_tests.rs"]
mod single_tests;
