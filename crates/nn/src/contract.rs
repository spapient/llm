//! 层契约与单算子层：[`ContractSpec`]、[`Contract`]、[`Linear`]、[`MatMul`]、[`Embedding`]
//! 及单轴层声明表。配套的非零维度校验与下界结果别名也收在这里。
//!
//! 本模块只承载「层向装配侧声明什么」；下界算术（[`crate::Bounds`]）、单轴骨架（`declare`）
//! 与权重编码（[`crate::WeightFormat`]）分居另外两块，经 crate 根路径互相引用，不复制第二份常数。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::axis;
use crate::{Bounds, LayerGeometry, ParamRole};

/// 层下界的结果：[`Contract::bounds`] 的返回类型。
pub type BoundsResult = Result<Option<crate::Bounds>>;

/// 已声明的算子名清单，仅用于报错文案（热路径不调用）。
fn declared_ops() -> String {
    let names: Vec<&str> = prof::catalogue::all()
        .iter()
        .map(prof::catalogue::Op::name)
        .collect();
    format!("已声明：{}", names.join("、"))
}

/// 逐个校验「必须非零」的维度，失败时点名装配位（`site`）。
///
/// `prefix` 自带分隔符（如 `"注意力层 "` / `"Embedding 的 "`），因此各层原有文案逐字保留。
pub(crate) fn nonzero_all(site: &str, prefix: &str, axes: &[(&str, usize)]) -> Result<()> {
    for (label, value) in axes {
        if *value == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{prefix}{label} 必须非零（{label} = {value}）"),
            )
            .with(site));
        }
    }
    Ok(())
}

/// 校验单个「必须非零」的维度，失败时点名层与字段：经典几何层族共用。
///
/// 与 [`nonzero_all`] 的分工是调用形态不同——这里逐字段校验并**返回**该值供构造器继续用，
/// `nonzero_all` 一次吃完一张表、只报第一个空轴。
pub(crate) fn nonzero(value: usize, field: &str, layer: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::new(Code::Invalid, format!("{layer}: `{field}` 必须非零")).with("nn"));
    }
    Ok(value)
}

/// 层定义产出的声明：名字、输出形状、算子列表、它导出的声明，以及**自己的下界**。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContractSpec {
    name: String,
    output: Shape,
    ops: Vec<String>,
    role: ParamRole,
}

impl ContractSpec {
    /// 校验并构造：`ops` 必须既存在于 [`prof::catalogue::all`]，又有 `registry::model` 的语义表。
    ///
    /// `ops` 收 `impl Into<String>` 的任意可迭代物：字面量数组 `["gemv_i4"]` 与已拼好的
    /// `Vec<String>` 都能直接传，调用侧不必为一条静态名字写 `.to_string()`。
    pub fn new(
        name: impl Into<String>,
        output: Shape,
        ops: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self> {
        Self::new_with_role(name, output, ops, ParamRole::Trainable)
    }

    /// 校验并构造，并显式声明参数角色；投影之外的层（归一化系数、偏置、嵌入表）必须经此入口
    /// 填角色——优化器分组从角色派生，猜字段名或层名前缀会把衰减组与豁免组放错。
    pub fn new_with_role(
        name: impl Into<String>,
        output: Shape,
        ops: impl IntoIterator<Item = impl Into<String>>,
        role: ParamRole,
    ) -> Result<Self> {
        let name = name.into();
        Self::check_geometry(&name, &output)?;
        let ops = Self::check_ops(&name, ops)?;
        Ok(Self {
            name,
            output,
            ops,
            role,
        })
    }

    /// 名字非空、输出无零宽轴、元素数不溢出。
    fn check_geometry(name: &str, output: &Shape) -> Result<()> {
        if name.trim().is_empty() {
            return Err(
                Error::new(Code::Invalid, "层名不能为空或只含空白").with("nn::ContractSpec")
            );
        }
        if let Some(axis) = output.dims().iter().position(|dim| *dim == 0) {
            return Err(Error::new(
                Code::Invalid,
                format!("层 `{name}` 的输出第 {axis} 轴为零宽"),
            )
            .with("nn::ContractSpec"));
        }
        output
            .dims()
            .iter()
            .try_fold(1usize, |size, dim| size.checked_mul(*dim))
            .ok_or_else(|| {
                Error::new(Code::Invalid, format!("层 `{name}` 的输出元素数溢出"))
                    .with("nn::ContractSpec")
            })?;
        Ok(())
    }

    /// 算子去重、已在声明目录、且有注册表语义表；返回按序去重后的名字表。
    fn check_ops(
        name: &str,
        ops: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Vec<String>> {
        // 热路径不建整表：每层只按名字查一次，全表清单只在报错时才拼。
        let mut checked: Vec<String> = Vec::new();
        for wanted in ops {
            let wanted = wanted.into();
            if checked.contains(&wanted) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("层 `{name}` 重复声明算子 `{wanted}`"),
                )
                .with("nn::ContractSpec"));
            }
            if prof::catalogue::named(&wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有声明（{}）", declared_ops()),
                )
                .with("nn::ContractSpec::new"));
            }
            if crate::registry::model(&wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有语义表，层无法推出下界也无法对拍"),
                )
                .with("nn::ContractSpec::new"));
            }
            checked.push(wanted);
        }
        Ok(checked)
    }

    crate::accessors! {
        /// 层名（与权重条目的前缀一致）。
        name -> &str = &name,
        /// 输出形状。
        output -> &Shape = &output,
        /// 要用的算子名。空切片表示这一层的编码还没有算子，因此无法计价。
        ops -> &[String] = &ops,
    }

    crate::accessors! {
        /// 参数角色：优化器分组与权重衰减豁免的派生依据。
        role -> ParamRole = role,
    }
}

/// 层契约：名字、输出形状、算子列表、它导出的声明，以及**自己的下界**。
pub trait Contract {
    /// 层名。
    fn name(&self) -> &str;

    /// 输出形状。
    fn output(&self) -> &Shape;

    /// 导出给 `cutile` 的声明。
    fn spca(&self) -> &ContractSpec;

    /// 本层的性能下界。
    fn bounds(&self) -> BoundsResult {
        Ok(None)
    }

    /// 若这一层是按形状构造的线性投影，返回它的类型化视图。
    fn linear(&self) -> Option<&Linear> {
        None
    }

    /// 若这一层是查表层，返回它的类型化视图。
    fn embedding(&self) -> Option<&crate::Embedding> {
        None
    }

    /// 若这一层是注意力组合块，返回它的类型化视图。
    fn attention_block(&self) -> Option<&crate::attention::AttentionBlock> {
        None
    }

    /// 若这一层是条件记忆层，返回它的类型化视图。
    fn memory(&self) -> Option<&crate::engram::MemoryBlock> {
        None
    }

    /// 若这一层是循环层，返回它的类型化视图。
    ///
    /// 与 [`Contract::linear`] 同一个用途：从检查点装配一侧只拿到 `dyn Contract`，要读回
    /// 「几层、几个方向、哪种单元」就得靠这条下行通道，而不是再猜一次名字。
    fn recurrent(&self) -> Option<&crate::recurrent::Recurrent> {
        None
    }
}

/// 线性层：`y = x · W`。用哪个算子由权重编码 [`crate::WeightFormat`] 决定，而不是一个全局常量。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Linear {
    spec: ContractSpec,
    in_features: usize,
    out_features: usize,
    format: crate::WeightFormat,
}

impl Linear {
    /// 默认编码（`WeightFormat::INT4`）对应的算子名。
    pub const OP: &str = "gemv_i4";

    /// 稠密 f32 权重对应的算子名。
    pub const DENSE: &str = "gemv_f32";

    /// 对称 int8 权重对应的算子名。
    pub const I8: &str = "gemv_i8";

    /// 对称 int2 权重对应的算子名。
    pub const I2: &str = "gemv_i2";

    /// 由输入/输出宽度构造，按 int4 权重（`WeightFormat::INT4`）计价；两个宽度都必须非零。
    pub fn new(name: impl Into<String>, in_features: usize, out_features: usize) -> Result<Self> {
        Self::with_format(name, crate::WeightFormat::INT4, in_features, out_features)
    }

    /// 按给定的权重编码构造。
    ///
    /// 编码没有对应算子时（例如 f16 / bf16 / i8），层的 `ops` 为空：它仍然是一个合法的投影
    /// 声明，只是推不出下界。把它做成 `Err` 反而会逼调用方"随便挑一个格式"来让构造成功。
    ///
    /// # Errors
    ///
    /// 任一宽度为零（[`Code::Invalid`]）；算子名由编码给出却没有通过 [`ContractSpec::new`] 的校验
    /// （只可能发生在编码与 `registry` 注册表不一致时）。
    pub fn with_format(
        name: impl Into<String>,
        format: crate::WeightFormat,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("线性层宽度必须非零（in = {in_features}，out = {out_features}）"),
            )
            .with("nn::Linear::with_format"));
        }
        let spec = ContractSpec::new_with_role(
            name,
            Shape::new([out_features]),
            Self::operator_for(format),
            ParamRole::Trainable,
        )?;
        Ok(Self {
            spec,
            in_features,
            out_features,
            format,
        })
    }

    crate::accessors! {
        /// 输入宽度。
        in_features -> usize = in_features,
        /// 输出宽度。
        out_features -> usize = out_features,
        /// 权重编码。它决定了本层用哪个算子，也决定了"这一层的下界是否可算"。
        format -> crate::WeightFormat = format,
    }

    /// 本层的性能下界，见 [`Contract::bounds`]。
    pub fn bounds(&self) -> BoundsResult {
        <Self as Contract>::bounds(self)
    }
}

impl Contract for Linear {
    crate::spec_accessors!();

    fn linear(&self) -> Option<&Linear> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        // 编码没有算子：不是"下界为零"，而是"这个格式还没有可计价的内核"。
        // 如实报出而不是拿另一个格式的算子凑一个数——那正是改造前 F32 权重按 int4
        // 报出 10 MiB 的原因。
        let Some(operator) = self.format.operator() else {
            // 理由由编码自己给：`Int4 { group }` 要说出"分组不同"，`Packed4` 要说出"缺标度条目"，
            // 否则读完只知道"没有算子"，不知道该改文件还是该换内核。
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "层 `{}` 的权重{}，无法推出下界",
                    self.spec.name(),
                    self.format.unpriced_reason()
                ),
            )
            .with("nn::Linear::bounds"));
        };
        crate::Bounds::of_geometry(
            operator,
            LayerGeometry::projection(self.in_features, self.out_features),
        )
        .map(Some)
    }
}

// ---- 单算子层：单轴声明、嵌入查表与矩阵乘 ----

// 单轴层声明表：类型、算子与可选参数角色只写一次；角色省略即 `Trainable`。
// 构造、校验、访问器、Contract 与下界仍由 `AxisCore` 单态化展开，不增加运行时状态。
axis! {
    /// RMSNorm 归一化：`y = γ · x / sqrt(mean(x²) + ε)`，沿最后一维 `len` 归一化。
    RmsNorm => "rmsnorm_f32", Norm;
    /// Softmax：沿最后一维 `len` 做数值稳定的归一化（减最大值后取指数再除和）。
    Softmax => "softmax_f32";
    /// GELU 激活（精确 erf 版）：逐元素 `0.5·x·(1 + erf(x/√2))`。
    Gelu => "gelu_f32";
    /// SiLU / Swish 激活：逐元素 `x · sigmoid(x)`。
    Silu => "silu_f32";
    /// 逐元素指数：`y = exp(a)`。
    Exp => "exp_f32";
    /// 逐元素自然对数：`y = ln(a)`。定义域由算子语义表负责（层不替内核判值域）。
    Log => "log_f32";
    /// LeakyReLU 激活：`y = x`（非负）或 `0.01·x`（负区）。
    LeakyRelu => "leaky_relu_f32";
    /// HardSigmoid 激活：`y = clamp(x/6 + 0.5, 0, 1)`。
    HardSigmoid => "hard_sigmoid_f32";
    /// Sigmoid 激活：`y = 1 / (1 + exp(-x))`，输出落在 `(0, 1)`。
    Sigmoid => "sigmoid_f32";
    /// Tanh 激活：`y = tanh(x)`，输出落在 `(-1, 1)`。
    Tanh => "tanh_f32";
    /// ReLU 激活：`y = max(x, 0)`。
    Relu => "relu_f32";
    /// RoPE 旋转位置编码：成对分量按位置做二维旋转；`len` 为头宽，奇数末尾原样透传。
    Rope => "rope_f32";
    /// 标量缩放：逐元素 `α · x`，系数由算子样本固定。
    Scale => "scale";
    /// 逐元素加法：`y = a + b`，两路输入等长 `len`。
    Add => "add_f32";
    /// 逐元素乘法：`y = a · b`，两路输入等长 `len`。
    Mul => "mul_f32";
    /// 逐元素减法：`y = a - b`，两路输入等长 `len`。
    Sub => "sub_f32";
    /// 逐元素除法：`y = a / b`，两路输入等长 `len`；除零语义归算子，层不越权改写。
    Div => "div_f32";
    /// 逐元素取大：`y = max(a, b)`。
    Max => "max_f32";
    /// 逐元素取小：`y = min(a, b)`。
    Min => "min_f32";
    /// 前缀扫描：`s_t = a·s_{t-1} + b·x_t`；时间轴串行依赖由独立 `scan` 骨架声明。
    Scan => "scan_f32";
}

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
    /// `vocab` / `dim` / `tokens` 任一为零（[`Code::Invalid`]）；算子未在 `registry` 注册或没有
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
    crate::spec_accessors!();

    fn embedding(&self) -> Option<&Embedding> {
        Some(self)
    }

    fn bounds(&self) -> BoundsResult {
        Bounds::of_geometry(Self::OP, LayerGeometry::shape(self.tokens, self.dim)).map(Some)
    }
}

/// 收缩轴档位：`K` 是**算子表声明的固定档**，不是层能编造的尺寸。
///
/// 矩阵乘在算子表里按 `K` 分成两档：`gemm_*` 用 8，`matmul_*` 用 4096（prefill 主链路代表值）。
/// 两档的下界相差 512 倍，所以档位必须由层**声明**——拿小档的算子给大档的层计价，
/// 报出来的就是一条偏小 512 倍的下界。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// 「精度 × 档位 → 算子名」的完整二维配置；空单元就是没有注册该组合。
/// 行、列分别由 `Precision` / `Gauge` 的显式 `repr(u8)` 判别值索引，查询为 O(1)。
const OPERATORS: [[Option<&str>; 2]; 4] = [
    [Some(MatMul::OP), Some("matmul_f32")],
    [Some("gemm_i2"), None],
    [Some("gemm_i4"), None],
    [Some("gemm_i8"), Some("matmul_i8")],
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

    /// 返回该精度与档位的算子；未注册组合返回 `None`。
    pub const fn operator(self, gauge: Gauge) -> Option<&'static str> {
        OPERATORS[self as usize][gauge as usize]
    }
}

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
    /// `rows` / `cols` 任一为零（[`Code::Invalid`]）；算子未在 `registry` 注册或没有语义表
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
    /// `rows` / `cols` 任一为零（[`Code::Invalid`]）；选中的算子未在 `registry` 注册或没有语义表。
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
    crate::spec_accessors!();

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
        Bounds::of_geometry(operator, LayerGeometry::shape(self.rows, self.cols)).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/single_tests.rs"]
mod single_tests;
