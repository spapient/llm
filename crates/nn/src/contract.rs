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
        let Some(operator) = Self::operator_for(self.format) else {
            // 理由由编码自己给：`Int4 { group }` 要说出"分组不同"，`Packed4` 要说出"缺标度条目"，
            // 否则读完只知道"没有算子"，不知道该改文件还是该换内核。
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "层 `{}` 的权重{}，无法推出下界",
                    self.spec.name(),
                    Self::unpriced_reason_for(self.format)
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
mod single_tests {
    //! `src/single.rs` 的全部用例：逐元素/归一化/激活、归约、查表、矩阵乘（精度与档位）、
    //! 前缀扫描与反量化共用一条契约——层不写第二份流量常数，下界一律由算子语义表推出。
    //!
    //! 逐元素层原先住在 `src/pointwise.rs`，测试因此分出去一份；该文件已并入 `single.rs`，
    //! 两者又挂同一个模块，分开只剩两个同义助手，故合并。

    use super::*;

    use crate::{Bounds, Contract, ParamRole};
    use diagnostic::Code;

    #[test]
    fn an_embedding_declares_its_table_and_output_shape() {
        let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
        assert_eq!(layer.name(), "tok_emb");
        assert_eq!(layer.vocab(), 32_000);
        assert_eq!(layer.dim(), 128);
        assert_eq!(layer.tokens(), 8);
        assert_eq!(layer.output().dims(), [8, 128]);
        assert_eq!(layer.spca().ops(), [Embedding::OP]);
        assert_eq!(layer.spca().role(), ParamRole::Embedding);
        assert!(layer.spca().role().wd_exempt());
    }

    #[test]
    fn an_embedding_floor_follows_the_gather_semantics() {
        let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
        let floor = layer.bounds().expect("合法").expect("有算子");
        // 独立按 gather 定义手算：8 个 token 各搬一条 128 宽的 f32 行。
        let elements = (8 * 128) as u64;
        assert_eq!(floor.read(), elements * 4, "读 8×128 个 f32");
        assert_eq!(floor.write(), elements * 4, "写 8×128 个 f32");
        assert_eq!(floor.flops(), 8.0, "每个 token 一次下标转换");
    }

    #[test]
    fn a_zero_dimension_embedding_is_rejected() {
        assert_eq!(
            Embedding::new("x", 0, 128, 8).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Embedding::new("x", 32, 0, 8).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            Embedding::new("x", 32, 128, 0).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn a_matmul_layer_binds_its_operator_and_shape() {
        let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
        assert_eq!(layer.name(), "blk.0.matmul");
        assert_eq!(layer.output().dims(), [16, 12]);
        assert_eq!(layer.spca().ops(), [MatMul::OP]);
        assert_eq!(layer.rows(), 16);
        assert_eq!(layer.cols(), 12);
    }

    #[test]
    fn the_matmul_floor_follows_the_gemm_semantics() {
        let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
        let floor = layer.bounds().expect("合法").expect("有算子");
        // 层不写第二份公式：下界必须与直接问语义表逐项相等。
        let same = Bounds::of_ctx(MatMul::OP, 16, 12).expect("同一份语义");
        assert_eq!(floor, same, "矩阵乘下界来自算子语义表");
        assert!(floor.read() > 0 && floor.write() > 0 && floor.flops() > 0.0);
    }

    #[test]
    fn a_zero_dimension_matmul_is_rejected() {
        assert_eq!(MatMul::new("x", 0, 12).unwrap_err().code(), Code::Invalid);
        assert_eq!(MatMul::new("x", 16, 0).unwrap_err().code(), Code::Invalid);
    }

    /// 单轴层与矩阵乘层都必须能当 `&dyn Contract` 用：装配侧只认这个视角。
    #[test]
    fn layers_work_as_trait_objects() {
        let axis: &dyn Contract = &Silu::new("mlp.silu", 64).expect("合法");
        assert_eq!(axis.output().dims(), [64]);
        assert!(axis.bounds().expect("合法").is_some());

        let matmul: &dyn Contract = &MatMul::new("proj", 16, 12).expect("合法");
        assert_eq!(matmul.output().dims(), [16, 12]);
        assert!(matmul.bounds().expect("合法").is_some());
    }

    /// 权重衰减豁免与可训练判定都从角色派生：归一化的系数不衰减，激活没有参数轴之外的例外。
    #[test]
    fn a_norm_layer_declares_the_norm_role_and_an_activation_the_trainable_role() {
        let norm = RmsNorm::new("blk.0.norm", 128).expect("合法");
        assert_eq!(norm.spca().role(), ParamRole::Norm);
        assert!(norm.spca().role().wd_exempt());

        let activation = Gelu::new("mlp.gelu", 128).expect("合法");
        assert_eq!(activation.spca().role(), ParamRole::Trainable);
    }

    /// 单轴层的通用断言：名字、形状、绑定的算子名，以及与语义表逐项相等的下界、非零的两笔流量。
    fn assert_layer<L: Contract>(layer: &L, name: &str, op: &str, len: usize) {
        assert_eq!(layer.name(), name);
        assert_eq!(layer.output().dims(), [len]);
        assert_eq!(layer.spca().ops(), [op]);
        let floor = layer.bounds().expect("单轴形状合法").expect("有算子");
        // 单轴算子不写第二份公式：下界必须与直接问语义表逐项相等。
        assert_eq!(
            floor,
            Bounds::of_ctx(op, len, 1).expect("同一份语义"),
            "{op} 的下界来自算子语义表"
        );
        assert!(floor.read() > 0 && floor.write() > 0, "{op} 有读有写");
    }

    /// 每个层只剩「类型 + 名字 + 长度」三项事实：构造、名字、算子名、形状、下界都由这两行给出，
    /// 于是加一个单轴层不必再抄五遍同样的断言。
    macro_rules! layer {
        ($ty:ident, $name:literal, $len:literal) => {
            assert_layer(&$ty::new($name, $len).expect("合法"), $name, $ty::OP, $len)
        };
    }

    /// 逐元素 / 归一化 / 激活层各自绑定自己的算子、各自从语义表取价。
    /// 这张清单也是「哪些层存在」的唯一一处：加层时这里必须多一行，否则新层没有任何断言。
    #[test]
    fn every_pointwise_layer_binds_its_declared_operator() {
        layer!(RmsNorm, "blk.0.norm", 128);
        layer!(Softmax, "attn.softmax", 64);
        layer!(Gelu, "mlp.gelu", 128);
        layer!(Silu, "mlp.silu", 128);
        layer!(LeakyRelu, "act.leaky", 128);
        layer!(HardSigmoid, "act.hard", 128);
        layer!(Rope, "attn.rope", 128);
        layer!(Scale, "proj.scale", 128);
        layer!(Add, "res.add", 128);
        layer!(Mul, "gate.mul", 128);
        layer!(Relu, "act.relu", 128);
        layer!(Sigmoid, "act.sigmoid", 128);
        layer!(Tanh, "act.tanh", 128);
        // 除 / 减 / 取大 / 取小 / 指数 / 对数此前算子表里有、层集里没有：并进来一并钉住。
        layer!(Sub, "blk.0.sub", 128);
        layer!(Div, "blk.0.div", 128);
        layer!(Max, "blk.0.max", 128);
        layer!(Min, "blk.0.min", 128);
        layer!(Exp, "blk.0.exp", 128);
        layer!(Log, "blk.0.log", 128);
    }

    /// 所有单轴层沿用同一条「零长度拒绝」口径，不留一个可以构造出空轴的后门。
    #[test]
    fn a_zero_length_axis_tensor_is_rejected() {
        assert_eq!(Gelu::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Silu::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(RmsNorm::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Sub::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Div::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Max::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Min::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Exp::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Log::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Relu::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Sigmoid::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Tanh::new("x", 0).unwrap_err().code(), Code::Invalid);
    }

    /// 收缩轴档位是算子表声明的固定档（`gemm_*` = 8、`matmul_*` = 4096），不是层能编造的尺寸。
    /// 实测两档的读侧与 flops 都精确相差 512 倍——这条用例锁的就是「档位真的进了计价」，
    /// 而不是层拿小档算子给大档层报一条偏小 512 倍的下界。
    #[test]
    fn the_matmul_gauge_is_declared_and_reaches_the_price() {
        let short =
            MatMul::with_precision("m", 16, 12, Precision::F32, Gauge::Short).expect("合法");
        let long = MatMul::with_precision("m", 16, 12, Precision::F32, Gauge::Long).expect("合法");
        assert_eq!(short.spca().ops(), ["gemm_f32"]);
        assert_eq!(long.spca().ops(), ["matmul_f32"]);
        let short = short.bounds().expect("合法").expect("有算子");
        let long = long.bounds().expect("合法").expect("有算子");
        assert_eq!(
            long.read(),
            short.read() * 512,
            "K 8 → 4096 是 512 倍的读数"
        );
        assert_eq!(
            long.flops(),
            short.flops() * 512.0,
            "同一条乘加，K 放大 512 倍"
        );
    }

    /// 精度决定读写字节、不决定算术量：同一档位下 int8 / int4 / int2 的读侧按元素宽度精确缩到
    /// 1/4、1/8、1/16，而 flops 与 f32 逐位相同——「几个数的乘加还是那么多次」。
    #[test]
    fn the_matmul_precision_scales_reads_and_never_the_arithmetic() {
        let floor = |precision| {
            MatMul::with_precision("m", 16, 12, precision, Gauge::Short)
                .expect("合法")
                .bounds()
                .expect("合法")
                .expect("有算子")
        };
        let (f32, i8, i4, i2) = (
            floor(Precision::F32),
            floor(Precision::Int8),
            floor(Precision::Int4),
            floor(Precision::Int2),
        );
        assert_eq!(i8.read() * 4, f32.read());
        assert_eq!(i4.read() * 8, f32.read());
        assert_eq!(i2.read() * 16, f32.read());
        for (label, other) in [("int8", i8), ("int4", i4), ("int2", i2)] {
            assert_eq!(other.flops(), f32.flops(), "{label} 与 f32 做同样多次乘加");
            assert_eq!(other.write(), f32.write(), "{label} 的输出仍是 f32");
        }
    }

    /// 算子表只注册了 int2 / int4 × `K = 8`：大收缩轴的这两个精度不存在。层的回答是
    /// 「没有这个组合」并说清缺哪一半，而不是退回小 K 的兄弟算子凑一条下界。
    #[test]
    fn an_unregistered_precision_rung_says_which_half_is_missing() {
        for precision in [Precision::Int2, Precision::Int4] {
            let layer =
                MatMul::with_precision("m", 16, 12, precision, Gauge::Long).expect("仍是合法声明");
            assert!(layer.spca().ops().is_empty(), "没有内核就没有算子名");
            let error = layer.bounds().unwrap_err();
            assert_eq!(error.code(), Code::Unsupported);
            let message = error.to_string();
            assert!(message.contains(precision.name()), "{message}");
            assert!(message.contains("K = 4096"), "{message}");
        }
    }

    /// 前缀扫描层：绑 `scan_f32`，读写覆盖整条时间轴，算术量是每个元素的递推。
    #[test]
    fn the_prefix_scan_binds_its_recurrence_operator_and_prices_it() {
        let scan = Scan::new("state", 16).expect("合法");
        assert_eq!(scan.spca().ops(), [Scan::OP]);
        assert_eq!(scan.output().dims(), [16], "扫描不改变长度");
        let floor = scan.bounds().expect("合法").expect("有算子");
        assert_eq!(floor, Bounds::of_ctx(Scan::OP, 16, 1).expect("同一份语义"));
        // 递推每元素读一次、写一次，两者都是 f32 满宽。
        assert_eq!(floor.read(), floor.write(), "读写各一次，各按 f32 满宽");
        assert_eq!(floor.read(), 16 * 4, "16 个 f32");
        assert!(floor.flops() > 0.0, "递推有算术，不是纯搬运");
        assert!(Scan::new("state", 0).is_err(), "零长度在构造期拒绝");
    }
}
