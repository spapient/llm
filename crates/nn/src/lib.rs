//! 层定义：[`Contract`]、[`ContractSpec`]、投影层 [`Linear`]、矩阵乘层 [`MatMul`] 与 [`LmHead`]，单轴层
//! ([`RmsNorm`]、[`Softmax`]、激活 [`Gelu`]/[`Silu`]/[`Relu`]/[`Sigmoid`]/[`Tanh`]、[`Rope`]、
//! [`Scale`]、[`Add`]、[`Mul`])、查表层 [`Embedding`]、注意力层 [`Attention`] 与循环层
//! [`Recurrent`]；权重编码 [`WeightFormat`]、层的性能下界 [`Bounds`] 与层栈总下界 [`Stack`]。
//!
//! **拥有**：`Contract`、`ContractSpec`、上述各层、`WeightFormat`、`Bounds`、`Stack`。**不拥有**：算子声明与
//! 模板（`op`）、kernel 机制（`cutile`）、设备峰值（`prof`）、权重字节与解析（`weights`）。
//! **依赖**：`weights`、`tensor`、`op`、`diagnostic`。**被依赖**：`train`。
//!
//! `nn` 是 DSL 的**消费者**而不是并列层：层定义唯一的产出是一份 [`ContractSpec`]，里面的算子名
//! 必须已经存在于 `op::builtin()`。写错名字在这里就失败，而不是等到 emit 阶段才报"未知算子"。
//!
//! # 下界有三种状态，不是一个数
//!
//! [`Contract::bounds`] 返回 [`BoundsResult`]，三种情况刻意分开：
//!
//! | 返回值 | 含义 |
//! |---|---|
//! | `Ok(Some(bounds))` | 有下界，且它由算子语义推出（不是手写常数） |
//! | `Ok(None)` | 层已声明，但**没有可计价的形状**（例如注解层没给出输入宽度） |
//! | `Err(..)` | 有算子却算不出来：形状不合法（没有合法内核），或该 [`WeightFormat`] 还没有算子 |
//!
//! `Err` 不会被 [`Stack::floor`] 放大成"整张模型算不出下界"：它进无法计价清单，其余层照常相加。
//! 一个 LoRA 适配器或一张 f32 权重不该让 300 层的和归零，也不该因此得到一个偏低的数字。
//!
//! # 例子
//!
//! ```
//! use llm_nn::{Contract, Linear};
//!
//! let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("算子已声明");
//! assert_eq!(layer.output().dims(), [4096]);
//! assert_eq!(layer.spca().ops(), ["gemv_i4"]);
//!
//! // 下界按读/写分账：读侧是 int4 权重主导，写侧只有一份 f32 输出。
//! let floor = layer.bounds().expect("形状合法").expect("有算子");
//! assert!(floor.read() > floor.write());
//! ```

#![deny(unsafe_code)]

// 生成的代码用 `::nn::..` 指回本 crate，因此 crate 内部也必须能按名字解析到自己：
// 这样注解在 `nn` 自己身上（`lm_head` 模块里的 `LmHead`）与在下游 crate 里是同一套展开。
extern crate self as nn;

#[doc(inline)]
pub use annotation::spec;

use diagnostic::{Code, Error, Result};

mod attention;
mod declare;
mod engram;
mod format;
mod recurrent;
mod lm_head {
    //! `lm_head`：由注解声明的固定形状层。

    /// 固定形状的输出头：`[4096]` 输出、输入宽度 4096，权重编码由 `gemv_i4` 承担。
    #[crate::spec(name = "lm_head", out = [4096], in = 4096, ops = ["gemv_i4"])]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct LmHead;
}
mod param_role {
    //! 参数角色闭集：一个参数槽归哪个优化器组（D11；权重衰减豁免与可训练判定由角色派生）。

    /// 参数角色：优化器分组的唯一事实源。
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub enum ParamRole {
        /// 普通可训练权重（投影矩阵、稠密权重）：参与权重衰减。
        #[default]
        Trainable,
        /// 偏置向量：可训练，豁免权重衰减。
        Bias,
        /// 归一化系数（RMSNorm γ）：可训练，豁免权重衰减。
        Norm,
        /// 词表嵌入表：可训练，豁免权重衰减。
        Embedding,
    }

    impl ParamRole {
        /// 该角色是否进入可训练参数表：四个闭集角色都是参数槽。
        pub const fn trainable(self) -> bool {
            true
        }

        /// 是否豁免权重衰减：偏置、归一化系数与嵌入表不衰减。
        pub const fn wd_exempt(self) -> bool {
            matches!(self, Self::Bias | Self::Norm | Self::Embedding)
        }

        /// 角色短名：诊断、快照与报告里的稳定标签，与变体一一对应。
        pub const fn as_str(self) -> &'static str {
            match self {
                Self::Trainable => "trainable",
                Self::Bias => "bias",
                Self::Norm => "norm",
                Self::Embedding => "embedding",
            }
        }
    }
}
mod norm;
mod single;
mod spatial;
mod stack;

#[doc(inline)]
pub use attention::{Attention, AttentionAlgo, AttentionBlock, DecodeAttention, LatentAttention};

#[doc(inline)]
pub use declare::Bounds;

#[doc(inline)]
#[doc(inline)]
pub use engram::{Bypass, MemoryBlock, Table, allocate, next_prime, phi, slot_of, slots};

#[doc(inline)]
pub use lm_head::LmHead;

#[doc(inline)]
#[doc(inline)]
pub use format::WeightFormat;

#[doc(inline)]
pub use param_role::ParamRole;

#[doc(inline)]
pub use norm::{BatchNorm2d, GroupNorm, InstanceNorm, LayerNorm};

#[doc(inline)]
pub use spatial::{
    AdaptiveAvgPool2d, AvgPool2d, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, Interpolate,
    MaxPool1d, MaxPool2d,
};

#[doc(inline)]
pub use recurrent::{Cell, Recurrent};

#[doc(inline)]
pub use single::{
    Add, Div, Embedding, Exp, Gauge, Gelu, HardSigmoid, LeakyRelu, Log, MatMul, Max, Min, Mul,
    Precision, Relu, RmsNorm, Rope, Scale, Scan, Sigmoid, Silu, Softmax, Sub, Tanh,
};

#[doc(inline)]
pub use stack::{Floor, Stack, Unpriced, WEIGHT_SUFFIX};

/// 形状由 `tensor` 拥有（见 `docs/ARCHITECTURE.md` 第三节），这里再导出：`Contract::output` 本来就返回它，
/// 消费者与 `#[spec]` 生成的代码都不该为此额外依赖 `tensor`。
pub use tensor::Shape;

/// 已声明的算子名清单，仅用于报错文案（热路径不调用）。
fn declared_ops() -> String {
    let names: Vec<&str> = op::builtin().iter().map(op::Op::name).collect();
    format!("已声明：{}", names.join("、"))
}

/// 层下界的结果：[`Contract::bounds`] 的返回类型。
pub type BoundsResult = Result<Option<Bounds>>;

/// 逐个校验「必须非零」的维度，失败时点名装配位（`site`）。
///
/// `prefix` 自带分隔符（如 `"注意力层 "` / `"Embedding 的 "`），因此各层原有文案逐字保留。
fn nonzero_all(site: &str, prefix: &str, axes: &[(&str, usize)]) -> Result<()> {
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
fn nonzero(value: usize, field: &str, layer: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::new(Code::Invalid, format!("{layer}: `{field}` 必须非零")).with("nn"));
    }
    Ok(value)
}

/// 层定义产出的声明：名字、输出形状、要用的算子名与参数角色。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractSpec {
    name: String,
    output: Shape,
    ops: Vec<String>,
    role: ParamRole,
}

impl ContractSpec {
    /// 校验并构造：`ops` 必须既存在于 `op::builtin()`，又有 `op::model()` 的语义表。
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
        // 热路径不建整表：每层只按名字查一次（`builtin_named` 无分配），全表清单只在
        // 报错时才拼——装配一个几百层的模型不该为每条算子名付一次全表 `Vec` 的分配。
        let mut checked = Vec::new();
        for wanted in ops {
            let wanted = wanted.into();
            if op::builtin_named(&wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有声明（{}）", declared_ops()),
                )
                .with("nn::ContractSpec::new"));
            }
            if op::model(&wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有语义表，层无法推出下界也无法对拍"),
                )
                .with("nn::ContractSpec::new"));
            }
            checked.push(wanted);
        }
        Ok(Self {
            name: name.into(),
            output,
            ops: checked,
            role,
        })
    }

    /// 层名（与权重条目的前缀一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 参数角色：优化器分组与权重衰减豁免的派生依据。
    pub fn role(&self) -> ParamRole {
        self.role
    }

    /// 输出形状。
    pub fn output(&self) -> &Shape {
        &self.output
    }

    /// 要用的算子名。空切片表示这一层的编码还没有算子，因此无法计价。
    pub fn ops(&self) -> &[String] {
        &self.ops
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
    fn embedding(&self) -> Option<&Embedding> {
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
}

/// 线性层：`y = x · W`。用哪个算子由权重编码 [`WeightFormat`] 决定，而不是一个全局常量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    spec: ContractSpec,
    in_features: usize,
    out_features: usize,
    format: WeightFormat,
}

impl Linear {
    /// 默认编码（[`WeightFormat::INT4`]）对应的算子名。
    pub const OP: &str = "gemv_i4";

    /// 稠密 f32 权重对应的算子名。
    pub const DENSE: &str = "gemv_f32";

    /// 对称 int8 权重对应的算子名。
    pub const I8: &str = "gemv_i8";

    /// 对称 int2 权重对应的算子名。
    pub const I2: &str = "gemv_i2";

    /// 由输入/输出宽度构造，按 int4 权重（[`WeightFormat::INT4`]）计价；两个宽度都必须非零。
    pub fn new(name: impl Into<String>, in_features: usize, out_features: usize) -> Result<Self> {
        Self::with_format(name, WeightFormat::INT4, in_features, out_features)
    }

    /// 按给定的权重编码构造。
    ///
    /// 编码没有对应算子时（例如 f16 / bf16 / i8），层的 `ops` 为空：它仍然是一个合法的投影
    /// 声明，只是推不出下界。把它做成 `Err` 反而会逼调用方"随便挑一个格式"来让构造成功。
    ///
    /// # Errors
    ///
    /// 任一宽度为零（[`Code::Invalid`]）；算子名由编码给出却没有通过 [`ContractSpec::new`] 的校验
    /// （只可能发生在编码与 `op` 注册表不一致时）。
    pub fn with_format(
        name: impl Into<String>,
        format: WeightFormat,
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
            format.operator(),
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
        format -> WeightFormat = format,
    }

    /// 本层的性能下界，见 [`Contract::bounds`]。
    pub fn bounds(&self) -> BoundsResult {
        <Self as Contract>::bounds(self)
    }
}

impl Contract for Linear {
    spec_accessors!();

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
        Bounds::of_op(operator, self.in_features, self.out_features).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/prop_tests.rs"]
mod prop_tests;
