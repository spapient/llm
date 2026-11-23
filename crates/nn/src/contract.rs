//! 层契约三件：[`ContractSpec`]（已校验的名字/形状/算子/角色）、[`Contract`] 行接口与
//! [`Linear`] 投影层。配套的非零维度校验与下界结果别名也收在这里。
//!
//! 本模块只承载「层向装配侧声明什么」；下界算术（[`crate::Bounds`]）、单轴骨架（`declare`）
//! 与权重编码（[`crate::WeightFormat`]）分居另外两块，经 crate 根路径互相引用，不复制第二份常数。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::ParamRole;

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
    /// 校验并构造：`ops` 必须既存在于 [`prof::catalogue::all`]，又有 [`registry::model`] 的语义表。
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
    /// （只可能发生在编码与 `op` 注册表不一致时）。
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
        crate::Bounds::of_op(operator, self.in_features, self.out_features).map(Some)
    }
}
