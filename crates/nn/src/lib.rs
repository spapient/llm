//! 层定义：[`Contract`]、[`ContractSpec`]、投影层 [`Linear`]、矩阵乘层 [`MatMul`] 与 [`LmHead`]，单轴层
//! ([`RmsNorm`]、[`Softmax`]、[`Gelu`]、[`Silu`]、[`Rope`]、[`Scale`]、[`Add`]、[`Mul`]、
//! [`ResidualAdd`])、归约层 ([`ReduceSum`]、[`ReduceMax`])、查表层 [`Embedding`] 与注意力层
//! [`Attention`]；权重编码 [`WeightFormat`]、层的性能下界 [`Bounds`] 与层栈总下界 [`Stack`]。
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
mod attention_block;
mod attnres;
mod axis;
mod bounds;
mod csa;
mod embed;
mod engram;
mod format;
mod kda;
mod lm_head {
    //! `lm_head`：由注解声明的固定形状层。
    //!
    //! 它曾住在 `src/main.rs`（L7 组装点），而 `docs/ARCHITECTURE.md` 的 L7 明写"不得定义新类型"：
    //! 类型属于产出它的层。形状、输入宽度与算子只写在注解里，`impl Contract` 与校验测试由
    //! [`crate::spec`] 生成。

    /// 固定形状的输出头：`[4096]` 输出、输入宽度 4096，权重编码由 `gemv_i4` 承担。
    ///
    /// 与 [`crate::Linear`] 的分工：宽度在编译期就定下来的层用字面量注解（声明即契约），宽度要按
    /// 模型配置计算出来的层用运行时形态（`$参数`，`spec` 字段与构造样板由注解生成）；两者都走同一
    /// 条 `ContractSpec` 校验路径，也走同一条下界路径：`in` 给了输入宽度，生成的 `bounds` 就把
    /// `(in, out)` 交给算子语义表。
    #[crate::spec(name = "lm_head", out = [4096], in = 4096, ops = ["gemv_i4"])]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct LmHead;
}
mod mamba;
mod matmul;
mod mhc;
mod mix;
mod mla;
mod moe;
mod param_role {
    //! 参数角色闭集：一个参数槽归哪个优化器组（D11；权重衰减豁免与可训练判定由角色派生）。
    //!
    //! 角色由层构造器写进 [`crate::ContractSpec`]；权重衰减豁免与可训练判定都从角色派生，优化器按
    //! 角色取组，不按字段名或层名前缀猜。闭集之外的角色在编译期不存在——新增一类参数必须先扩这个
    //! 闭集，而不是在优化器侧再写一份名字表。

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

    #[cfg(test)]
    mod tests {
        use super::*;

        /// 闭集四个角色都是参数槽：都进入可训练参数表。
        #[test]
        fn every_role_is_a_parameter_slot() {
            for role in [
                ParamRole::Trainable,
                ParamRole::Bias,
                ParamRole::Norm,
                ParamRole::Embedding,
            ] {
                assert!(role.trainable(), "{role:?} 应当是可训练参数槽");
            }
        }

        /// 权重衰减豁免只从角色派生：偏置、归一化系数与嵌入表不衰减，投影权重衰减。
        #[test]
        fn weight_decay_exemption_follows_the_role() {
            assert!(!ParamRole::Trainable.wd_exempt(), "投影权重参与衰减");
            for exempt in [ParamRole::Bias, ParamRole::Norm, ParamRole::Embedding] {
                assert!(exempt.wd_exempt(), "{exempt:?} 应豁免权重衰减");
            }
        }

        /// 短名与变体一一对应：诊断、快照与报告按这个名字对齐。
        #[test]
        fn short_names_match_the_variants() {
            assert_eq!(ParamRole::Trainable.as_str(), "trainable");
            assert_eq!(ParamRole::Bias.as_str(), "bias");
            assert_eq!(ParamRole::Norm.as_str(), "norm");
            assert_eq!(ParamRole::Embedding.as_str(), "embedding");
        }
    }
}
mod codec;
mod emit;
mod hybrid;
mod latent;
mod patch;
mod pointwise;
mod policy;
mod spike;
mod stack;

#[doc(inline)]
pub use attnres::{AttnRes, Sources, splice};

#[doc(inline)]
pub use attention::{Attention, AttentionAlgo};

#[doc(inline)]
pub use attention_block::AttentionBlock;

#[doc(inline)]
pub use bounds::Bounds;

#[doc(inline)]
pub use codec::{BYTES, ByteDecoder, ByteEncoder, ORDERS};

#[doc(inline)]
pub use emit::{Emission, Emitter, Halt};

#[doc(inline)]
pub use csa::{
    BlockTable, CSA_M, CSA_TOPK_FLASH, CSA_TOPK_PRO, HCA_M, WIN, compress, dense_softmax, dsa_topk,
    effective_len, is_doubly_stochastic, sinkhorn, spectral_norm,
};

#[doc(inline)]
pub use policy::{Calib, HEAD_LAYERS_TOKEN, brier};

#[doc(inline)]
pub use embed::Embedding;

#[doc(inline)]
pub use engram::{Bypass, MemoryBlock, Table, allocate, next_prime, phi, slot_of, slots};

#[doc(inline)]
pub use hybrid::{Hybrid, Kind, MEMORY_PER_ATTENTION, MemoryStep, Query, State, Tower};

#[doc(inline)]
pub use kda::{Kda, Memory};

#[doc(inline)]
pub use latent::{Backread, Global, LOCAL_MAX, LOCAL_MIN, Layer};

#[doc(inline)]
pub use lm_head::LmHead;

#[doc(inline)]
pub use mamba::{Carry, Mamba};

#[doc(inline)]
pub use matmul::MatMul;

#[doc(inline)]
pub use mhc::{MHC_STREAMS, Mhc};

#[doc(inline)]
pub use mla::{
    Latent, Mla, Path, V3_HEAD_DIM, V3_HEADS, V3_LATENT, V3_LAYERS, V3_ROPE, dense_width, ratio,
    row_width,
};

#[doc(inline)]
pub use moe::{K3_ACTIVE, K3_EXPERTS, Moe};

#[doc(inline)]
pub use format::WeightFormat;

#[doc(inline)]
pub use param_role::ParamRole;

#[doc(inline)]
pub use patch::{Boundary, Patches};

#[doc(inline)]
pub use pointwise::{Add, Gelu, Mul, ResidualAdd, RmsNorm, Rope, Scale, Silu, Softmax};

#[doc(inline)]
pub use policy::{
    Channel, Gating, MissRate, N_REF, TAU_SCALE, W, gamma, nu, pi, rms, sparsity, tau,
};

#[doc(inline)]
pub use mix::{Audit, Encoder, Events, Lif, accumulate, encode_id, steepest};

#[doc(inline)]
pub use pointwise::{ReduceMax, ReduceSum};

#[doc(inline)]
pub use spike::{Gradients, Master, Membrane, Spiking, Surrogate};

#[doc(inline)]
pub use stack::{Floor, Stack, Unpriced, WEIGHT_SUFFIX};

/// 形状由 `tensor` 拥有（见 `docs/ARCHITECTURE.md` 第三节），这里再导出：`Contract::output` 本来就返回它，
/// 消费者与 `#[spec]` 生成的代码都不该为此额外依赖 `tensor`。
pub use tensor::Shape;

/// 层下界的结果：[`Contract::bounds`] 的返回类型。
///
/// 单独起名是为了让 `#[spec]` 生成的方法签名不必让调用方依赖 `diagnostic`
/// （与 [`Shape`] 再导出是同一个理由）。
pub type BoundsResult = Result<Option<Bounds>>;

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
    /// 两条都要求是刻意的：只声明不实现的算子无法对拍，也就推不出它的[`Bounds`]，层一旦
    /// 引用它，下界与 `verify` 就都失去依据。把这条提前到构造期，比在 emit 或对拍阶段才发现
    /// 要便宜得多。
    ///
    /// 空 `ops` 是合法的：那表示"这一层存在，但它的编码还没有算子"（见 [`WeightFormat`]）。
    /// 这种情况不是校验失败——层仍然有自己的名字与输出形状，只是推不出下界。
    pub fn new(name: impl Into<String>, output: Shape, ops: Vec<String>) -> Result<Self> {
        Self::new_with_role(name, output, ops, ParamRole::Trainable)
    }

    /// 校验并构造，并显式声明参数角色；投影之外的层（归一化系数、偏置、嵌入表）必须经此入口
    /// 填角色——优化器分组从角色派生，猜字段名或层名前缀会把衰减组与豁免组放错。
    pub fn new_with_role(
        name: impl Into<String>,
        output: Shape,
        ops: Vec<String>,
        role: ParamRole,
    ) -> Result<Self> {
        let known = op::builtin();
        for wanted in &ops {
            if !known.iter().any(|declared| declared.name() == wanted) {
                return Err(Error::new(
                    Code::Missing,
                    format!(
                        "算子 `{wanted}` 没有声明（已声明：{}）",
                        known
                            .iter()
                            .map(op::Op::name)
                            .collect::<Vec<_>>()
                            .join("、")
                    ),
                )
                .with("nn::ContractSpec::new"));
            }
            if op::model(wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有语义表，层无法推出下界也无法对拍"),
                )
                .with("nn::ContractSpec::new"));
            }
        }
        Ok(Self {
            name: name.into(),
            output,
            ops,
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
    ///
    /// 默认实现是 `Ok(None)`：一个层只知道名字、输出形状与算子名，**不知道规约轴有多宽**，就把
    /// 不出算子的 `(m, n)`。默认给 `None` 而不是 `Err`，是因为"没有可计价的形状"不是错误；它也
    /// 不会被静默吞掉——[`Stack::floor`] 把它列进无法计价清单。
    ///
    /// 有形状的层（[`Linear`]、带 `in = ..` 的注解层）必须覆盖它。
    fn bounds(&self) -> BoundsResult {
        Ok(None)
    }

    /// 若这一层是按形状构造的线性投影，返回它的类型化视图。
    ///
    /// 默认 `None`：`#[spec]` 声明出来的层只有形状与算子列表，**没有参数表**，因此没有
    /// "把真实权重的第几行喂给它"这回事。真实权重的执行路径只处理 [`Linear`]——把"任意层
    /// 都能执行"当约定，等于让每个层自己再发明一遍参数绑定，而下界与执行也会各自理解一次形状。
    fn linear(&self) -> Option<&Linear> {
        None
    }

    /// 若这一层是查表层，返回它的类型化视图。
    ///
    /// 默认 `None`：与 [`linear`](Self::linear) 同一类类型化访问器，装配侧据此取词表/行宽几何，
    /// 不靠层类型匹配特判。
    fn embedding(&self) -> Option<&Embedding> {
        None
    }

    /// 若这一层是注意力组合块，返回它的类型化视图。
    ///
    /// 默认 `None`：装配侧经此取回 Q/K/V/注意力/输出投影五个子层与块几何，不在外层按层名前缀
    /// 分组扇出。
    fn attention_block(&self) -> Option<&crate::attention_block::AttentionBlock> {
        None
    }

    /// 若这一层是条件记忆层，返回它的类型化视图。
    ///
    /// 默认 `None`：与 [`linear`](Self::linear) 同一类类型化访问器。装配侧经此取回槽数/行宽/
    /// 阶/种子/门控五元几何，不在外层按层名或算子猜"这层是不是旁路"。
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
    ///
    /// 构造与下界都从这一处取，层里不会出现第二个字符串；编码到算子名的映射见
    /// [`WeightFormat::operator`]。
    pub const OP: &str = "gemv_i4";

    /// 稠密 f32 权重对应的算子名。
    ///
    /// 与 [`Linear::OP`] 同属一个骨架、同一批原语，差别只在权重怎么取（解包 vs 直接点积）。
    /// 两条算子并存，是因为它们的**代价不一样**：同一份 4096 方阵，f32 的读下界是 int4 的
    /// 8 倍，拿一个当另一个的替代品就是在报一个错得很像对的数。
    pub const DENSE: &str = "gemv_f32";

    /// 对称 int8 权重对应的算子名。
    ///
    /// 与 [`Linear::OP`] 同骨架：权重逐元素一字节、不打包，每 [`WeightFormat::Int8`] 声明的分组
    /// 共享一条标度；权重宽度是 int4 的两倍，读下界不能按 int4 套。
    pub const I8: &str = "gemv_i8";

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
        let ops = format.operator().map(str::to_string).into_iter().collect();
        let spec = ContractSpec::new_with_role(
            name,
            Shape::new([out_features]),
            ops,
            ParamRole::Trainable,
        )?;
        Ok(Self {
            spec,
            in_features,
            out_features,
            format,
        })
    }

    /// 输入宽度。
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 输出宽度。
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// 权重编码。它决定了本层用哪个算子，也决定了"这一层的下界是否可算"。
    pub fn format(&self) -> WeightFormat {
        self.format
    }

    /// 本层的性能下界，见 [`Contract::bounds`]。
    ///
    /// `(m, n)` 取 `(in_features, out_features)`：对本算子而言规约轴是输入宽度、输出轴是
    /// 输出宽度，与 `op` 里那条求和表达式 `nm,m,ng,n->n` 的轴顺序一致。
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
mod tests;
