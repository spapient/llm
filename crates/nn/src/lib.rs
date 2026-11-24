//! 层定义：[`Contract`]、[`ContractSpec`]、投影层 [`Linear`]、矩阵乘层 [`MatMul`] 与 [`LmHead`]，单轴层
//! ([`RmsNorm`]、[`Softmax`]、激活 [`Gelu`]/[`Silu`]/[`Relu`]/[`Sigmoid`]/[`Tanh`]、[`Rope`]、
//! [`Scale`]、[`Add`]、[`Mul`])、查表层 [`Embedding`]、注意力层 [`Attention`] 与循环层
//! [`Recurrent`]；权重编码 [`WeightFormat`]、层的性能下界 [`Bounds`] 与层栈总下界 [`Stack`]。
//!
//! **拥有**：`Contract`、`ContractSpec`、上述各层、算子参考算术与注册表、`WeightFormat`、`Bounds`、`Stack`。**不拥有**：算子契约词汇与
//! 档案模板（`prof`）、kernel 机制（`cutile`）、权重字节与解析（`weights`）。
//! **依赖**：`weights`、`tensor`、`prof`、`annotation`、`diagnostic`。**被依赖**：`train`。
//!
//! `nn` 是 DSL 的**消费者**而不是并列层：层定义唯一的产出是一份 [`ContractSpec`]，里面的算子名
//! 必须已经存在于 `prof::catalogue::all()`。写错名字在这里就失败，而不是等到 emit 阶段才报"未知算子"。
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
// 不稳定特性一律挂在 `cfg(nightly)` 上（由本 crate 的 `build.rs` 按 `rustc --version` 判定）：
// nightly 上走手写车道，stable 上走 `kernel.rs` 里 `gemv::simd` 的标量退路，两边的能力差异
// 只在「向量化没向量化」，不在语义。`rust-toolchain.toml` 不必把整个工作区钉在 nightly。
//
// 本 crate 只用到 `portable_simd`（16 车道 `std::simd` 源码，架构无关，不按目标门控）。
// SME（`sme`）与 i8 dotprod（`dotprod`）的 `target_feature` 与运行时探测门已随瓦片通道迁入
// `prof::native`，各自在 `prof` 的 `lib.rs` 按 nightly + aarch64 声明；本 crate 经
// `cutile::native` 转发调用（nn → cutile → prof），不再持有这些门，也不直接 import `prof` 的内核层。
#![cfg_attr(nightly, feature(portable_simd))]

// 生成的代码用 `::nn::..` 指回本 crate，因此 crate 内部也必须能按名字解析到自己：
// 这样注解在 `nn` 自己身上（`lm_head` 模块里的 `LmHead`）与在下游 crate 里是同一套展开。
extern crate self as nn;

#[doc(inline)]
pub use annotation::spec;

mod attention;
mod contract;
mod declare;
mod engram;
mod recurrent;
mod weight;
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
mod single;
mod spatial;
mod stack;

#[doc(inline)]
pub use attention::{Attention, AttentionAlgo, AttentionBlock, DecodeAttention, LatentAttention};

#[doc(inline)]
pub use contract::{BoundsResult, Contract, ContractSpec, Linear};
#[doc(inline)]
pub use declare::{Bounds, OpLayer};
#[doc(inline)]
pub use weight::WeightFormat;

/// 构造期非零校验：层族共用的私有辅助，经 crate 根路径 `crate::nonzero` / `crate::nonzero_all` 引用。
pub(crate) use contract::{nonzero, nonzero_all};

#[doc(inline)]
pub use engram::{Bypass, MemoryBlock, Table, allocate, next_prime, phi, slot_of, slots};

#[doc(inline)]
pub use lm_head::LmHead;

#[doc(inline)]
pub use param_role::ParamRole;

#[doc(inline)]
pub use spatial::{
    AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d,
    GroupNorm, InstanceNorm, Interpolate, LayerNorm, MaxPool1d, MaxPool2d,
};

#[doc(inline)]
pub use recurrent::{Cell, Recurrent, RecurrentTensor};

#[doc(inline)]
pub use single::{
    Add, Div, Embedding, Exp, Gauge, Gelu, HardSigmoid, LeakyRelu, Log, MatMul, Max, Min, Mul,
    Precision, Relu, RmsNorm, Rope, Scale, Scan, Sigmoid, Silu, Softmax, Sub, Tanh,
};

#[doc(inline)]
pub use stack::{Floor, Stack, Unpriced, WEIGHT_SUFFIX};

/// 稠密计算核：矩阵乘族、激活、归一化、反量化、逐元素、归约与 SIMD/SME 通道。
///
/// 它原先独立成 crate（`gemm`），与本 crate 之间的唯一通路是 `op` 的 `pub use` 门面；合并后
/// 层定义直接引用内核，声明层退回 `prof`。
pub mod kernel;

/// 神经算子：卷积族、注意力族、池化与范数、扫描骨架（原独立 crate `conv`）。
pub mod conv;

/// 算子注册表：名字 → 语义表 + 可微性。与 `prof::catalogue` 按名字逐条对齐，错位在编译期失败。
pub mod registry;

/// 内核模块按原路径转出：内核源码与新下游都写 `nn::gemv::…` / `nn::gemm::…`。
///
/// 归一化**层**（契约登记）与空间层同居 `spatial`，`nn::kernel::norm` 才是它调用的**核**，
/// 同名不同物，靠命名空间区分而不是改名。
pub use kernel::{
    dequant, elementwise, embed, epilogue, fast, gemm, gemv, matmul, reduce, rmsnorm, rope, scalar,
    softmax, unary,
};

/// 卷积族模块按原路径转出：`nn::layer_norm` / `nn::adaptive_pool` / `nn::scan_f32` …
///
/// 这些名字在 `nn` 根都没有别的含义（归一化**层**是 `LayerNorm` 类型，不是 `layer_norm` 模块），
/// 所以平铺一层不会遮蔽任何东西；`nn::conv::*` 的完整路径同时保留。
pub use conv::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d, scan_f32,
};

/// 标量内核表：把「算子名 → 内核函数」装配成一个常量数组，供上层按注册名派发。
///
/// 只做语法搬运，表项类型由调用方决定（`Entry` 在展开点按名字解析）：名字取自注册表，
/// 函数取自本 crate 的参考内核，两者同序由调用方的 `const` 断言锁住。
#[macro_export]
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => {
        [$( Entry { op: stringify!($run), run: $run } ),*]
    };
}

/// 单算子入口（`gemv_f32` / `rmsnorm_f32` …）：下游原先写 `op::gemv_f32`，现在写 `nn::gemv_f32`。
pub use kernel::{
    Scalar, abs_f32, add_f32, apply_epilogue, ceil_f32, clamp_f32, cos_f32, dequant_i2, dequant_i4,
    dequant_i8, div_f32, embed_f32, exp_f32, exp2_f32, floor_f32, gelu_f32, gemm_f32, gemm_i2,
    gemm_i4, gemm_i8, gemv_f32, gemv_i2, gemv_i4, gemv_i8, hard_sigmoid_f32, leaky_relu_f32,
    log_f32, log1p_f32, matmul_i8, max_f32, min_f32, mul_f32, neg_f32, recip_f32, reduce_max_f32,
    reduce_prod_f32, reduce_sum_f32, relu_f32, remainder_f32, residual_add_f32, rmsnorm_f32,
    rope_f32, round_f32, rsqrt_f32, scale_f32, sigmoid_f32, sign_f32, silu_f32, sin_f32,
    softmax_f32, sqrt_f32, square_f32, sub_f32, tanh_f32, trunc_f32,
};

// 契约层在 `prof`：内核源码与骨架宏都按 crate 根解析契约路径，这里把它们搬回来，于是内核源码
// 里写的是 `crate::model::…` / `crate::operator_via!`，而不是到处写依赖 crate 名。
pub use model::quant;
pub use prof::same_bytes;
pub(crate) use prof::tpl::guard;
pub use prof::tpl::formula;
pub use prof::{einsum, model, operator, sample, sch, sig, tpl};
pub use prof::{formula_cell, formula_owner, operator_here, operator_via};
pub use sch::layout;

/// 形状由 `tensor` 拥有（见 `docs/ARCHITECTURE.md` 第三节），这里再导出：`Contract::output` 本来就返回它，
/// 消费者与 `#[spec]` 生成的代码都不该为此额外依赖 `tensor`。
pub use tensor::Shape;

#[cfg(test)]
#[path = "../tests/lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/prop_tests.rs"]
mod prop_tests;
