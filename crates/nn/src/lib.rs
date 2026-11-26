//! 层定义：[`Contract`]、[`ContractSpec`]、投影层 [`Linear`]、矩阵乘层 [`MatMul`] 与 [`LmHead`]，单轴层
//! ([`RmsNorm`]、[`Softmax`]、激活 [`Gelu`]/[`Silu`]/[`Relu`]/[`Sigmoid`]/[`Tanh`]、[`Rope`]、
//! [`Scale`]、[`Add`]、[`Mul`])、查表层 [`Embedding`]、注意力层 [`Attention`] 与循环层
//! [`Recurrent`]；权重编码 [`WeightFormat`]、层的性能下界 [`Bounds`] 与层栈总下界 [`Stack`]。
//!
//! **拥有**：`Contract`、`ContractSpec`、上述各层、算子参考算术与注册表、`Bounds`、`Stack`。**不拥有**：算子契约词汇与
//! 档案模板（`prof`）、kernel 机制（`cutile`）、权重字节与解析（`weights`）、共享格式身份（`dtype`）。
//! **依赖**：`dtype`、`weights`、`tensor`、`prof`、`annotation`、`diagnostic`。**被依赖**：`train`。
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
// 不稳定特性一律挂在 `cfg(nightly)` 上：该 cfg 不由构建脚本探测，而由 nightly 验证环境经
// `RUSTFLAGS=--cfg=nightly` 注入；默认 stable 构建不挂，走 `gemv::simd` 的标量退路，两边的能力
// 差异只在「向量化没向量化」，不在语义。`rust-toolchain.toml` 不必把整个工作区钉在 nightly。
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
    #[derive(
        Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
    )]
    #[serde(rename_all = "snake_case")]
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
mod spatial;
mod stack;

#[doc(inline)]
pub use attention::{Attention, AttentionAlgo, AttentionBlock, DecodeAttention, LatentAttention};

#[doc(inline)]
pub use contract::{
    Add, BoundsResult, Contract, ContractSpec, Div, Embedding, Exp, Gauge, Gelu, HardSigmoid,
    LeakyRelu, Linear, Log, MatMul, Max, Min, Mul, Precision, Relu, RmsNorm, Rope, Scale, Scan,
    Sigmoid, Silu, Softmax, Sub, Tanh,
};
#[doc(inline)]
pub use declare::{Bounds, LayerGeometry, OpLayer};
#[doc(inline)]
pub use stack::{LayerConfig, ModelConfig};
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
pub use stack::{Floor, Stack, Unpriced, WEIGHT_SUFFIX};

/// 稠密计算核：矩阵乘族、激活、归一化、反量化、逐元素、归约与 SIMD/SME 通道。
///
/// 它原先独立成 crate（`gemm`），与本 crate 之间的唯一通路是 `op` 的 `pub use` 门面；合并后
/// 层定义直接引用内核，声明层退回 `prof`。
#[path = "clamp.rs"]
pub mod clamp_f32;
pub mod dequant;
pub mod elementwise;
pub mod embed;
pub mod epilogue;
pub mod fast;
pub mod gemm;
pub mod gemv;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod rmsnorm;
pub mod rope;
pub mod scalar;
pub mod softmax;
pub mod unary;

/// 神经算子：卷积族、注意力族、池化与范数、扫描骨架（原独立 crate `conv`）。
pub mod conv;

/// 算子注册表：名字 → 语义表 + 可微性。与 `prof::catalogue` 按名字逐条对齐，错位在编译期失败。
pub mod registry;

/// 卷积族模块按原路径转出：`nn::layer_norm` / `nn::adaptive_pool` / `nn::scan_f32` …
pub use conv::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d, scan_f32,
};

/// 标量内核表：把「算子名 → 内核函数」装配成一个常量数组，供上层按注册名派发。
#[macro_export]
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => {
        [$( Entry { op: stringify!($run), run: $run } ),*]
    };
}

pub use clamp_f32 as clamp;
pub use dequant::{dequant_i2, dequant_i4, i8 as dequant_i8};
pub use elementwise::{
    add_f32, add_f32x4, div_f32, max_f32, min_f32, mul_f32, remainder_f32, residual_add_f32,
    sub_f32,
};
pub use epilogue::apply as apply_epilogue;
pub use gemm::{f32 as gemm_f32, i2 as gemm_i2, i4 as gemm_i4, i8 as gemm_i8};
pub use gemv::{f32 as gemv_f32, i2 as gemv_i2, i4 as gemv_i4, i8 as gemv_i8};
pub use matmul::i8 as matmul_i8;
pub use reduce::{reduce_max_f32, reduce_prod_f32, reduce_sum_f32};
pub use scalar::Scalar;
pub use unary::abs_f16;
pub use unary::ops::{
    abs_f32, ceil_f32, cos_f32, exp_f32, exp2_f32, floor_f32, gelu_f32, hard_sigmoid_f32,
    leaky_relu_f32, log_f32, log1p_f32, neg_f32, recip_f32, relu_f32, round_f32, rsqrt_f32, scale,
    sigmoid_f32, sign_f32, silu_f32, sin_f32, sqrt_f32, square_f32, tanh_f32, trunc_f32,
};
pub use {embed as embed_f32, rmsnorm as rmsnorm_f32, rope as rope_f32, softmax as softmax_f32};

// 契约层在 `prof`：内核源码与骨架宏都按 crate 根解析契约路径，这里把它们搬回来，于是内核源码
// 里写的是 `crate::model::…` / `crate::operator_via!`，而不是到处写依赖 crate 名。
pub use model::quant;
pub use prof::same_bytes;
pub use prof::tpl::formula;
pub(crate) use prof::tpl::guard;
pub use prof::{einsum, model, operator, sample, sch, sig, tpl};
pub use prof::{formula_cell, formula_owner, operator_here, operator_via};
pub use sch::layout;

/// 形状由 `tensor` 拥有（见 `docs/ARCHITECTURE.md` 第三节），这里再导出：`Contract::output` 本来就返回它，
/// 消费者与 `#[spec]` 生成的代码都不该为此额外依赖 `tensor`。
pub use tensor::Shape;

#[cfg(test)]
mod tests {
    //! crate 表面：声明式 spec 宏在定义期就拒掉不认识的操作数、`ParamRole` 闭集与它派生的
    //! 两条判定、以及重导出面本身。`ParamRole` 的动作原在单独文件，挂同一模块故合并。

    use super::*;
    use diagnostic::Code;

    /// 只声明了输出、没有输入宽度：层合法，但没有可计价的形状。
    #[crate::spec(name = "no_axes", out = [8], ops = ["gemv_i4"])]
    struct NoAxes;

    /// 静态注解直接声明参数角色，不再需要手写一份 Contract 实现。
    #[crate::spec(name = "fixed_norm", out = [8], ops = ["rmsnorm_f32"], role = "norm")]
    struct FixedNorm;

    testkit::test_cases! {
    fn an_unknown_operator_is_rejected_at_definition_time() {
        let error = ContractSpec::new(
            "bad",
            Shape::new([4]),
            vec!["gemv_i4".to_string(), "flash_attention".to_string()],
        )
        .expect_err("未声明的算子");
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("flash_attention"), "{error}");
    }

    fn a_linear_layer_points_at_a_declared_operator() {
        let layer = Linear::new("blk.0.attn_q", 8, 4).expect("合法");
        assert_eq!(layer.name(), "blk.0.attn_q");
        assert_eq!(layer.output().dims(), [4]);
        assert_eq!(layer.spca().ops(), ["gemv_i4"]);
        assert_eq!(layer.out_features(), 4);
        assert_eq!(layer.format(), WeightFormat::INT4);
    }

    fn a_zero_width_layer_is_rejected() {
        assert_eq!(Linear::new("x", 0, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(Linear::new("x", 4, 0).unwrap_err().code(), Code::Invalid);
    }

    fn a_linear_layer_declares_its_floor() {
        let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("合法");
        let floor = layer.bounds().expect("生产形状").expect("有算子");
        // 层自己不算代价：它必须与算子语义逐个数字一致，否则就是第二份公式。
        assert_eq!(
            floor,
            Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义")
        );
        assert!(
            floor.read() > 8 * 1024 * 1024,
            "光 int4 权重就 8 MiB：{}",
            floor.read()
        );
    }

    fn a_layer_can_exist_without_a_legal_kernel_shape() {
        // 宽度非零，所以层能构造；但 8 不是量化分组的整数倍，它没有合法内核，也就没有下界。
        let layer = Linear::new("odd", 8, 4).expect("宽度非零即可构造");
        assert_eq!(
            layer.bounds().unwrap_err().code(),
            Code::Invalid,
            "形状非法仍然是错误，不是「无法计价」"
        );
    }

    fn a_format_without_an_operator_is_declared_but_unpriced() {
        // 同一份权重，文件里写的是 f16：层可以声明，但还没有 16 位元素的算子给它计价。
        let layer = Linear::with_format("blk.0.attn_q", WeightFormat::F16, 4096, 4096)
            .expect("宽度非零即可构造");
        assert!(layer.spca().ops().is_empty(), "没有算子可声明");
        let error = layer.bounds().unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("f16"), "{error}");
    }

    /// f32 权重现在有算子，而且报出的是它自己的代价（64 MiB 读），不是 int4 的 10.5 MB。
    fn a_dense_weight_is_priced_by_the_dense_operator() {
        let layer = Linear::with_format("blk.0.attn_q", WeightFormat::F32, 4096, 4096)
            .expect("宽度非零即可构造");
        assert_eq!(layer.spca().ops(), [Linear::DENSE]);
        let floor = layer.bounds().expect("合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_op(Linear::DENSE, 4096, 4096).expect("同一份语义")
        );
        // 读侧 = 权重（4096² 个 f32）+ 输入向量（4096 个 f32）：下界按方向分账，输入不是免费的。
        assert_eq!(
            floor.read(),
            64 * 1024 * 1024 + 4096 * 4,
            "f32 权重 + 一次输入向量的读下界"
        );
    }

    fn a_float_weight_is_not_priced_as_int4() {
        // 这就是改造前的隐性错误：同一个 [4096, 4096] 条目，f32 的存储本身是 64 MiB，
        // 而 int4 的读下界 ≈ 10.5 MB —— 差 6.4 倍，且旧代码会把前者报成后者。
        let int4 = Bounds::of_op(Linear::OP, 4096, 4096).expect("合法");
        let float_bytes = Shape::new([4096, 4096]).element_bytes(tensor::DType::F32);
        assert_eq!(float_bytes, 64 * 1024 * 1024);
        let ratio = float_bytes as f64 / int4.read() as f64;
        assert!((6.3..6.5).contains(&ratio), "f32 / int4 读侧 = {ratio}");
        // 两条算子的差不能被静默抹平：稠密那条报出自己的代价，量化那条报自己的。
        let dense = Linear::with_format("w", WeightFormat::F32, 4096, 4096).expect("可声明");
        let dense = dense.bounds().expect("合法").expect("有算子");
        assert_eq!(
            dense.read(),
            float_bytes as u64 + 4096 * 4,
            "f32 读下界 = 权重字节数 + 输入向量"
        );
        assert!(dense.read() > int4.read());
    }

    fn an_annotated_layer_is_a_checked_declaration() {
        let layer = LmHead;
        assert_eq!(layer.name(), "lm_head");
        assert_eq!(layer.output().dims(), [4096]);
        assert_eq!(layer.spca().ops(), ["gemv_i4"]);
    }

    fn an_annotated_layer_prices_its_floor_from_the_same_semantics() {
        let floor = LmHead.bounds().expect("形状合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义"),
            "注解层的下界与 Linear 同源"
        );
        // 与手写的同宽度投影逐项相等：注解只是另一种声明方式，不是另一条公式。
        let linear = Linear::new(LmHead.name(), 4096, 4096).expect("合法");
        assert_eq!(floor, linear.bounds().expect("合法").expect("有算子"));
    }

    fn an_annotated_layer_without_an_input_width_has_no_floor() {
        // 没有 `in`：层合法，但规约轴宽度未知，算子语义表无法被调用。
        assert_eq!(NoAxes.bounds().expect("不是错误"), None);
    }

    fn the_annotated_implementation_is_shared_by_every_instance() {
        let first = LmHead.spca();
        let second = LmHead.spca();
        assert!(core::ptr::eq(first, second), "规格只构造一次（OnceLock）");
    }

    fn a_spec_works_as_a_trait_object() {
        let layer = Linear::new("blk.0.attn_k", 32, 6).expect("合法");
        let dynamic: &dyn Contract = &layer;
        assert_eq!(dynamic.spca().name(), "blk.0.attn_k");
        assert!(dynamic.bounds().expect("合法").is_some());
    }

    fn a_spec_without_an_explicit_role_defaults_to_trainable() {
        let spec = ContractSpec::new("legacy", Shape::new([4]), vec!["gemv_i4".to_string()])
            .expect("算子已声明");
        assert_eq!(spec.role(), ParamRole::Trainable);
    }

    fn an_explicit_role_is_round_trip_declared() {
        let spec = ContractSpec::new_with_role(
            "blk.0.norm",
            Shape::new([128]),
            vec!["rmsnorm_f32".to_string()],
            ParamRole::Norm,
        )
        .expect("算子已声明");
        assert_eq!(spec.role(), ParamRole::Norm);
        assert!(spec.role().wd_exempt());
    }

    fn a_projection_layer_declares_the_trainable_role() {
        let layer = Linear::new("blk.0.attn_q", 8, 4).expect("合法");
        assert_eq!(layer.spca().role(), ParamRole::Trainable);
        assert!(!layer.spca().role().wd_exempt());
    }

    /// 闭集四个角色都是参数槽：都进入可训练参数表。
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
    fn weight_decay_exemption_follows_the_role() {
        assert!(!ParamRole::Trainable.wd_exempt(), "投影权重参与衰减");
        for exempt in [ParamRole::Bias, ParamRole::Norm, ParamRole::Embedding] {
            assert!(exempt.wd_exempt(), "{exempt:?} 应豁免权重衰减");
        }
    }

    /// 短名与变体一一对应：诊断、快照与报告按这个名字对齐。
    fn short_names_match_the_variants() {
        assert_eq!(ParamRole::Trainable.as_str(), "trainable");
        assert_eq!(ParamRole::Bias.as_str(), "bias");
        assert_eq!(ParamRole::Norm.as_str(), "norm");
        assert_eq!(ParamRole::Embedding.as_str(), "embedding");
    }

    /// 运行期配置化层：算子名、长度与角色都是构造参数，而契约面必须与编译期类型层逐字一致。
    ///
    /// 这不是"两条实现看起来差不多"，而是同一个 `AxisCore` 的两种入口：配置化若与类型化
    /// 有任何一处口径不同（算子、角色、输出形状、下界），装配侧就会按配置拼出一份「和类型层
    /// 说的不一样」的模型，而它看起来完全合法。
    fn a_configured_layer_declares_like_its_compiled_counterpart() {
        let configured = OpLayer::new("blk.0.norm", RmsNorm::OP, 128, ParamRole::Norm).expect("合法");
        let compiled = RmsNorm::new("blk.0.norm", 128).expect("合法");

        assert_eq!(configured.name(), compiled.name());
        assert_eq!(configured.output().dims(), compiled.output().dims());
        assert_eq!(configured.spca(), compiled.spca());
        assert_eq!(
            configured.bounds().expect("合法"),
            compiled.bounds().expect("合法")
        );
        assert_eq!(configured.op(), RmsNorm::OP);
        assert_eq!(configured.len(), 128);
        assert!(!configured.is_empty());
    }

    /// 配置化层拒绝零长度：与 `axis!` 生成的层同一条构造期判据。
    fn a_configured_layer_rejects_a_zero_length() {
        let error = OpLayer::new("n", RmsNorm::OP, 0, ParamRole::Norm).expect_err("零长度");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// `Hash` 让层直接当配置表的键：同一份声明落到同一个桶，不同权重编码不合并。
    ///
    /// 键相等必须与层相等同义——`Linear` 的 `format` 进 `PartialEq` 也进 `Hash`，否则
    /// int4 与 int8 两个只差编码的层会在表里互相顶掉。
    fn layers_are_usable_as_map_keys() {
        use std::collections::HashMap;

        let mut linear: HashMap<Linear, &str> = HashMap::new();
        let int4 = Linear::new("blk.0.q", 8, 4).expect("合法");
        let i8 = Linear::with_format("blk.0.q", WeightFormat::INT8, 8, 4).expect("合法");
        linear.insert(int4.clone(), "int4");
        linear.insert(i8.clone(), "int8");
        assert_eq!(linear.get(&int4), Some(&"int4"));
        assert_eq!(linear.get(&i8), Some(&"int8"));
        assert_eq!(linear.len(), 2, "编码不同就是不同的键");

        let mut configured: HashMap<OpLayer, usize> = HashMap::new();
        let layer = OpLayer::new("n", RmsNorm::OP, 128, ParamRole::Norm).expect("合法");
        configured.insert(layer.clone(), 1);
        assert_eq!(configured.get(&layer), Some(&1));
    }

    fn a_static_annotation_declares_its_parameter_role() {
        assert_eq!(FixedNorm.spca().role(), ParamRole::Norm);
        assert!(FixedNorm.spca().role().wd_exempt());
    }

    fn a_config_stack_reuses_the_typed_layers_and_their_views() {
        let stack = Stack::from_configs([
            LayerConfig::Axis {
                name: "blk.0.norm".into(),
                op: RmsNorm::OP.to_string(),
                len: 128,
                role: ParamRole::Norm,
            },
            LayerConfig::Linear {
                name: "blk.0.q".into(),
                in_features: 128,
                out_features: 64,
                format: WeightFormat::INT4,
            },
            LayerConfig::AttentionBlock {
                name: "blk.1".into(),
                hidden: 128,
                kv_width: 64,
                head_dim: 32,
                seq: 16,
                algo: AttentionAlgo::Flash,
                format: WeightFormat::INT4,
            },
        ])
        .expect("配置合法");

        assert_eq!(stack.layers().len(), 3);
        assert_eq!(stack.layers()[0].output().dims(), [128]);
        assert_eq!(
            stack.layers()[1].linear().expect("线性视图").out_features(),
            64
        );
        assert_eq!(
            stack.layers()[2]
                .attention_block()
                .expect("组合块视图")
                .heads(),
            2
        );
        assert!(stack.floor().unpriced().is_empty());
    }

    fn config_construction_keeps_constructor_and_stack_diagnostics() {
        let invalid = LayerConfig::Axis {
            name: "bad".into(),
            op: "not_registered".into(),
            len: 8,
            role: ParamRole::Trainable,
        };
        let error = Stack::from_configs([invalid]).expect_err("未知算子");
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("layers[0]"), "{error}");
        assert!(error.to_string().contains("type = axis"), "{error}");

        let duplicate = || LayerConfig::Linear {
            name: "same".into(),
            in_features: 32,
            out_features: 32,
            format: WeightFormat::INT4,
        };
        let error = Stack::from_configs([duplicate(), duplicate()]).expect_err("重名");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn a_spec_centralizes_structural_invariants() {
        for name in ["", "  \t"] {
            let error = ContractSpec::new(name, Shape::new([4]), [RmsNorm::OP]).expect_err("空名字");
            assert_eq!(error.code(), Code::Invalid);
        }
        let error = ContractSpec::new("zero", Shape::new([4, 0]), [RmsNorm::OP]).expect_err("零宽轴");
        assert_eq!(error.code(), Code::Invalid);
        let error = ContractSpec::new("overflow", Shape::new([usize::MAX, 2]), [RmsNorm::OP])
            .expect_err("元素数溢出");
        assert_eq!(error.code(), Code::Invalid);
        let error = ContractSpec::new("duplicate", Shape::new([4]), [RmsNorm::OP, RmsNorm::OP])
            .expect_err("重复算子");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn model_config_is_strict_serializable_and_buildable() {
        let text = r#"
    [[layers]]
    type = "axis"
    name = "blk.0.norm"
    op = "rmsnorm_f32"
    len = 128

    [[layers]]
    type = "linear"
    name = "blk.0.q"
    in_features = 128
    out_features = 64

    [layers.format]
    type = "int4"
    group = 32

    [[layers]]
    type = "recurrent"
    name = "encoder"
    cell = "gru"
    input = 128
    hidden = 64
    "#;
        let config: ModelConfig = toml::from_str(text).expect("严格 TOML 配置");
        assert_eq!(config.layers.len(), 3);
        assert_eq!(config.layers[0].kind(), "axis");
        assert_eq!(config.layers[0].name(), "blk.0.norm");
        assert_eq!(config.layers[1].kind(), "linear");

        let encoded = toml::to_string(&config).expect("配置可序列化");
        let round_trip: ModelConfig = toml::from_str(&encoded).expect("配置往返");
        assert_eq!(round_trip, config);

        let stack = Stack::from_config(config).expect("模型配置装配");
        assert_eq!(stack.layers()[0].spca().role(), ParamRole::Trainable);
        let recurrent = stack.layers()[2].recurrent().expect("循环层视图");
        assert_eq!(recurrent.layers(), 1);
        assert_eq!(recurrent.dirs(), 1);
        assert!(recurrent.bias());
        assert!(!recurrent.batch_first());
    }

    fn model_config_rejects_misspelled_fields_at_every_level() {
        let unknown_model = "model_name = \"bad\"\nlayers = []\n";
        assert!(toml::from_str::<ModelConfig>(unknown_model).is_err());

        let unknown_layer = r#"
    [[layers]]
    type = "axis"
    name = "norm"
    op = "rmsnorm_f32"
    len = 8
    length = 8
    "#;
        assert!(toml::from_str::<ModelConfig>(unknown_layer).is_err());

        let unknown_format = r#"
    [[layers]]
    type = "linear"
    name = "q"
    in_features = 8
    out_features = 8
    [layers.format]
    type = "int4"
    group = 32
    groups = 32
    "#;
        assert!(toml::from_str::<ModelConfig>(unknown_format).is_err());
    }

    fn layer_config_kind_matches_its_wire_name() {
        let config = LayerConfig::MatMul {
            name: "project".into(),
            rows: 2,
            cols: 3,
            precision: Precision::F32,
            gauge: Gauge::Short,
        };
        assert_eq!(config.kind(), "matmul");
        let encoded = toml::to_string(&ModelConfig::new(vec![config])).expect("配置序列化");
        assert!(encoded.contains("type = \"matmul\""), "{encoded}");
    }

    fn every_layer_config_has_a_stable_unique_wire_name() {
        let configs = [
            LayerConfig::Axis {
                name: "a".into(),
                op: RmsNorm::OP.into(),
                len: 8,
                role: ParamRole::Norm,
            },
            LayerConfig::Linear {
                name: "l".into(),
                in_features: 8,
                out_features: 8,
                format: WeightFormat::INT4,
            },
            LayerConfig::MatMul {
                name: "m".into(),
                rows: 1,
                cols: 8,
                precision: Precision::F32,
                gauge: Gauge::Short,
            },
            LayerConfig::Embedding {
                name: "e".into(),
                vocab: 16,
                dim: 8,
                tokens: 1,
            },
            LayerConfig::Attention {
                name: "a".into(),
                seq: 1,
                head_dim: 8,
                algo: AttentionAlgo::Flash,
            },
            LayerConfig::DecodeAttention {
                name: "d".into(),
                q_rows: 1,
                head_dim: 8,
                algo: AttentionAlgo::Flash,
            },
            LayerConfig::LatentAttention {
                name: "la".into(),
                seq: 1,
                latent: 8,
                heads: 1,
            },
            LayerConfig::AttentionBlock {
                name: "ab".into(),
                hidden: 8,
                kv_width: 8,
                head_dim: 8,
                seq: 1,
                algo: AttentionAlgo::Flash,
                format: WeightFormat::INT4,
            },
            LayerConfig::Recurrent {
                name: "r".into(),
                cell: Cell::Gru,
                input: 8,
                hidden: 8,
                layers: 1,
                dirs: 1,
                bias: true,
                batch_first: false,
            },
            LayerConfig::Memory {
                name: "mem".into(),
                width: 8,
                order: 3,
                seed: 7,
                buckets: 32,
                gate: 0.5,
            },
        ];
        let expected = [
            "axis",
            "linear",
            "matmul",
            "embedding",
            "attention",
            "decode_attention",
            "latent_attention",
            "attention_block",
            "recurrent",
            "memory",
        ];
        let mut names = std::collections::BTreeSet::new();
        for (config, expected) in configs.into_iter().zip(expected) {
            assert_eq!(config.kind(), expected);
            assert!(names.insert(expected), "配置短名必须唯一");
            let encoded = toml::to_string(&config).expect("层配置可序列化");
            let decoded: LayerConfig = toml::from_str(&encoded).expect("层配置可反序列化");
            assert_eq!(decoded, config);
        }
        assert_eq!(names.len(), 10);
    }
    }
}

#[cfg(test)]
#[path = "../tests/prop_tests.rs"]
mod prop_tests;

#[cfg(test)]
#[path = "../tests/kernel_prop_numeric.rs"]
mod prop_numeric;

#[cfg(test)]
mod pointwise_tests {
    include!("../tests/kernel_pointwise_tests.rs");
}
