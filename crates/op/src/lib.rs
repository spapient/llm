//! 声明层与注册表：算子声明（`ops/<name>.toml`）、模板（`ops/<name>.k`）、它们的解析，
//! 以及「算子名 → 模板 → 语义表」的唯一一份注册表（[`REGISTRY`]）。
//!
//! 声明语料在**构建期**由 [`build.rs`](../build.rs) 用构建依赖 `toml` 解析 `ops/*.toml`，展开成
//! `diagnostic::data::{Table, Value}` 字面量写进 `$OUT_DIR/declared.rs`，再由 [`declared`] / [`names`]
//! 暴露（ADR 0022）。运行期只遍历这棵树：不读 `ops/`，也没有任何 TOML 解析器。
//!
//! 这一层是**内容**而不是机制：写算子的人只改本 crate 的 `ops/` 与 `src/<name>.rs`，`cutile` 负责协商能力、
//! 拼接片段、emit 与对拍。因此新增算子不必改 `cutile` 的任何一行代码，领域数据也不会
//! 与机制代码混在同一个 crate 里。
//!
//! # 三 crate 分层
//!
//! 本 crate 不做算术，只做装配。[`llm_spec`] 是契约（"算子是什么"：签名、布局、模板词法、
//! 代价表达式、语义模型、样本），[`llm_gemm`] 是稠密核（"一行乘一列怎么算"），
//! [`llm_conv`] 是神经算子（"窗口怎么滑过张量"）。依赖单向：
//! `spec ← gemm ← conv ← op`，于是内核不可能读到声明（它没有声明可读），
//! 「内核能否违反声明」这一问在编译期就有答案。
//!
//! 三个 crate 的模块与类型别名都从本 crate 根用 `pub use` 整体再导出，所以 12 个下游
//! 写的 `op::gemv::…` / `op::conv::…` / `op::gemv_f32` 一个字都不用改：门面是路径的承诺。
//!
//! 算子的标量参考与确定性样本在 [`mod@model`] 与 [`sample`]（第二份实现，与声明、模板共同构成
//! "正确性来自双源"，cutile README §11.1）；`cutile` 只提供对拍口径与执行环境，不认识任何算子名。
//!
//! 同一副骨架的多种精度写在 [`gemv`]（`gemm` 内）：一份泛型实现 + 每个格式一份
//! [`gemv::Format`] 契约。它们的 `MODEL` 由 `#[annotation::model]` 注解生成：配置在编译期定下，
//! "注解说的是哪份格式"由类型（`Quantized<32>` / `Dense<32>`）保证，写错就编译不过。
//!
//! 代价（`flops` / `bytes`）不在任何一层里手写：算子只声明 `EXPR`（如 `nm,m,ng,n->n`）与每个操作数
//! 的位宽/方向，由 [`einsum`] 推出；[`model::Cost`] 只是它的再导出。这样新增算子不会引入
//! 第二条流量公式，roofline 的分子也只有一处来源。
//!
//! # 例子
//!
//! ```
//! // 注册表里的每一条都自带声明与模板；要哪一条就按名字取，不靠位置。
//! for op in llm_op::builtin() {
//!     assert!(op.declaration().table("op").is_some(), "{}", op.name());
//!     assert!(!op.template().is_empty(), "{}", op.name());
//! }
//! let int4 = llm_op::builtin_named("gemv_i4").expect("已注册");
//! assert_eq!(int4.name(), "gemv_i4");
//! assert_eq!(
//!     llm_op::model("gemv_f32").map(|model| model.group),
//!     Some(1),
//!     "稠密算子的形状粒度是 1"
//! );
//! ```

#![deny(unsafe_code)]

pub mod decl;

// 内核与神经算子各自成 crate（分层见 `docs/ARCHITECTURE.md` §二）：`op` 只留声明层与注册表。
// 用 `pub use` 把两个 crate 的模块与别名整体搬回本 crate 根，12 个下游 crate 的
// `op::gemv::…` / `op::conv::…` / `op::gemv_f32` 路径零改动——门面是路径的承诺，不是再实现。
pub use llm_conv::{
    Scalar, adaptive_pool, avg_pool2d, batch_norm, conv2d, decode_attention, group_norm,
    instance_norm, interpolate, layer_norm, max_pool1d, max_pool2d, scan_f32,
};
pub use llm_conv::{attention, classic, conv, deconv, scan};
pub use llm_gemm::{
    activation, dequant, elementwise, fast, gemm, gemv, matmul, norm, reduce, unary,
};
pub use llm_gemm::{
    add_f32, dequant_i2, dequant_i4, dequant_i8, div_f32, embed_f32, exp_f32, gelu_f32, gemm_f32,
    gemm_i2, gemm_i4, gemm_i8, gemv_f32, gemv_i2, gemv_i4, gemv_i8, hard_sigmoid_f32,
    leaky_relu_f32, log_f32, matmul_i8, max_f32, min_f32, mul_f32, reduce_max_f32, reduce_sum_f32,
    residual_add_f32, rmsnorm_f32, rope_f32, scale_f32, silu_f32, softmax_f32, sub_f32,
};

// 契约层（签名、布局、模板词法、代价表达式、语义模型、样本、算子契约）在 `spec`：
// 这一层只回答"算子是什么"，被 `gemm` / `conv` / `op` 三方同时依赖而不成环。
// 用 `pub use` 兜住模块路径，12 个下游 crate 的 `op::model::…` / `op::sig::…` 零改动。
pub use llm_spec::same_bytes;
// `$crate::formula_cell!` / `$crate::operator_via!` 在骨架宏里按 crate 根解析，这里把 spec 导出的
// 两个宏搬回本 crate 根：12 个下游与本 crate 内部的 `#[model]` 展开都不必改路径。
pub use llm_spec::tpl::{formula, parse};
pub use llm_spec::{einsum, model, operator, sample, sch, sig, tpl};
pub use llm_spec::{formula_cell, operator_here, operator_via};
pub use model::quant;
pub use sample::{dropout, init, rnd};
pub use sch::layout;

use diagnostic::data::Table;

use model::Model;

/// 一次注册：一个算子的四件事——名字、模板、语义表、是否可微。
#[derive(Debug, Clone, Copy)]
pub struct Registration {
    /// 算子名（与 `ops/<name>.toml` 的文件名一致）。
    pub name: &'static str,
    /// `<name>.k` 的正文。
    pub template: &'static str,
    /// 代价、样本与标量参考。
    pub model: Model,
    /// 训练反向图里是否可微：`true` 的算子必须配套一条反向公式（`train::backprop` 在编译期
    /// 据此断言公式目录与可微算子集一一对应，缺一条或指向不可微算子都是编译错误）。
    pub differentiable: bool,
}

/// 内核表装配：一行一个「算子名 → 适配函数」，展开成 `Entry { op, run }` 的数组字面量。
#[macro_export]
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => {
        [$( Entry { op: stringify!($run), run: $run } ),*]
    };
}

/// 全量算子清单：**唯一一次**声明。注册表与上层类型化 API 都从这张表展开。
#[macro_export]
macro_rules! for_each_operator {
    ($m:ident) => {
        $m! {
            adaptive_pool_f32: $crate::adaptive_pool::AdaptivePoolModel => "adaptive_pool_f32",
            add_f32: $crate::add_f32::AddF32 => "binary",
            attention_f32: $crate::attention::AttentionF32 => "attention_f32",
            avg_pool2d_f32: $crate::avg_pool2d::AvgPool2dModel => "avg_pool2d_f32",
            batch_norm_f32: $crate::batch_norm::BatchNormModel => "batch_norm_f32",
            conv2d_f32: $crate::conv2d::Conv2dModel => "conv2d_f32",
            decode_attention_f32: $crate::attention::decode_attention::DecodeAttentionF32 => "decode_attention_f32",
            decode_flash_attention_f32: $crate::attention::decode_attention::DecodeFlashAttentionF32 => "decode_flash_attention_f32",
            dequant_i2: $crate::dequant_i2::DequantI2F32 => "dequant_packed",
            dequant_i4: $crate::dequant_i4::DequantI4F32 => "dequant_packed",
            dequant_i8: $crate::dequant_i8::DequantI8F32 => "dequant_i8",
            div_f32: $crate::div_f32::DivF32 => "binary",
            embed_f32: $crate::embed_f32::EmbedF32 => "embed_f32",
            exp_f32: $crate::exp_f32::ExpF32 => "exp_f32",
            flash_attention_f32: $crate::attention::FlashAttentionF32 => "flash_attention_f32",
            gelu_f32: $crate::gelu_f32::GeluF32 => "gelu_f32",
            gemm_f32: $crate::gemm_f32::GemmF32 => "gemm_f32",
            gemm_i2: $crate::gemm_i2::GemmI2Model => "gemm_packed",
            gemm_i4: $crate::gemm_i4::GemmI4Model => "gemm_packed",
            gemm_i8: $crate::gemm_i8::GemmI8Model => "gemm_i8",
            gemv_f32: $crate::gemv_f32::F32 => "gemv_f32",
            gemv_i2: $crate::gemv_i2::I2 => "gemv_packed",
            gemv_i4: $crate::gemv_i4::I4 => "gemv_packed",
            gemv_i8: $crate::gemv_i8::I8 => "gemv_i8",
            group_norm_f32: $crate::group_norm::GroupNormModel => "group_norm_f32",
            hard_sigmoid_f32: $crate::hard_sigmoid_f32::HardSigmoidF32 => "hard_sigmoid_f32",
            instance_norm_f32: $crate::instance_norm::InstanceNormModel => "instance_norm_f32",
            interpolate_f32: $crate::interpolate::InterpolateModel => "interpolate_f32",
            latent_attention_f32: $crate::attention::LatentAttentionF32 => "latent_attention_f32",
            layer_norm_f32: $crate::layer_norm::LayerNormModel => "layer_norm_f32",
            leaky_relu_f32: $crate::leaky_relu_f32::LeakyReluF32 => "leaky_relu_f32",
            log_f32: $crate::log_f32::LogF32 => "log_f32",
            matmul_f32: $crate::matmul::MatmulF32 => "matmul_f32",
            matmul_i8: $crate::matmul_i8::MatmulI8 => "matmul_i8",
            max_f32: $crate::max_f32::MaxF32 => "max_f32",
            max_pool1d_f32: $crate::max_pool1d::MaxPool1dModel => "max_pool1d_f32",
            max_pool2d_f32: $crate::max_pool2d::MaxPool2dModel => "max_pool2d_f32",
            min_f32: $crate::min_f32::MinF32 => "min_f32",
            mul_f32: $crate::mul_f32::MulF32 => "binary",
            reduce_max_f32: $crate::reduce_max_f32::ReduceMaxF32 => "reduce_max_f32",
            reduce_sum_f32: $crate::reduce_sum_f32::ReduceSumF32 => "reduce_sum_f32",
            residual_add_f32: $crate::residual_add_f32::ResidualAddF32 => "binary",
            rmsnorm_f32: $crate::rmsnorm_f32::RmsNormF32 => "rmsnorm_f32",
            rope_f32: $crate::rope_f32::RopeF32 => "rope_f32",
            scale_f32: $crate::scale_f32::ScaleF32 => "scale_f32",
            scan_f32: $crate::scan_f32::ScanF32 => "scan_f32",
            silu_f32: $crate::silu_f32::SiluF32 => "silu_f32",
            softmax_f32: $crate::softmax_f32::SoftmaxF32 => "softmax_f32",
            sub_f32: $crate::sub_f32::SubF32 => "binary",
        }
    };
}

/// 生成编译期内嵌的算子注册表：一行一个算子类型 + 模板 stem。
macro_rules! registry_impl {
    ($($name:ident: $ty:ty => $template:literal),* $(,)?) => {
        /// 编译期内嵌的算子注册表。
        pub const REGISTRY: [Registration; <[()]>::len(&[$(registry_impl!(@one $ty)),*])] = [
            $(
                Registration {
                    name: <$ty as crate::operator::Operator>::NAME,
                    template: include_str!(concat!("../ops/", $template, ".k")),
                    model: crate::operator::model_of::<$ty>(),
                    differentiable: <$ty as crate::operator::Operator>::DIFFERENTIABLE,
                }
            ),*
        ];

        /// 清单里的方法名必须与 `Operator::NAME` 逐字节相同（编译期）。
        const _: () = {
            $(
                ::core::assert!(
                    crate::same_bytes(
                        stringify!($name),
                        <$ty as crate::operator::Operator>::NAME,
                    ),
                    "算子清单的方法名与 Operator::NAME 不一致"
                );
            )*
        };
    };
    (@one $ty:ty) => { () };
}

crate::for_each_operator!(registry_impl);

/// 名字是否在注册表里；const 上下文可用。
pub const fn contains(name: &str) -> bool {
    let mut index = 0;
    while index < REGISTRY.len() {
        if same_bytes(name, REGISTRY[index].name) {
            return true;
        }
        index += 1;
    }
    false
}

/// 算子语义注册表：声明层有的算子，这里必须有它的代价、样本与标量参考。
pub fn model(name: &str) -> Option<Model> {
    REGISTRY
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.model)
}

/// 构建期生成的声明表（`build.rs` 读 `ops/*.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/declared.rs"));
}

pub use generated::{declared, names};

/// 一份算子声明：编译期生成的声明表与同名 `<name>.k` 模板。
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    name: &'static str,
    declaration: &'static Table,
    template: &'static str,
}

impl Op {
    /// 由名字、声明表与模板构造；名字取 `<name>.toml` 的文件名。
    pub fn new(name: &'static str, declaration: &'static Table, template: &'static str) -> Self {
        Self {
            name,
            declaration,
            template,
        }
    }

    /// 算子名（与 `<name>.toml` 的文件名一致）。
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// 声明表：`[op]` / `[axes]` / `[needs]` / `[layouts]` / `[[variants]]`。
    pub fn declaration(&self) -> &'static Table {
        self.declaration
    }

    /// 模板正文：C 风格文本 + `{{插值}}` + `@原语` 槽位。
    pub fn template(&self) -> &'static str {
        self.template
    }
}

/// 按名字取一条内嵌算子；名字不在注册表里时返回 `None`。
pub fn builtin_named(name: &str) -> Option<Op> {
    let entry = REGISTRY.iter().find(|entry| entry.name == name)?;
    Some(Op::new(entry.name, declared(entry.name)?, entry.template))
}

/// 内嵌声明，顺序与 [`REGISTRY`] 一致。
pub fn builtin() -> Vec<Op> {
    REGISTRY
        .iter()
        .filter_map(|entry| {
            let declaration = declared(entry.name)?;
            Some(Op::new(entry.name, declaration, entry.template))
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests/op_tests.rs"]
mod tests;

/// 注册表 ↔ 语义表：注解不再生成运行期测试，两者的一致性收在这一处核对。
#[cfg(test)]
#[path = "../tests/model_tests.rs"]
mod model_tests;

/// 各内核 `Operator` 投影与注册表逐字段一致（ADR 0050 §2）。
#[cfg(test)]
#[path = "../tests/operator_tests.rs"]
mod operator_tests;

/// 注册表 ↔ 内核的跨层核对：只有同时看得见声明层与内核的地方才能写这种测试。
#[cfg(test)]
#[path = "../tests/registry_tests.rs"]
mod registry_tests;
