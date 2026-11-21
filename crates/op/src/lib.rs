//! 声明层：算子声明（`ops/<name>.toml`）、模板（`ops/<name>.k`）与它们的解析。
//!
//! 声明语料在**构建期**由 [`build.rs`](../build.rs) 用构建依赖 `toml` 解析 `ops/*.toml`，展开成
//! `diagnostic::data::{Table, Value}` 字面量写进 `$OUT_DIR/declared.rs`，再由 [`declared`] / [`names`]
//! 暴露（ADR 0022）。运行期只遍历这棵树：不读 `ops/`，也没有任何 TOML 解析器。
//!
//! 这一层是**内容**而不是机制：写算子的人只改本 crate 的 `ops/` 与 `src/<name>.rs`，`cutile` 负责协商能力、
//! 拼接片段、emit 与对拍。因此新增算子不必改 `cutile` 的任何一行代码，领域数据也不会
//! 与机制代码混在同一个 crate 里。
//!
//! 两段文本的语义（骨架 / 轴 / 需求 / 布局 / 降级链，模板里的 `{{插值}}` 与 `@原语`）在这里
//! 被解析成[`decl::OpDecl`]，词汇表（[`layout`]、[`sch`]、[`tpl`]、[`sig`]）也在这里：机制层只消费
//! 解析结果，不再看 TOML。模板的入口签名是 ABI 的唯一事实源：实参顺序与输出缓冲名都从它派生。
//! 闭集的内容与书写规则见 [`README.md`](../../cutile/README.md) §4。
//!
//! 除了声明与模板，算子语义也在本 crate：求和表达式、确定性样本与标量参考（[`mod@model`]）。它们是
//! 第二份实现，与声明、模板共同构成"正确性来自双源"（cutile README §11.1）；`cutile` 只提供对拍口径
//! 与执行环境，不认识任何算子名。
//!
//! 同一副骨架的多种精度写在 [`gemv`]：一份泛型实现（求和、代价、样本、校验、遍历）+ 每个格式
//! 一份 [`gemv::Format`] 契约（权重元素、位宽、粒度、解包）。`gemv_i4` / `gemv_f32` 因此只写
//! "这份格式是什么"，不重写骨架。它们的 `MODEL` 由 `#[annotation::model]` 注解生成：配置在编译期
//! 定下，"注解说的是哪份格式"由类型（`Quantized<32>` / `Dense<32>`）保证，写错就编译不过。
//!
//! 代价（`flops` / `bytes`）不在本 crate 里手写：算子只声明 `EXPR`（如 `nm,m,ng,n->n`）与每个操作数
//! 的位宽/方向，由 [`einsum`] 推出；[`model::Cost`] 只是它的再导出（代价层并入本 crate 后
//! 连再导出也不再是跨 crate 补丁）。这样新增算子
//! 不会引入第二条流量公式，roofline 的分子也只有一处来源。
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
// `fast::sme` 的 SME 通道依赖两个不稳定特性，且**这两个特性名只在 aarch64 上存在**：
// x86_64 等其它目标根本不认识它们，直接 `#![feature(...)]` 会 E0635（unknown feature）。
// 因此用 `cfg_attr` 只在 aarch64 上声明，而非"声明了但在别的架构上无作用"——
// 特性门不是惰性的，名字在编译目标上不认识就是硬错误。
// - `aarch64_unstable_target_feature`：放行 `#[target_feature(enable = "sme")]`（尚未稳定）。
// - `stdarch_aarch64_feature_detection`：放行运行时探测 `is_aarch64_feature_detected!("sme")`。
#![cfg_attr(target_arch = "aarch64", feature(aarch64_unstable_target_feature))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_feature_detection))]
// `std::simd` 在两个地方用，且都靠它保持 `unsafe_code = "deny"` 不扩大开岛范围：
// `fast::sme::pack_a` 的分块转置（只存在于 aarch64），以及 `gemv::simd` 的解包点积——
// 后者**不按目标门控**，因为它是架构无关的 16 车道源码，aarch64 上落成 NEON、x86_64 上落成
// SSE2 基线，两个目标编译的是同一份算术，于是两条验证链跑的是同一段代码。
#![feature(portable_simd)]

pub mod activation;
pub mod attention;
pub mod classic;
pub mod conv;
pub mod decl;
pub mod deconv;
pub mod dequant;
pub mod elementwise;
pub mod fast;
pub mod gemm;
pub mod gemv;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod scan;
pub mod unary;

// 一元 / 二元 / 归约 / 反量化 / GEMV / GEMM / 注意力骨架内的声明式子模块，re-export 到 crate 根保持 `op::<name>::MODEL` / `forward` 路径。
pub use activation::{
    embed as embed_f32, rmsnorm as rmsnorm_f32, rope as rope_f32, softmax as softmax_f32,
};
pub use attention::decode_attention;
pub use classic::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d,
};
pub use dequant::{dequant_i2, dequant_i4, i8 as dequant_i8};
pub use elementwise::{add_f32, div_f32, max_f32, min_f32, mul_f32, residual_add_f32, sub_f32};
pub use gemm::{f32 as gemm_f32, i2 as gemm_i2, i4 as gemm_i4, i8 as gemm_i8};
pub use gemv::{f32 as gemv_f32, i2 as gemv_i2, i4 as gemv_i4, i8 as gemv_i8};
// 契约层（签名、布局、模板词法、代价表达式、语义模型、样本、算子契约）在 `op-spec`：
// 这一层只回答"算子是什么"，被 `op-kernel` / `op-neural` / `op` 三方同时依赖而不成环。
// 用 `pub use` 兜住模块路径，12 个下游 crate 的 `op::model::…` / `op::sig::…` 零改动。
pub use llm_op_spec::same_bytes;
// `$crate::formula_cell!` / `$crate::operator_via!` 在骨架宏里按 crate 根解析，这里把 spec 导出的
// 两个宏搬回本 crate 根：12 个下游与 all内部的 `#[model]` 展开都不必改路径。
pub use llm_op_spec::{formula_cell, operator_here, operator_via};
pub use llm_op_spec::tpl::{formula, parse};
pub use llm_op_spec::{einsum, model, operator, sample, sch, sig, tpl};
pub(crate) use llm_op_spec::tpl::guard;
pub use matmul::i8 as matmul_i8;
pub use model::quant;
pub use reduce::{reduce_max_f32, reduce_sum_f32};
pub use sample::{dropout, init, rnd};
pub use scan::{Scalar, f32 as scan_f32};
pub use sch::layout;
pub use unary::{
    exp_f32, gelu_f32, hard_sigmoid_f32, leaky_relu_f32, log_f32, scale_f32, silu_f32,
};

use diagnostic::data::Table;

use model::Model;

/// 一次注册：一个算子的四件事——名字、模板、语义表、是否可微。
///
/// **这是唯一一处注册表**：加一个算子只改这里。声明正文由 [`build.rs`](../build.rs) 在编译期
/// 从 `ops/<name>.toml` 生成，[`builtin()`] 与 [`model()`] 都从 [`REGISTRY`] 投影出来，不存在
/// "声明层有、语义层没有"（或反过来）的第二份事实。
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
///
/// `Entry` 由调用方定义（字段名固定为 `op` / `run`）；`train` 的全量参考表与 `serve` 的
/// f32 子集表共用这一份展开规则，名字与顺序的断言留在各自 crate（前者同序锁注册表、后者只查存在性）。
#[macro_export]
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => {
        [$( Entry { op: stringify!($run), run: $run } ),*]
    };
}

/// 全量算子清单：**唯一一次**声明。注册表与上层类型化 API 都从这张表展开。
///
/// 一行一个算子：`方法名: 类型路径 => 模板 stem`。方法名与
/// [`Operator::NAME`](operator::Operator::NAME) 在编译期逐字节对拍（见 `registry_impl!`），
/// 因此"清单写了一个名字、类型说另一个"无法通过编译；输入个数取 `Operator::ARITY`，
/// 由 `arity_matches_the_samples` 再与样本对拍。
///
/// 模板仍是独立字面量：多算子共用一份泛型模板（`gemv_i2`/`gemv_i4` → `gemv_packed`）时，
/// 模板名与注册名本来就不是同一个事实。
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
///
/// 名字、可微性、输入个数全部取自 [`Operator`](operator::Operator)，语义表是 [`model_of`] 的投影；
/// 一行的可变部分只剩方法名、类型与 `ops/<模板>.k` 的 stem。
macro_rules! registry_impl {
    ($($name:ident: $ty:ty => $template:literal),* $(,)?) => {
        /// 编译期内嵌的算子注册表。
        ///
        /// 新增算子在这里补一行，否则发布出去的二进制里没有它；
        /// `the_registry_covers_the_generated_table_exactly` 把"生成表里有、注册表里没有"（或反过来）
        /// 直接判为失败。数组长度由表本身数出来，不另写一个常数——那是第二个会漂移的数字。
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
///
/// 装配期的"内核表子集断言"走这个入口，归属判定只此一份；运行期查找仍由 [`builtin_named`]
/// / [`model()`] 给出整条记录。本工具链 const 上下文尚未稳定 `str` 相等比较，逐字节比对收在这里。
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
///
/// 语义表和声明同属一条注册（[`REGISTRY`]），因此两者不会漂移，`cutile` 一侧不动。
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
///
/// 两段内容都是 `'static`：声明表随二进制发布，模板由 `include_str!` 编进来，[`builtin`]
/// 不拷贝一个字符。运行期没有 TOML 文本，也没有解析器。
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
///
/// 存在的理由和 `model()` 一样：调用方要的是**某一条**算子，而不是"第一条算子"。按位置取
/// （`builtin()[0]`）在注册表从一条变成两条的那天会静默换人——而算子之间并不等价（稠密 f32
/// 与 int4 的代价差 8 倍）。
pub fn builtin_named(name: &str) -> Option<Op> {
    let entry = REGISTRY.iter().find(|entry| entry.name == name)?;
    Some(Op::new(entry.name, declared(entry.name)?, entry.template))
}

/// 内嵌声明，顺序与 [`REGISTRY`] 一致。
///
/// 声明表是编译进去的常量，模板也是；这里的 `Vec` 只装指针，不拷贝任何正文。注册表里
/// 缺一条生成表时会被跳过——`cargo test` 的 `every_builtin_entry_declares_an_operator`
/// 会把这种漂移判为失败。
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

#[cfg(test)]
#[path = "../tests/dropout_tests.rs"]
mod dropout_tests;

#[cfg(test)]
#[path = "../tests/einsum_tests.rs"]
mod einsum_tests;
