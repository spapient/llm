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

// `#[annotation::model]` 生成的代码用 `::llm_op::..` 指回本 crate（与 `nn` 的
// `extern crate self as nn` 同一条约定：注解绑定它所属契约的路径，而不是硬编码调用方的名字）。
extern crate self as llm_op;

pub mod attention;
pub mod attention_tiled;
pub mod decl;
pub mod dequant;
pub mod einsum;
pub mod elementwise;
pub mod fast;
pub mod formula;
pub mod classic;
pub mod conv;
pub mod deconv;
pub mod norm;
pub mod dropout;
pub mod init;
pub mod rnd;
pub mod rnn;
pub mod gemm;
pub mod gemv;
pub mod matmul;
pub mod model;
pub mod operator;
pub mod softmax;
pub(crate) use tpl::parse;
pub mod reduce;
pub mod scan;
pub mod sch;
pub mod sig;
pub mod tpl;
pub mod unary;

// 一元 / 二元 / 归约 / 反量化 / GEMV / GEMM / 注意力骨架内的声明式子模块，re-export 到 crate 根保持 `op::<name>::MODEL` / `forward` 路径。
pub use attention::decode_attention;
pub use dequant::{dequant_i2, dequant_i4, dequant_i8};
pub use classic::{
    adaptive_pool_f32, avg_pool2d_f32, batch_norm_f32, conv2d_f32, group_norm_f32,
    instance_norm_f32,
    interpolate_f32, layer_norm_f32, max_pool1d_f32, max_pool2d_f32,
};
pub use elementwise::{add_f32, div_f32, max_f32, min_f32, mul_f32, residual_add_f32, sub_f32};
pub use gemm::{gemm_f32, gemm_i2, gemm_i4, gemm_i8};
pub use gemv::{gemv_f32, gemv_i2, gemv_i4, gemv_i8};
pub use matmul::matmul_i8;
pub use model::quant;
pub use reduce::{reduce_max_f32, reduce_sum_f32};
pub use scan::scan_f32;
pub use sch::layout;
pub use softmax::{embed_f32, rmsnorm_f32, rope_f32, softmax_f32};
pub use unary::{exp_f32, gelu_f32, hard_sigmoid_f32, leaky_relu_f32, log_f32, scale_f32, silu_f32};

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
            adaptive_pool_f32: $crate::adaptive_pool_f32::AdaptivePoolModel => "adaptive_pool_f32",
            add_f32: $crate::add_f32::AddF32 => "binary",
            attention_f32: $crate::attention::AttentionF32 => "attention_f32",
            avg_pool2d_f32: $crate::avg_pool2d_f32::AvgPool2dModel => "avg_pool2d_f32",
            batch_norm_f32: $crate::batch_norm_f32::BatchNormModel => "batch_norm_f32",
            conv2d_f32: $crate::conv2d_f32::Conv2dModel => "conv2d_f32",
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
            group_norm_f32: $crate::group_norm_f32::GroupNormModel => "group_norm_f32",
            hard_sigmoid_f32: $crate::hard_sigmoid_f32::HardSigmoidF32 => "hard_sigmoid_f32",
            instance_norm_f32: $crate::instance_norm_f32::InstanceNormModel => "instance_norm_f32",
            interpolate_f32: $crate::interpolate_f32::InterpolateModel => "interpolate_f32",
            layer_norm_f32: $crate::layer_norm_f32::LayerNormModel => "layer_norm_f32",
            leaky_relu_f32: $crate::leaky_relu_f32::LeakyReluF32 => "leaky_relu_f32",
            log_f32: $crate::log_f32::LogF32 => "log_f32",
            matmul_f32: $crate::matmul::MatmulF32 => "matmul_f32",
            matmul_i8: $crate::matmul_i8::MatmulI8 => "matmul_i8",
            max_f32: $crate::max_f32::MaxF32 => "max_f32",
            max_pool1d_f32: $crate::max_pool1d_f32::MaxPool1dModel => "max_pool1d_f32",
            max_pool2d_f32: $crate::max_pool2d_f32::MaxPool2dModel => "max_pool2d_f32",
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

/// `const` 上下文里逐字节比两个名字（没有可用的 `str` 比较）。
///
/// 全仓共用这一份：`train` 的内核表、注册表清单与 `Operator::NAME` 的编译期对拍、以及 `train`
/// 的内核表对拍都调它，不再各写一遍。
pub const fn same_bytes(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
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
mod tests {
    use super::*;
    use diagnostic::Code;
    use diagnostic::data::Value;

    #[test]
    fn every_builtin_entry_declares_an_operator() {
        for op in builtin() {
            assert!(!op.name().is_empty());
            assert!(op.declaration().table("op").is_some(), "{}", op.name());
            assert!(!op.template().is_empty(), "{}", op.name());
        }
    }

    /// `Operator::ARITY` 必须等于样本里非输出缓冲（`y`）的个数：类型化 API 的定长输入以此为准。
    #[test]
    fn arity_matches_the_samples() {
        macro_rules! check {
            ($($name:ident: $ty:ty => $template:literal),* $(,)?) => {
                $(
                    let shape = <$ty as crate::operator::Operator>::SHAPE;
                    let sample = <$ty as crate::operator::Operator>::sample(shape.0, shape.1);
                    let inputs = sample.buffers.iter().filter(|buffer| buffer.name != "y").count();
                    assert_eq!(
                        <$ty as crate::operator::Operator>::ARITY,
                        inputs,
                        "算子 `{}` 的 ARITY 与样本输入个数不同",
                        stringify!($name)
                    );
                )*
            };
        }
        crate::for_each_operator!(check);
    }

    /// 生成表按文件名升序（旧的 `op::load(dir)` 也这样排序）。
    #[test]
    fn declared_names_are_sorted() {
        let generated = names();
        assert!(
            generated.windows(2).all(|pair| pair[0] < pair[1]),
            "{generated:?}"
        );
        for name in generated {
            assert!(declared(name).is_some(), "{name} 有名字却没有表");
        }
    }

    /// 注册表与构建期生成的声明表必须逐条对上。
    ///
    /// 少一条是"声明层有、注册表没有"——[`builtin`] 会静默跳过它，于是算子发布了却取不到；
    /// 多一条是"注册表有、`ops/` 里没有这个文件"，`include_str!` 让它在编译期就挂。两个方向都判。
    #[test]
    fn the_registry_covers_the_generated_table_exactly() {
        let registered: Vec<&str> = REGISTRY.iter().map(|entry| entry.name).collect();
        assert_eq!(
            registered,
            names(),
            "注册表与 `ops/*.toml` 的算子集必须一一对应"
        );
        assert_eq!(builtin().len(), REGISTRY.len(), "每条注册都必须有声明");
    }

    /// 声明与语义同属一条注册：不该出现"内嵌里有、语义表没有"的算子。
    #[test]
    fn every_builtin_entry_has_a_model() {
        for op in builtin() {
            assert!(model(op.name()).is_some(), "{} 没有语义表", op.name());
        }
        assert!(model("no_such_op").is_none());
    }

    /// const 归属查询与整条注册表一致：在册为真，闭集外为假。
    #[test]
    fn registry_contains_matches_table() {
        for entry in REGISTRY {
            assert!(contains(entry.name), "{} 在册却查不到", entry.name);
        }
        assert!(!contains("no_such_op_f32"));
    }

    /// 测试要的是**量化那条**算子，不是"注册表里的第一条"。
    ///
    /// 按位置取会在注册表从一条变两条的那天静默换人——而两条算子并不等价（稠密 f32 与 int4
    /// 的代价差 8 倍）。名字写在这里，注册表增删都不会让这条测试去验另一个算子。
    fn int4() -> Op {
        builtin_named("gemv_i4").expect("gemv_i4 已注册")
    }

    /// 在 `variants` 里按名字取一个变体表；测试改写声明时定位到具体变体。
    fn variant_mut<'a>(declaration: &'a mut Table, name: &str) -> &'a mut Table {
        declaration
            .array_mut("variants")
            .expect("variants")
            .iter_mut()
            .find_map(|value| {
                let Value::Table(variant) = value else {
                    return None;
                };
                (variant.string("name") == Some(name)).then_some(variant)
            })
            .expect("变体存在")
    }

    #[test]
    fn the_builtin_template_only_uses_declared_primitives() {
        let op = int4();
        let declaration =
            decl::OpDecl::parse(op.declaration(), op.template()).expect("内置声明必须自洽");
        assert_eq!(declaration.name, op.name());
    }

    #[test]
    fn a_signature_with_an_unsupported_buffer_type_is_rejected_at_load() {
        // ABI 的唯一事实源是 `.k` 的签名：把 `scales` 写成 `double *` 必须在装载期就失败，
        // 而不是等 `cc` 报 C 的指针类型不兼容——诊断要带算子名与形参名。
        let op = int4();
        let broken = op
            .template()
            .replace("const float *scales", "double *scales");
        let error = decl::OpDecl::parse(op.declaration(), &broken).expect_err("类型不在闭集内");
        let text = error.to_string();
        assert!(text.contains(op.name()), "{text}");
        assert!(text.contains("scales"), "{text}");
        assert!(text.contains("double"), "{text}");
    }

    #[test]
    fn a_template_primitive_must_be_declared_in_needs() {
        let op = int4();
        let template = format!("{}\n@atomic(y, n, out);\n", op.template());
        let error = decl::OpDecl::parse(op.declaration(), &template)
            .expect_err("模板里的原语不在 needs 里必须报错");
        assert!(error.to_string().contains("atomic"), "{error}");
        assert!(error.to_string().contains(op.name()), "{error}");
    }

    /// 模板传的能力参数必须被 `[needs]` 允许：否则设备按声明验、代码按模板生成。
    #[test]
    fn a_call_argument_must_match_the_declared_need() {
        let op = int4();
        let broken = op.template().replace("sat: off", "sat: on");
        let error =
            decl::OpDecl::parse(op.declaration(), &broken).expect_err("模板与需求矛盾必须报错");
        let text = error.to_string();
        assert!(text.contains(op.name()), "{text}");
        assert!(text.contains("cvt"), "{text}");
        assert!(text.contains("sat"), "{text}");
        assert!(text.contains("on"), "{text}");
    }

    /// 闭集内的能力属性传了却没声明，等于协商永远不检查它。
    #[test]
    fn a_capability_argument_must_be_declared_in_needs() {
        let op = int4();
        let broken = op
            .template()
            .replace("sat: off", "sat: off, accuracy: approx");
        let error =
            decl::OpDecl::parse(op.declaration(), &broken).expect_err("未声明的能力参数必须报错");
        let text = error.to_string();
        assert!(text.contains(op.name()), "{text}");
        assert!(text.contains("cvt"), "{text}");
        assert!(text.contains("accuracy"), "{text}");
    }

    /// 闭集之外的命名参数是片段级提示，不构成能力声明，不能被这条判据误伤。
    #[test]
    fn a_fragment_only_named_argument_stays_legal() {
        let op = int4();
        let extra = op.template().replace("sat: off", "sat: off, along: 32");
        decl::OpDecl::parse(op.declaration(), &extra).expect("片段级命名参数不该被能力判据拒绝");
    }

    #[test]
    fn every_variant_must_cover_the_shared_template() {
        let op = int4();
        // 把 `bar` 从算子级需求挪到只有 `pipe2` 的增量需求里：`direct` 仍然会 emit 出 `@bar`，
        // 所以装载期必须拒绝，而不是等协商选中 `direct` 才在 substitute 里发现片段缺失。
        let mut broken = op.declaration().clone();
        broken.table_mut("needs").expect("needs").remove("bar");
        variant_mut(&mut broken, "pipe2").set("extra_needs.bar", Value::table(vec![]));
        let error =
            decl::OpDecl::parse(&broken, op.template()).expect_err("变体没有覆盖共享模板必须报错");
        assert!(error.to_string().contains("direct"), "{error}");
        assert!(error.to_string().contains("bar"), "{error}");
    }

    /// 判据 1（cutile README §12.2）：首选变体必须把非 MMA 工作盖住，否则该换 S3/S4。
    #[test]
    fn the_preferred_variant_must_hide_its_non_mma_work() {
        let op = int4();
        // 占比降下来链仍然单调（0.10 → 0.05），所以先失败的只会是掩盖判据。
        let mut hidden = op.declaration().clone();
        variant_mut(&mut hidden, "pipe2").set("mma_share", Value::real(0.10));
        variant_mut(&mut hidden, "direct").set("mma_share", Value::real(0.05));
        assert_ne!(
            hidden.at("variants[0].mma_share"),
            op.declaration().at("variants[0].mma_share"),
            "声明里的占比写法变了"
        );
        let error =
            decl::OpDecl::parse(&hidden, op.template()).expect_err("首选变体掩盖不足必须报错");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("首选变体 `pipe2`"), "{error}");
        assert!(error.to_string().contains("45%"), "{error}");
    }

    /// 兜底变体的掩盖账超阈值是事实，不是声明错误：它存在的理由正是"设备缺原语时仍能跑"。
    #[test]
    fn a_falling_back_variant_may_miss_the_masking_bar() {
        let op = int4();
        let op = decl::OpDecl::parse(op.declaration(), op.template()).expect("内置声明应当通过");
        let direct = op.variant("direct").expect("兜底变体");
        let masking = sch::Masking::of(direct.mma_share, &direct.schedule);
        assert!(masking.claims_mma(), "{masking}");
        assert!(!masking.within_limit(), "{masking}");
        assert!(masking.to_string().contains("降级兜底"), "{masking}");
    }
}
