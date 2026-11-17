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
//! 的位宽/方向，由 [`einsum`] 推出；[`model::Cost`] 只是它的再导出（`docs/adr/0007`，并入本 crate 后
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

pub mod add_f32;
pub mod attention;
pub mod decl;
pub mod decode_attention;
pub mod decode_flash_attention;
pub mod dequant_i4;
pub mod dequant_i8;
pub mod einsum;
pub mod elementwise;
pub mod embed_f32;
pub mod flash_attention;
pub mod gelu_f32;
pub mod gemm;
pub mod gemm_f32;
pub mod gemm_i4;
pub mod gemm_i8;
pub mod gemv;
pub mod gemv_f32;
pub mod gemv_i4;
pub mod gemv_i8;
pub mod layout;
pub mod model;
pub mod mul_f32;
pub mod reduce;
pub mod reduce_max_f32;
pub mod reduce_sum_f32;
pub mod residual_add_f32;
pub mod rmsnorm_f32;
pub mod rope_f32;
pub mod scale_f32;
pub mod sch;
pub mod sig;
pub mod silu_f32;
pub mod softmax_f32;
pub mod tpl;
pub mod unary;

use diagnostic::data::Table;

use model::Model;

/// 一次注册：一个算子的三件事——名字、模板、语义表。
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
}

/// 生成编译期内嵌的算子注册表：一行一个算子，`"名字" => 语义模块`。
///
/// 名字同时是 `ops/<name>.toml` 与 `ops/<name>.k` 的文件名（`include_str!` 由它拼出路径），模块是
/// `MODEL` 所在的 `src/<模块>.rs`。两列多数同名，只有 `attention_f32` 与 `flash_attention_f32`
/// 不同名——它们共用 [`attention`] 的全部语义，只换标量参考（见 [`flash_attention`]）。
///
/// 一条注册原本要写五句、把同一份结构逐字抄 22 遍（110 行），而能变的只有这两列。加一个算子
/// 仍然只改本调用里的一行 + `ops/` 两个文件 + 一个模块。
macro_rules! registry {
    ($($name:literal => $module:ident),* $(,)?) => {
        /// 编译期内嵌的算子注册表。
        ///
        /// 新增算子在这里补一行，否则发布出去的二进制里没有它；
        /// `the_registry_covers_the_generated_table_exactly` 把"生成表里有、注册表里没有"（或反过来）
        /// 直接判为失败。数组长度由表本身数出来，不另写一个常数——那是第二个会漂移的数字。
        pub const REGISTRY: [Registration; <[()]>::len(&[$(registry!(@one $name)),*])] = [
            $(
                Registration {
                    name: $name,
                    template: include_str!(concat!("../ops/", $name, ".k")),
                    model: $module::MODEL,
                }
            ),*
        ];
    };
    (@one $name:literal) => { () };
}

registry! {
    "add_f32" => add_f32,
    "attention_f32" => attention,
    "decode_attention_f32" => decode_attention,
    "decode_flash_attention_f32" => decode_flash_attention,
    "dequant_i4" => dequant_i4,
    "dequant_i8" => dequant_i8,
    "embed_f32" => embed_f32,
    "flash_attention_f32" => flash_attention,
    "gelu_f32" => gelu_f32,
    "gemm_f32" => gemm_f32,
    "gemm_i4" => gemm_i4,
    "gemm_i8" => gemm_i8,
    "gemv_f32" => gemv_f32,
    "gemv_i4" => gemv_i4,
    "gemv_i8" => gemv_i8,
    "mul_f32" => mul_f32,
    "reduce_max_f32" => reduce_max_f32,
    "reduce_sum_f32" => reduce_sum_f32,
    "residual_add_f32" => residual_add_f32,
    "rmsnorm_f32" => rmsnorm_f32,
    "rope_f32" => rope_f32,
    "scale_f32" => scale_f32,
    "silu_f32" => silu_f32,
    "softmax_f32" => softmax_f32,
}

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

/// const 上下文里逐字节比两个名字。
const fn same_bytes(left: &str, right: &str) -> bool {
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
mod tests;
