//! 算子契约：签名、布局、模板词法、代价表达式与语义模型。
//!
//! 这一层只回答"算子是什么"，不回答"怎么算"：执行核在 `gemm` / `conv`，
//! 注册表与声明解析在 `op`。四个 crate 的偏序是
//! `spec` → `gemm` → `conv` → `op`，箭头指向使用者；这里没有任何反向依赖。
//!
//! 拥有：模板的[词法](tpl)（`{{插值}}` 与 `@原语` 的解释规则）、[入口签名](sig)（ABI 的唯一事实源）、
//! [布局闭集](sch)、[代价表达式与求和语义](einsum)、[语义模型](model)（粒度、形状、样本、标量参考
//! 的契约）、[确定性样本](sample)与[算子契约](operator)。
//!
//! 不拥有：`ops/*.toml` 的解析与声明表（`op`）、任何执行核（`gemm` / `conv`）、
//! 设备能力协商与 emit（`cutile`）。代价（`flops` / `bytes`）仍只有一处来源：算子声明 `EXPR`，
//! 由[`einsum`]推出，[`model::Cost`] 只是它的再导出。
//!
//! 被谁依赖：`gemm`、`conv`（执行核投影 [`operator::Operator`] 与 [`model::Model`]）、
//! `op`（注册表把 [`operator::model_of`] 投影成 `REGISTRY` 的一行）。
//!
//! # 例子
//!
//! ```
//! // 契约层不依赖任何执行核：名字比对与类型闭集都在这里，`cargo test -p llm-spec` 就能验。
//! assert!(llm_spec::same_bytes("gemv_i4", "gemv_i4"));
//! assert_eq!(
//!     llm_spec::model::DType::parse_name("f32").map(llm_spec::model::DType::name),
//!     Some("f32"),
//! );
//! ```

#![deny(unsafe_code)]

pub mod einsum;
pub mod epilogue;
pub mod model;
pub mod operator;
pub mod sample;
pub mod sch;
pub mod sig;
pub mod tpl;

pub use model::quant;
pub use sample::{dropout, init, rnd};
pub use sch::layout;
pub use tpl::formula;
pub use tpl::parse;

/// `const` 上下文里逐字节比两个名字（没有可用的 `str` 比较）。
pub const fn same_bytes(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut idx = 0;
    while idx < left.len() {
        if left[idx] != right[idx] {
            return false;
        }
        idx += 1;
    }
    true
}

#[cfg(test)]
#[path = "../tests/support/lib_spec_tests.rs"]
mod spec_tests;
