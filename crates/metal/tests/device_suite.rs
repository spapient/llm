//! `metal` 片段层与档案对账的装配根。
//!
//! `Cargo.toml` 是 `autotests = false`，`fragments.rs` 必须由本文件挂起才会被编译；
//! 三张表来自 `build.rs` 的 `$OUT_DIR/tables.rs`，经 `llm_metal` 根导出。

pub use llm_metal::*;

mod fragments;
