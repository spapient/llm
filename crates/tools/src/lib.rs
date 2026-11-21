//! 仓库工具：配置驱动的静态质量门与人可读输出。

#![deny(unsafe_code)]

pub mod bench;
pub mod call;
pub mod card;
pub mod ccn;
pub mod docs;
pub mod dup;
pub mod escape;
pub mod gate;
pub mod lex;
pub mod regress;
pub mod stats;
pub mod strip;

/// 构建期生成的词汇表（`build.rs` 读 `lints/vocabulary.toml` 展开成 `Grid` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/vocabulary.rs"));
}

pub use ccn::{Finding, analyze};
pub use gate::Gate;
pub use generated::vocabulary;
pub use lex::{Delim, Pos, Token, parse};
pub use strip::{strip_tests, submodules, test_modules};
