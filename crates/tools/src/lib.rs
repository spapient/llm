//! 仓库工具：配置驱动的静态质量门与人可读输出。
//!
//! **拥有**：词法分析（[`lex`]）、圈复杂度与函数长度（[`ccn`]）、文件行数（[`dup`]）、
//! 禁止调用形态（[`call`]）、文档漂移（[`docs`]）、测试代码剔除（[`strip`]）、质量门阈值（[`gate`]）、
//! 回退判定（[`regress`]）、统一基准记录（[`mod@bench`]）、实测卡三态判定（[`card`]）、
//! 样本统计（[`stats`]）、JSON 转义（[`escape`]）。
//! **不拥有**：任何运行时领域数据类型、退出码（由调用方映射）、人可读排版（`layout::Panel`）。
//!
//! 模块互不调用，只共享 [`lex`]（唯一语法事实来源）与 [`escape`]（唯一转义事实来源）：
//! 分析器之间没有先后依赖，`gate` 不认任何分析器的类型，判定顺序由调用方给定。
//! **依赖**：`diagnostic`（错误契约与数据树）。
//!
//! 设计原则（参考 burn）：标准是数据（项目 `lints/vocabulary.toml` 的 `[gate]`），分析器是纯函数，
//! 结果可对拍；分析器本身不打印、不退出，门禁是否失败由调用方按 [`Gate`] 判定。
//!
//! 词汇表在**构建期**由 [`build.rs`](../build.rs) 用构建依赖 `toml` 解析 `lints/vocabulary.toml`，
//! 展开成 `diagnostic::data::{Grid, Value}` 字面量写进 `$OUT_DIR/vocabulary.rs`，再由
//! [`vocabulary`] 暴露（ADR 0022）。运行期只遍历这棵树：不读配置文件，也没有任何 TOML 解析器。

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
