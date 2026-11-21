//! L1 基础设施底座：全仓库唯一的错误类型、唯一的声明数据树与人读渲染原语。
//!
//! 模块按职责正交拆分，彼此不反向引用：
//!
//! - [`code`]：错误类别闭集 [`Code`]，其短名与退出码是写进 shell 的稳定契约。
//! - [`error`]：带类别、上下文与同级诊断的 [`Error`]，以及统一别名 [`Result`]。
//! - [`data`]：编译期语料在运行期的形状（`Table` / `Value`），**没有解析器**。
//! - [`scan`]：带字节位置的逐字节游标（`Cursor`），只做读字节 / 跳空白 / 吃必需字节，
//!   文法仍在各自 crate 里。
//! - [`render`]：把已经算好的数排成人读文本的原语（面板 / Markdown 表 / 单位 / 列表），
//!   **只排版、不做判定**。
//!
//! **拥有**：`Error`、`Code`、`Result`、`data` 的 `Table` / `Value`、`scan` 的 `Cursor` 与
//! `render` 的 `Panel` / `Grid` / `Align`。**不拥有**：任何领域
//! 的错误码字面量——各层用自己的语义填 `Code`，但类型只有这一个；也不解释任何领域语义
//! （`ops/` 属于 `op`，`devices/` 属于 `prof`，`prim/` 属于 `prim`，设置层属于
//! `cli::settings`），更不认识 TOML：语料由各 crate 的 `build.rs` 在编译期解析成 `data`
//! 的表（ADR 0022），运行期没有任何 TOML 解析器。**依赖**：无（只用 `std`）。**被依赖**：
//! 除 `testkit` 外的全部成员（见 `docs/ARCHITECTURE.md` 的所有权表）。
//!
//! 错误类型、数据树与人读渲染共处一个 crate 的理由是依赖方向而非语义：它们都必须比任何数据层更低，
//! 并且消费者（`cli`）本来就依赖本 crate，再单开一个只装字符串函数的叶只是多一条要登记的边。
//! 读文件统一走 [`Error::from_io`]——“路径不存在”（`missing` → 4）与“存在但读不出来”
//! （`io` → 5）是两类事，又直接决定退出码，不该由每个调用点各猜一遍。
//!
//! # 例子
//!
//! ```
//! use llm_diagnostic::{Code, Error};
//!
//! let error = Error::new(Code::Unsupported, "dp4a 不支持 f32").with("cap::negotiate");
//! assert_eq!(error.code(), Code::Unsupported);
//! assert_eq!(error.context(), ["cap::negotiate"]);
//! assert!(error.to_string().contains("unsupported"));
//! ```

#![deny(unsafe_code)]

pub mod code;
pub mod data;
pub mod error;
pub mod render;
pub mod scan;

#[cfg(feature = "build")]
pub mod buildgen;

pub use code::Code;
pub use error::{Error, Result};
