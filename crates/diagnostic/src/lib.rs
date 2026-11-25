//! L1 基础设施底座：全仓库唯一的错误类型、逐字节游标与人读渲染原语。
//!
//! - [`code`]：错误类别闭集 [`Code`]，其短名与退出码是写进 shell 的稳定契约。
//! - [`error`]：带类别、上下文与同级诊断的 [`Error`]，以及统一别名 [`Result`]。
//! - [`scan`]：带字节位置的逐字节游标（`Cursor`），不解释领域文法。
//! - [`render`]：面板、Markdown 表、单位与列表，只排版、不做判定。
//! - [`buildgen`]：可选的构建期目录扫描与 TOML 读取；各消费方生成强类型静态视图。
//!
//! 本 crate 不持有领域语义或通用运行期数据树。读文件统一走 [`Error::from_io`]，区分
//! “路径不存在”与“存在但读不出来”，避免每个调用点各自猜退出码。
//!
//! # 例子
//!
//! ```
//! use llm_diagnostic::{Code, Error};
//!
//! let error = Error::new(Code::Unsupported, "dp4a 不支持 f32").with("cap::negotiate");
//! assert_eq!(error.code(), Code::Unsupported);
//! assert_eq!(error.ctx(), ["cap::negotiate"]);
//! assert!(error.to_string().contains("unsupported"));
//! ```

#![deny(unsafe_code)]
#![deny(clippy::pedantic)]

pub mod code;
pub mod error;
pub mod render;
pub mod scan;

#[cfg(feature = "build")]
pub mod buildgen;

pub use code::Code;
pub use error::{Error, Result};
