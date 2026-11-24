//! Metal 后端执行层：运行时编译执行（`runtime`）与见证驱动（`witness`）。
//! 对外路径为 `llm-shard::metal::*`；内部 helper 只在 crate 内可见，经本文件聚合重导出。

pub(crate) mod runtime;
pub(crate) mod witness;

pub use runtime::{compile_run_metal, metal_kernel};
pub use witness::metal_driver;

pub(crate) use runtime::run_metal;
