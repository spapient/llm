//! Metal 后端执行层：host 驱动骨架（`host`）、单段设备源（`device`）、分块段源（`chunk`）、
//! 多段装配（`stage`）、合成程序（`program`）、运行时编译执行（`runtime`）与见证驱动（`witness`）。
//! 对外路径仍为 `llm_device::metal::*`；内部 helper 只在 crate 内可见，经本文件聚合重导出。

pub(crate) mod chunk;
pub(crate) mod device;
pub(crate) mod host;
pub(crate) mod program;
pub(crate) mod runtime;
pub(crate) mod stage;
pub(crate) mod witness;

// 对外公开入口（被 `crate::backends` 与外部测试经 `llm_device::metal::*` 引用）。
pub use device::{metal_grid_kernel, metal_harness, metal_loop_kernel};
pub use host::{metal_benchmark_driver, metal_dispatch_driver};
pub use program::{reduce, rmsnorm, scan, softmax};
pub use runtime::{compile_run_metal, metal_kernel};
pub use witness::metal_driver;

// crate 内复用的族入口（旧文件为 `pub(crate)`，不向外部泄漏）。
pub(crate) use chunk::{metal_chunk_rmsnorm, metal_chunk_scan, metal_chunk_softmax};
pub(crate) use runtime::run_metal;
