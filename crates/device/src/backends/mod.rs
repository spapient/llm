//! Metal / CUDA / 昇腾设备后端：实现 [`llm_optim::backend::Backend`]（`Codegen` 出源 + `Driver` 跑源）。
//! 三族各占一子模块，共享几何与回读在 [`common`]，`Launch` 并行装配在 [`parallel`]；本文件只保留
//! 进程级注册入口与跨模块重导出，对外路径仍为 `llm_device::backends::*`。

pub(crate) mod ascend;
pub(crate) mod common;
pub(crate) mod cuda;
pub(crate) mod harness;
pub(crate) mod metal;
pub(crate) mod parallel;

// 测试经 `use super::*` 取用的裸名：旧巨石文件顶部的 crate 内重导出面，拆分后由本文件聚合保留。
// 仅 `#[cfg(test)]` 编译：lib 代码走子模块路径，这些别名不参与生产导入面。
#[cfg(test)]
mod test_imports {
    pub(crate) use super::metal::MetalBackend;
    pub(crate) use super::parallel::{cuda, fold_of, metal, unsupported};
    pub(crate) use crate::platform::{ascend_device, detect_cce};
    pub(crate) use diagnostic::Code;
    pub(crate) use llm_optim::backend::{Driver, Launch, Lower, Source};
    pub(crate) use prof::model::Sample;
    pub(crate) use prof::sig::Signature;
}
#[cfg(test)]
pub(crate) use test_imports::*;

pub use crate::metal::{compile_run_metal, metal_driver, metal_kernel};

static METAL_BACKEND: metal::MetalBackend = metal::MetalBackend;
static CUDA_BACKEND: cuda::CudaBackend = cuda::CudaBackend;
static ASCEND_BACKEND: ascend::AscendBackend = ascend::AscendBackend;

/// 把 Metal / CUDA / 昇腾后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    llm_optim::backend::register(&METAL_BACKEND);
    llm_optim::backend::register(&CUDA_BACKEND);
    llm_optim::backend::register(&ASCEND_BACKEND);
}

#[cfg(test)]
#[path = "../../tests/backends.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../testkit/tests/kernel.rs"]
mod cutile_x_kernel;

#[cfg(test)]
#[path = "../../tests/program.rs"]
mod program_tests;
