//! Metal / CUDA / 昇腾设备后端：实现 [`crate::backend::Backend`]（见证装配 + 编译执行回读）。
//! 三族各占一子模块，共享见证装配与回读在 `common`；本文件只保留进程级注册入口与跨模块重导出。

pub(crate) mod ascend;
pub(crate) mod common;
pub(crate) mod cuda;
pub(crate) mod metal;

pub use crate::metal::{compile_run_metal, metal_driver, metal_kernel};

static METAL_BACKEND: metal::MetalBackend = metal::MetalBackend;
static CUDA_BACKEND: cuda::CudaBackend = cuda::CudaBackend;
static ASCEND_BACKEND: ascend::AscendBackend = ascend::AscendBackend;

/// 把 Metal / CUDA / 昇腾后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    crate::backend::register(&METAL_BACKEND);
    crate::backend::register(&CUDA_BACKEND);
    crate::backend::register(&ASCEND_BACKEND);
}
