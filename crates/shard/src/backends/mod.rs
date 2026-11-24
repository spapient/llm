//! 设备后端注册：Metal / CUDA / 昇腾 / Intel OpenCL 共用见证契约。
//! 各族实现独立，公共装配与回读字面在 `common`，本文件只保留进程级注册入口。

pub(crate) mod ascend;
pub(crate) mod common;
pub(crate) mod cuda;
pub(crate) mod metal;
#[cfg(windows)]
pub(crate) mod opencl;

pub use crate::metal::{compile_run_metal, metal_driver, metal_kernel};

static METAL_BACKEND: metal::MetalBackend = metal::MetalBackend;
static CUDA_BACKEND: cuda::CudaBackend = cuda::CudaBackend;
static ASCEND_BACKEND: ascend::AscendBackend = ascend::AscendBackend;
#[cfg(windows)]
static OPENCL_BACKEND: opencl::OpenClBackend = opencl::OpenClBackend;

/// 把全部设备后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    crate::backend::register(&METAL_BACKEND);
    crate::backend::register(&CUDA_BACKEND);
    crate::backend::register(&ASCEND_BACKEND);
    #[cfg(windows)]
    crate::backend::register(&OPENCL_BACKEND);
}
