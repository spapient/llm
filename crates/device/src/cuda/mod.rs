//! CUDA 后端执行层：按职责拆成共享骨架（[`shared`]）、单源网格族（[`grid`]）、WMMA GEMM
//! （[`gemm`]）、归一化/softmax（[`chunk`]）、分块扫描（[`scan`]）与多段公共装配（[`segments`]）。
//! 对外路径仍为 `llm_device::cuda::*`；内部 helper 只在 crate 内可见，经本文件聚合重导出。

pub(crate) mod chunk;
pub(crate) mod gemm;
pub(crate) mod grid;
pub(crate) mod scan;
pub(crate) mod segments;
pub(crate) mod shared;

pub use chunk::{cuda_rmsnorm, cuda_rmsnorm_driver, cuda_softmax, cuda_softmax_driver};
pub use gemm::gemm_kernel;
pub use grid::{
    cuda_benchmark_driver, cuda_chunk_reduce, cuda_chunk_reduce_driver, cuda_grid_driver,
    cuda_grid_kernel, cuda_harness, cuda_loop_driver, cuda_loop_kernel, cuda_point_driver,
    cuda_quantized_gemv_i4,
};
pub use scan::{cuda_scan, cuda_scan_driver};

// crate 内复用的族入口（旧文件为 `pub(crate)`，不向外部泄漏）。
pub(crate) use chunk::{cuda_chunk_rmsnorm, cuda_chunk_softmax};
pub(crate) use scan::cuda_chunk_scan;
pub use shared::{buf_args, host_buf_decl};
