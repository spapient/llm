//! Metal / CUDA 设备后端：实现 [`cutile::backend::Backend`]（`Codegen` 出源 + `Driver` 跑源）。
//!
//! 这是 L6 的设备落地：唯一碰 `nvcc` / `xcrun metal` / `metallib` / MSVC `cl.exe` 的地方。
//! 厂商编译器的**探测**在 `crate::toolchain`，本模块只按档案取探测结果、拼命令行、跑源。
//! cutile（L3 机制层）只认 [`cutile::backend::Backend`] 缝，不认识这里的任何厂商符号；本模块经
//! [`install`] 把 Metal / CUDA 后端注册进 cutile 的进程级表，由 L7（`cli`）在启动时调用。
//!
//! 双源说明：Metal 是「设备源（.metal）+ host 驱动（.m）」两段文本，CUDA 是单段 `.cu`；这个差异
//! 收在 [`cutile::backend::Source`] 里，由各自的 [`Driver::compile_run`] 解包，不外泄到机制层。
//!
//! 三条厂商链各自成文件（[`metal`] / [`cuda`] / [`ascend`]）——它们之间没有共享状态，只有基准几何
//! 与见证回读这两段字面共用，因此留在本模块；本模块还剩装配（[`install`]）与三家的原路径再导出。


pub mod ascend;
pub mod cuda;
pub mod metal;

use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::Signature;

// ---------------------------------------------------------------------------
// 基准几何与见证回读：三条厂商链共用，因此不归任何一家
// ---------------------------------------------------------------------------

fn benchmark_output_len(signature: &Signature, sample: &Sample) -> Result<usize> {
    let output = cutile::harness::output_name(signature)?;
    sample
        .buffer(output)
        .map(|buffer| buffer.values.len())
        .ok_or_else(|| Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))
}

fn benchmark_grid(grid: &op::decl::Grid, sample: &Sample) -> Result<[usize; 3]> {
    if grid.dims.is_empty() || grid.dims.len() > 3 {
        return Err(Error::new(
            Code::Unsupported,
            "设备基准要求一到三个明确的网格维度",
        ));
    }
    let mut dims = [1usize; 3];
    for (index, (_, scalar)) in grid.dims.iter().enumerate() {
        dims[index] = grid_extent(scalar, sample)?;
    }
    Ok(dims)
}

/// 一个网格标量就是这一维的长度：必须是正数的样本取值。
fn grid_extent(scalar: &str, sample: &Sample) -> Result<usize> {
    let value = sample
        .scalar(scalar)
        .ok_or_else(|| Error::new(Code::Invalid, format!("网格标量 `{scalar}` 没有样本取值")))?;
    usize::try_from(value)
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or_else(|| Error::new(Code::Invalid, format!("网格标量 `{scalar}` 必须为正数")))
}

/// CUDA 与昇腾的 host 驱动共用这一段字面：同一个 `driver::readback` 在两种产物上必须读出同一种
/// 文本，各写一份时改一处就会漂。
fn witness_readback(max: usize) -> String {
    format!(
        "\x20   for (int t = 0; t < total; ++t) {{\n\
         \x20       for (int i = 0; i < h_counts[t]; ++i) {{\n\
         \x20           union {{ float f; unsigned int u; }} v;\n\
         \x20           v.f = h_out[t * {max} + i];\n\
         \x20           printf(\"%u\\n\", v.u);\n\
         \x20       }}\n\
         \x20       fprintf(stderr, \"witness %d\\n\", h_counts[t]);\n\
         \x20   }}\n"
    )
}

// ---------------------------------------------------------------------------
// 装配
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 装配
// ---------------------------------------------------------------------------

static METAL_BACKEND: MetalBackend = MetalBackend;
static CUDA_BACKEND: CudaBackend = CudaBackend;
static ASCEND_BACKEND: AscendBackend = AscendBackend;

/// 把 Metal / CUDA / 昇腾后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    cutile::backend::register(&METAL_BACKEND);
    cutile::backend::register(&CUDA_BACKEND);
    cutile::backend::register(&ASCEND_BACKEND);
}

#[cfg(test)]

pub(crate) use ascend::AscendBackend;
pub(crate) use cuda::CudaBackend;
pub(crate) use metal::MetalBackend;

pub use crate::metal::host::{compile_run_metal, metal_driver, metal_kernel};
pub use ascend::compile_run_ascend;
pub use cuda::compile_run_cuda;

#[path = "../tests/backends.rs"]
mod tests;
