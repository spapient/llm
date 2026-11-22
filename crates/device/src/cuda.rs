//! CUDA 侧装配：把一份 CUDA C 内核与它的签名/样本装成可直接 `nvcc` 编译的翻译单元。
//!
//! 只放与 CUDA 有关的模板：内核壳（`*_kernel`）、host 驱动（`*_driver`）、发射与读回。
//! 后端无关的 lowering 在 [`crate::dialect`]，形态分派在 [`crate::program`]。

mod driver;

use crate::dialect::{Cuda, Dialect, f32_output_shape, readback_bits};
use cutile::harness::output_of;
use diagnostic::{Code, Error, Result};
use prof::decl::Grid;
use nn::model::Sample;
use nn::sig::Signature;

pub(crate) use driver::*;
pub use driver::{cuda_grid_driver, cuda_harness};

/// 生产态 CUDA 循环注入网格内核（`[grid]` 骨架的 CUDA 方言）：gemv 一维、gemm 二维共用。
///
/// 与 Metal 版同一条文本改写（循环头绑到线程下标、正文跨步原样保留），差异只在方言
/// [`crate::dialect::Cuda`] 里三处语法：内部内核追加 `int3 tid`、`kernel_main` 用 `blockIdx` 装配
/// `tid`、设备函数用 `__device__`。空网格（`dims` 为空）退化为串行：不追加 `tid`、不绑循环。
///
/// # Errors：模板里找不到声明的循环、签名与样本不一致时返回说明。
pub fn cuda_loop_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    if grid.dims.is_empty() {
        return Cuda::serial_kernel(kernel_source, signature, sample);
    }
    Cuda::injected_kernel(kernel_source, signature, sample, grid)
}

/// 生产态 CUDA 合作式 int4 GEMV 内核：一个 warp（32 lane）协作算一行，K 轴按 lane 跨步，
/// 归约走 `__shfl_down_sync`。当前只落地 `gemv_i4`。
///
/// # Errors：算子不是 `gemv_i4` 或线程数不是 32 时返回说明。
pub fn cuda_quantized_gemv_i4(signature: &Signature, threads: usize) -> Result<String> {
    if signature.op() != "gemv_i4" || threads != 32 {
        return Err(Error::new(
            Code::Unsupported,
            "CUDA 合作式 GEMV 当前仅支持 gemv_i4 与 32-lane warp",
        ));
    }
    let params = cuda_launch_params(signature);
    Ok(format!(
        r#"typedef unsigned char u8;
__global__ void kernel_main({params}) {{
    const int row = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    float acc = 0.0f;
    for (int k = lane; k < M; k += 32) {{
        const unsigned int packed = (unsigned int)d_w[row * (M / 2) + (k / 2)];
        const int code = ((k & 1) == 0 ? (packed & 0x0f) : (packed >> 4)) - 8;
        const float scale = d_scales[row * (M / 32) + (k / 32)];
        acc = fmaf((float)code * scale, d_x[k], acc);
    }}
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (lane == 0) d_y[row] = acc;
}}"#
    ))
}

/// CUDA 串行（零维网格）与循环注入：由 [`Dialect::serial_kernel`] / [`Dialect::injected_kernel`]
/// 以 `Cuda` 方言落地，缓冲声明取 [`Dialect::bufs`]，标量由 [`crate::dialect::launch_args`] 内联进内部内核，
/// `kernel_main` 不再重复声明标量形参。
///
/// 生产态 CUDA 循环注入的 host 驱动：按 `dims` 组成 `dim3` 网格发射 `kernel_main`，
/// 分配/拷贝/回读与 [`cuda_grid_driver`] 同一口径（输出长度取样本值）。
///
/// 实参只传缓冲：`kernel_main` 的形参表由 [`Dialect::tid_kernel_main`] 生成，只有缓冲——
/// 标量在 `tid_kernel_main` 里已按样本值内联进内部内核调用，不再进发射实参。这里若跟着
/// [`crate::dialect::launch_args`] 传全套形参，就会出现「实参 7 个、形参 4 个」的编译错误（`nvcc`/`mcc` 同报
/// `too many arguments`），而这正是真机上才暴露的那类漂移。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲时返回说明。
pub fn cuda_loop_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    dims: &[usize],
) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let grid = cuda_grid(dims);
    let args = buf_args(signature);
    let emit = cuda_launch(
        "kernel_main",
        &format!("dim3({}, {}, {})", grid[0], grid[1], grid[2]),
        1,
        &args,
    );
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

/// 文件头一行 include 段 + 设备源码；设备源码缺尾换行时补一个，免得 `main` 粘在最后一行上。
pub(crate) fn cuda_head(includes: &str, device_source: &str) -> String {
    let mut text = String::from(includes);
    text.push_str(device_source);
    if !device_source.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// CUDA 前导：运行时与 stdio 两行 include + 设备源。
pub(crate) fn cuda_prologue(device_source: &str) -> String {
    cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    )
}

/// `[grid]` 求出的各维大小 → `dim3` 的三维；缺的维保持 1（`dim3` 的缺省就是 1）。
pub(crate) fn cuda_grid(dims: &[usize]) -> [usize; 3] {
    let mut grid = [1usize; 3];
    for (slot, size) in grid.iter_mut().zip(dims) {
        *slot = *size;
    }
    grid
}

/// 启动实参：只传缓冲（形状是编译期常量，不进内核形参）。
pub(crate) fn buf_args(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// 生产态 CUDA/MUSA 基准驱动：每次采样用 `cudaEvent` 记一对时间戳，热身后按 `repeat` 轮
/// 各跑 16 批，`stderr` 每轮打一行 `ns`。
///
/// 四段模板全部在这里之外命名：
///
/// 1. 缓冲区（`cuda_host_bufs_grid` + `alloc_guards`）
/// 2. 计时（`timing_loop`，先热身再按轮计时）
/// 3. 回读（`readback_block`，按位打印）
/// 4. 释放（`cuda_release`）
///
/// 本函数只负责按上述顺序拼装，不含任何格式串。
///
/// # Errors
///
/// 缺输出缓冲、输出不是 f32，或缓存参数与样本不一致时返回说明。
#[allow(clippy::too_many_arguments)]
pub fn cuda_benchmark_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: [usize; 3],
    threads: [usize; 3],
    launch_args: &str,
    repeat: usize,
    musa: bool,
) -> Result<String> {
    let (output, out_len) = f32_output_shape(signature, sample)?;
    let mut source = cuda_prologue(device_source);
    source.push_str(musa_shims(musa));
    source.push_str(CUDA_CHECK_PROLOGUE);
    source.push_str(&cuda_host_bufs_grid(
        sample,
        signature.params(),
        &output,
        out_len,
    )?);
    source.push_str(&alloc_guards(signature));
    source.push_str("    CUTILE_CUDA_CHECK(cudaGetLastError());\n");
    source.push_str(&timing_loop(grid, threads, launch_args, repeat));
    source.push_str(&readback_block(&output, out_len));
    source.push_str(&cuda_release(signature));
    source.push_str(CUDA_EPILOGUE);
    Ok(source)
}

/// 收尾：释放每个缓冲参数对应的设备端指针（只释放真分配过的，标量是值传递）。
pub(crate) fn cuda_release(signature: &Signature) -> String {
    let mut text = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
    {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text
}

/// 生产态 CUDA 分块归约内核（[`Dialect::chunk_reduce`] 的 CUDA 方言）。
pub fn cuda_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    Cuda::chunk_reduce(kernel_source, signature, sample, chunk, reduce_func)
}

/// 生产态驱动共用的错误检查宏与 `main` 起点。
pub(crate) const CUDA_CHECK_PROLOGUE: &str = r#"
#define CUTILE_CUDA_CHECK(expr) do { \
    cudaError_t cutile_error = (expr); \
    if (cutile_error != cudaSuccess) { \
        fprintf(stderr, "cuda: %s\n", cudaGetErrorString(cutile_error)); \
        return 1; \
    } \
} while (0)
int main() {
"#;

/// 生产态驱动的收尾：事件对象销毁 + 正常返回。
pub(crate) const CUDA_EPILOGUE: &str = "    CUTILE_CUDA_CHECK(cudaEventDestroy(start));\n\
     \x20   CUTILE_CUDA_CHECK(cudaEventDestroy(stop));\n\
     \x20   return 0;\n}\n";

/// MUSA 兼容层：把 CUDA event / error API 映射到 MUSA 同名接口；非 MUSA 目标返回空串。
pub(crate) fn musa_shims(musa: bool) -> &'static str {
    if musa {
        "#define cudaEvent_t musaEvent_t\n\
         #define cudaEventCreate musaEventCreate\n\
         #define cudaEventRecord musaEventRecord\n\
         #define cudaEventSynchronize musaEventSynchronize\n\
         #define cudaEventElapsedTime musaEventElapsedTime\n\
         #define cudaEventDestroy musaEventDestroy\n\
         #define cudaGetLastError musaGetLastError\n"
    } else {
        ""
    }
}

/// 每个缓冲参数的分配失败检查：任何一次分配失败都必须立刻返回 1，不能让后续 launch 读到空指针。
pub(crate) fn alloc_guards(signature: &Signature) -> String {
    let mut out = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
    {
        out.push_str(&format!(
            "    if (!d_{}) {{ fprintf(stderr, \"CUDA 缓冲分配失败: {}\\n\"); return 1; }}\n",
            param.name, param.name
        ));
    }
    out
}

/// 回读块：把输出缓冲拷回主机并按位打印（打印循环下沉到 [`readback_bits`]）。
pub(crate) fn readback_block(output: &str, len: usize) -> String {
    format!(
        "    CUTILE_CUDA_CHECK(cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost));\n{}",
        readback_bits(&format!("h_{output}"), len, "    ")
    )
}

/// 计时循环：先热身两轮，再按 `repeat` 轮计时，每轮固定 16 次 launch 后报单次纳秒。
pub(crate) fn timing_loop(
    grid: [usize; 3],
    threads: [usize; 3],
    launch_args: &str,
    repeat: usize,
) -> String {
    format!(
        r#"    const dim3 grid({}, {}, {});
    const dim3 block({}, {}, {});
    cudaEvent_t start, stop;
    CUTILE_CUDA_CHECK(cudaEventCreate(&start));
    CUTILE_CUDA_CHECK(cudaEventCreate(&stop));
    for (int warm = 0; warm < 2; ++warm) {{
        for (int batch = 0; batch < 16; ++batch) kernel_main<<<grid, block>>>({launch_args});
    }}
    CUTILE_CUDA_CHECK(cudaGetLastError());
    CUTILE_CUDA_CHECK(cudaDeviceSynchronize());
    for (int sample = 0; sample < {repeat}; ++sample) {{
        CUTILE_CUDA_CHECK(cudaEventRecord(start));
        for (int batch = 0; batch < 16; ++batch) kernel_main<<<grid, block>>>({launch_args});
        CUTILE_CUDA_CHECK(cudaGetLastError());
        CUTILE_CUDA_CHECK(cudaEventRecord(stop));
        CUTILE_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUTILE_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        if (!(elapsed_ms > 0.0f)) {{ fprintf(stderr, "CUDA event 时间戳不可用\n"); return 2; }}
        fprintf(stderr, "ns %.3f\n", (double)elapsed_ms * 1.0e6 / 16.0);
    }}
"#,
        grid[0], grid[1], grid[2], threads[0], threads[1], threads[2],
    )
}
