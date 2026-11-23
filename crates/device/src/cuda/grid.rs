//! 单源网格族：逐点 / 循环注入内核、合作式 int4 GEMV、串行 harness，以及各自的 host 驱动。
//! 公共 host 装配、缓冲与回读在 [`super::shared`]，这里只换内核壳与派发行。

use crate::dialect::{Cuda, Dialect, f32_output_shape, launch_args, length_scalar};
use cutile::harness::{output_name, output_of};
use diagnostic::{Code, Error, Result};
use prof::decl::Grid;
use prof::model::Sample;
use prof::sig::{Param, Signature};

use super::shared::{
    CUDA_CHECK_PROLOGUE, CUDA_EPILOGUE, Driver, alloc_guards, buf_args, cuda_host_bufs_grid,
    cuda_launch, cuda_launch_params, cuda_prologue, cuda_readback, cuda_release, musa_shims,
    program, timing_loop,
};

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
    let emit = cuda_launch(
        "kernel_main",
        &format!("dim3({}, {}, {})", grid[0], grid[1], grid[2]),
        1,
        &buf_args(signature),
    );
    program(
        device_source,
        signature,
        sample,
        output,
        values.len(),
        &emit,
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

/// 生产态 CUDA/MUSA 基准驱动：每次采样用 `cudaEvent` 记一对时间戳，热身后按 `repeat` 轮
/// 各跑 16 批，`stderr` 每轮打一行 `ns`。
///
/// 四段模板全部在这里之外命名：
///
/// 1. 缓冲区（`cuda_host_bufs_grid` + `alloc_guards`）
/// 2. 计时（`timing_loop`，先热身再按轮计时）
/// 3. 回读（`cuda_readback`，按位打印）
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
    source.push_str(&cuda_readback(&output, out_len, None, true));
    source.push_str(&cuda_release(signature));
    source.push_str(CUDA_EPILOGUE);
    Ok(source)
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

/// 生产态分块归约程序：`blocks` 个线程各归约一块，回读部分结果后由主机折叠成一个输出。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲时返回说明。
pub fn cuda_chunk_reduce_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    blocks: usize,
    fold: &str,
) -> Result<String> {
    let output = output_name(signature)?;
    let emit = cuda_launch("kernel_main", &blocks.to_string(), 1, &buf_args(signature));
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, blocks)
        .folding(fold)
        .assemble(&head, &emit)
}

/// 拼出可在 GPU 上真实执行的 CUDA 网格程序：`kernel_main<<<grid,1>>>` 发射 + host 分配/拷贝/回读。
/// 与 [`cuda_harness`] 同口径，但发射是生产态网格（并行），输出缓冲长度由 `output_len` 显式给出——
/// 分块归约的输出是 `ceil(N/chunk)` 个部分结果，大于签名的单元素 `y`。
pub fn cuda_grid_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: usize,
    output_len: usize,
) -> Result<String> {
    let output = output_name(signature)?;
    let emit = cuda_launch("kernel_main", &grid.to_string(), 1, &buf_args(signature));
    program(device_source, signature, sample, output, output_len, &emit)
}

/// 生产态 CUDA 逐点网格内核（[`Dialect::grid_kernel`] 的 CUDA 方言）。
pub fn cuda_grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    Cuda::grid_kernel(
        kernel_source,
        signature,
        Some(length_scalar(signature, sample)?),
    )
}

/// 逐点算子的 host 驱动：块内 256 线程、网格 `ceil(N/256)`，分配/回读与 [`cuda_grid_driver`]
/// 同口径（输出长度取样本值）。kernel 侧由 [`cuda_grid_kernel`] 的守卫挡住末块多余线程。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲时返回说明。
pub fn cuda_point_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    blocks: usize,
    threads: usize,
) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let emit = cuda_launch(
        "kernel_main",
        &blocks.to_string(),
        threads,
        &buf_args(signature),
    );
    program(
        device_source,
        signature,
        sample,
        output,
        values.len(),
        &emit,
    )
}

/// 拼出可在 GPU 上真实执行的 CUDA 程序：`.k` 内核转 `__device__`，单线程 `__global__` 包装在
/// `<<<1,1>>>` 里调用，host `main` 负责分配/拷贝/回读——**语义对拍门**，性能另由 roofline 门承载。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn cuda_harness(kernel_source: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let params = signature.params();
    let args = launch_args(sample, params)?;
    let head = format!(
        "{}{}",
        cuda_prelude(kernel_source, &kernel, sample),
        cuda_wrapper(&kernel, params, &args)
    );
    let emit = cuda_launch("cutile_launch", "1", 1, &args);
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

/// `__global__` 包装的形参声明：缓冲是设备指针，用 `d_` 前缀与 host 侧 `cuda_host_bufs` 的
/// `cudaMalloc(&d_<name>)` 同源；标量直接传值。
pub(crate) fn cuda_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            let base = param.ctype.c_name();
            if param.ctype.is_buf() {
                format!("{base} *d_{}", param.name)
            } else {
                format!("{base} {}", param.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 文件头：CUDA 头、身份注释，以及整体转成 `__device__` 的内核源码。
pub(crate) fn cuda_prelude(kernel_source: &str, kernel: &str, sample: &Sample) -> String {
    let device = Cuda::prelude(kernel_source, kernel);
    let mut text = format!(
        "#include <cuda_runtime.h>\n#include <cstdio>\n/* cutile cuda harness | kernel={kernel} m={} n={} | 单线程设备执行（语义门，非并行性能门） */\n{device}",
        sample.m, sample.n
    );
    if !device.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// 设备侧包装：单线程 `<<<1,1>>>` 调内核，入口定义只有一处。
pub(crate) fn cuda_wrapper(kernel: &str, params: &[Param], launch_args: &str) -> String {
    format!(
        "\n__global__ void cutile_launch({}) {{\n    {kernel}({launch_args});\n}}\n",
        cuda_params(params)
    )
}
