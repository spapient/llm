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

/// host 侧缓冲声明、设备分配与只读入向拷贝；输出缓冲长度用显式 `output_len`（分块归约的部分
/// 结果数），而非样本里的单元素。

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

