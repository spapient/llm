//! CUDA 侧装配：把一份 CUDA C 内核与它的签名/样本装成可直接 `nvcc` 编译的翻译单元。
//!
//! 只放与 CUDA 有关的模板：内核壳（`*_kernel`）、host 驱动（`*_driver`）、发射与读回。
//! 后端无关的 lowering 在 [`crate::dialect`]，形态分派在 [`crate::backends`]。

use crate::dialect::{
    Cuda, Dialect, RMS_EPS, bind_chunk_loop, carry_fold, chunk_head, f32_output_shape,
    inject_chunk_preamble, launch_args, length_scalar, length_usize, readback_bits, readonly_bufs,
    scalar_name, scan_correct_body, thread_param_typed,
};
use cutile::harness::{c_literal, c_type, output_name, output_of};
use diagnostic::{Code, Error, Result};
use nn::model::{Sample, Values};
use nn::sig::{Param, Signature};
use prof::decl::Grid;

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

/// host 驱动的公共骨架：文件头 → `main` → 缓冲分配 → 一次派发 → 回读 → 释放 → 返回。
pub(crate) struct Driver<'a> {
    signature: &'a Signature,
    sample: &'a Sample,
    output: &'a str,
    output_len: usize,
    fold: Option<&'a str>,
}

impl<'a> Driver<'a> {
    /// 由签名、样本与输出缓冲长度（分块归约取部分结果数，其余取样本长度）装配。
    pub(crate) fn new(
        signature: &'a Signature,
        sample: &'a Sample,
        output: &'a str,
        output_len: usize,
    ) -> Self {
        Self {
            signature,
            sample,
            output,
            output_len,
            fold: None,
        }
    }

    /// 主机折叠算子（只有分块归约用）：`max` 取大、其余求和。读到的是 `len` 个部分结果，折叠后仍
    /// 只回读一个值——输出个数与签名声明一致，折叠律与设备侧 `_cutile_<op>_f32` 同源。
    pub(crate) fn folding(mut self, fold: &'a str) -> Self {
        self.fold = Some(fold);
        self
    }

    /// 拼完整程序：`head` 是文件头（include + 身份注释 + 设备源，方言由调用方给），`emit` 是派发那一行。
    pub(crate) fn assemble(&self, head: &str, emit: &str) -> Result<String> {
        let mut text = String::from(head);
        text.push_str("\nint main(void) {\n");
        text.push_str(&cuda_host_bufs_grid(
            self.sample,
            self.signature.params(),
            self.output,
            self.output_len,
        )?);
        text.push_str(emit);
        let readback = match self.fold {
            Some(fold) => cuda_folded_readback(self.output, self.output_len, fold),
            None => cuda_readback(self.output, self.output_len),
        };
        text.push_str(&readback);
        text.push_str(&cuda_release(self.signature));
        text.push_str("    return 0;\n}\n");
        Ok(text)
    }
}

/// 一次派发 + 同步失败即报错：内核名、网格、线程数与实参由调用方给，报错那一行只有这一份。
pub(crate) fn cuda_launch(kernel: &str, grid: &str, threads: usize, args: &str) -> String {
    format!(
        "    {kernel}<<<{grid}, {threads}>>>({args});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    )
}

/// 结果回读：设备 → 主机，再按 `float` 位模式逐元素打印（打印循环下沉到
/// [`readback_bits`]，三端共用同一份 `union` 位级保真代码）。
pub(crate) fn cuda_readback(output: &str, len: usize) -> String {
    format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n{}",
        readback_bits(&format!("h_{output}"), len, "    ")
    )
}

/// 结果回读（主机折叠）：分块归约读到 `len` 个部分结果，按 `fold` 折成一个再打印。
pub(crate) fn cuda_folded_readback(output: &str, len: usize, fold: &str) -> String {
    let (init, combine) = if fold == "max" {
        (
            "-3.402823466e38f",
            format!("if (h_{output}[i] > acc) acc = h_{output}[i];"),
        )
    } else {
        ("0.0f", format!("acc += h_{output}[i];"))
    };
    format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   float acc = {init};\n\
         \x20   for (int i = 0; i < {len}; ++i) {{\n\
         \x20       {combine}\n\
         \x20   }}\n\
         \x20   union {{ float f; unsigned int u; }} v; v.f = acc;\n\
         \x20   printf(\"%u\\n\", v.u);\n"
    )
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
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, output_len).assemble(&head, &emit)
}

/// host 侧缓冲声明、设备分配与只读入向拷贝；输出缓冲长度用显式 `output_len`（分块归约的部分
/// 结果数），而非样本里的单元素。
pub(crate) fn cuda_host_bufs_grid(
    sample: &Sample,
    params: &[Param],
    output: &str,
    output_len: usize,
) -> Result<String> {
    let mut text = String::new();
    for param in params.iter().filter(|param| param.ctype.is_buf()) {
        let Some(buf) = sample.buf(&param.name) else {
            return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
        };
        text.push_str(&cuda_buf_host(param, &buf.values, output, output_len));
    }
    Ok(text)
}

/// C 系 host 侧缓冲声明：只读落静态初值、可写开数组。CUDA 与昇腾逐字同源（`h_` 主机数组 +
/// 设备侧 `d_` 指针），Metal 的 ObjC host 不用这套（见 [`crate::metal`]）。
pub(crate) fn host_buf_decl(param: &Param, values: &Values, ctype: &str, size: usize) -> String {
    if param.mutable {
        format!("    {ctype} h_{}[{size}];\n", param.name)
    } else {
        format!(
            "    static const {ctype} h_{}[{size}] = {{{}}};\n",
            param.name,
            c_literal(values)
        )
    }
}

/// 单个缓冲的 host 声明 + 设备分配 + （只读时）入向拷贝；`output` 与 `output_len` 只用于把输出
/// 缓冲的尺寸换成部分结果数，其余缓冲尺寸取样本原值。
pub(crate) fn cuda_buf_host(
    param: &Param,
    values: &Values,
    output: &str,
    output_len: usize,
) -> String {
    let ctype = c_type(values);
    let size = if param.name == output {
        output_len
    } else {
        values.len()
    };
    let mut text = host_buf_decl(param, values, ctype, size);
    text.push_str(&format!(
        "    {ctype} *d_{} = 0;\n    cudaMalloc((void **)&d_{}, {size} * sizeof({ctype}));\n",
        param.name, param.name
    ));
    if !param.mutable {
        text.push_str(&format!(
            "    cudaMemcpy(d_{}, h_{}, {size} * sizeof({ctype}), cudaMemcpyHostToDevice);\n",
            param.name, param.name
        ));
    }
    text
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

/// `kernel_main` 的完整形参表：缓冲 + 标量（模板里的 `M` / `N` / `K` 是运行时值，循环注入与
/// 串行都要把它们透传给内部内核）。
pub(crate) fn cuda_launch_params(signature: &Signature) -> String {
    let scalars = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buf())
        .map(|param| format!("{} {}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let bufs = Cuda::bufs(signature);
    if scalars.is_empty() {
        bufs
    } else {
        format!("{bufs},\n    {scalars}")
    }
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
    let head = cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    );
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

// ---------------------------------------------------------------------------
// CUDA lowering：声明式骨架 → CUDA 源（逐点网格 / 分块归约）+ verify 单线程 harness。
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 张量核特化的 f32 GEMM（WMMA）CUDA lowering
// ---------------------------------------------------------------------------

/// 输出块边长。
const TILE: usize = 64;
/// 单个累积片段边长。
const FRAG: usize = 32;
/// 收缩轴每层宽度。
const BK: usize = 8;
/// CTA 线程数（wave128）。
const THREADS: usize = 128;
/// 每边片段数（2×2）。
const PER_SIDE: usize = TILE / FRAG;

/// WMMA GEMM 的设备源（`kernel_main` 即入口，网格 `ceil(M/64) × ceil(N/64)`、128 线程）。
///
/// # Errors
///
/// 签名不是「两个只读 f32 缓冲 + 一个输出 + 三个标量」、标量缺样本值、或 `K % 8 != 0` 时返回说明。
pub fn gemm_kernel(signature: &Signature, sample: &Sample) -> Result<String> {
    let (a, b) = input_pair(signature)?;
    let y = output_name(signature)?;
    let (m, n, k) = shape(signature, sample)?;
    require_k(k)?;
    let exact = tiles_exactly(m, n);
    let mut text = String::from(
        "#define __MUSA_QY2_MMA__ 1\n#include <musa_runtime.h>\n#include <cuda_runtime.h>\n#include <mma.h>\n\nusing namespace mtmusa::wmma;\n",
    );
    text.push_str(&format!(
        "\n// 形状特化：M={m} N={n} K={k}，网格 {}×{}，128 线程\n",
        m.div_ceil(TILE),
        n.div_ceil(TILE)
    ));
    text.push_str(&format!(
        "__global__ void __launch_bounds__({THREADS}, 8) kernel_main(const float* __restrict__ d_{a}, const float* __restrict__ d_{b}, float* __restrict__ d_{y}) {{\n"
    ));
    text.push_str(&shared_decls(exact));
    text.push_str(&accumulators());
    text.push_str(&main_loop(a, b, m, n, k, exact));
    text.push_str(&write_out(y, m, n, exact));
    text.push_str("}\n");
    Ok(text)
}

/// 共享内存：两个输入片；非整除时加输出暂存（4 块 32×32）。
fn shared_decls(exact: bool) -> String {
    let mut text =
        format!("    __shared__ float as[{TILE}][{BK}];\n    __shared__ float bs[{TILE}][{BK}];\n");
    if !exact {
        text.push_str(&format!(
            "    __shared__ float cs[{}][{FRAG}][{FRAG}];\n",
            PER_SIDE * PER_SIDE
        ));
    }
    text
}

/// 4 个累积片段并清零。
fn accumulators() -> String {
    let count = PER_SIDE * PER_SIDE;
    format!(
        "    fragment<accumulator, 32, 32, 8, float> acc[{count}];\n    #pragma unroll\n    for (int u = 0; u < {count}; ++u) fill_fragment(acc[u], 0.0f);\n"
    )
}

/// 主循环：载入一片 A/B → 同步 → 4 次 `mma_sync` → 同步。
fn main_loop(a: &str, b: &str, m: usize, n: usize, k: usize, exact: bool) -> String {
    format!(
        "    for (int k0 = 0; k0 < {k}; k0 += {BK}) {{\n{}{}        __syncthreads();\n{}        __syncthreads();\n    }}\n",
        load_a(a, m, k, exact),
        load_b(b, n, exact),
        mma(),
    )
}

/// A 片载入：每线程一个 `float4`（`64×8` 元素 = 128 个），非整除时按行守卫。
fn load_a(a: &str, m: usize, k: usize, exact: bool) -> String {
    let base = format!("((int)blockIdx.y * {TILE} + row) * {k} + k0 + v * 4");
    let fetch = if exact {
        format!("*(const float4*)(d_{a} + {base})")
    } else {
        format!(
            "(((int)blockIdx.y * {TILE} + row) < {m}) ? *(const float4*)(d_{a} + {base}) : make_float4(0.0f, 0.0f, 0.0f, 0.0f)"
        )
    };
    format!(
        "        {{ int row = threadIdx.x >> 1, v = threadIdx.x & 1; float4 val = {fetch};\n\
         \x20         as[row][v*4+0] = val.x; as[row][v*4+1] = val.y; as[row][v*4+2] = val.z; as[row][v*4+3] = val.w; }}\n"
    )
}

/// B 片载入：转置存 `bs[n][k]`；整除时 `float4` 读，否则逐元素按列守卫。
fn load_b(b: &str, n: usize, exact: bool) -> String {
    let body = if exact {
        format!(
            "float4 val = *(const float4*)(d_{b} + (k0 + kk) * {n} + (int)blockIdx.x * {TILE} + n0);\n\
             \x20           bs[n0+0][kk] = val.x; bs[n0+1][kk] = val.y; bs[n0+2][kk] = val.z; bs[n0+3][kk] = val.w;"
        )
    } else {
        format!(
            "#pragma unroll\n            for (int t = 0; t < 4; ++t) {{\n\
             \x20               int col = (int)blockIdx.x * {TILE} + n0 + t;\n\
             \x20               bs[n0+t][kk] = (col < {n}) ? d_{b}[(k0 + kk) * {n} + col] : 0.0f;\n            }}"
        )
    };
    format!(
        "        {{ int kk = threadIdx.x >> 4, n0 = (threadIdx.x & 15) * 4;\n          {body}\n        }}\n"
    )
}

/// 4 次 `mma_sync`：`acc[i*2+j] += fa[i] × fb[j]`。
fn mma() -> String {
    format!(
        "        fragment<matrix_a, 32, 32, 8, precision::tf32, row_major> fa[{PER_SIDE}];\n\
         \x20       fragment<matrix_b, 32, 32, 8, precision::tf32, col_major> fb[{PER_SIDE}];\n\
         \x20       #pragma unroll\n        for (int i = 0; i < {PER_SIDE}; ++i) load_matrix_sync(fa[i], &as[i * {FRAG}][0], {BK});\n\
         \x20       #pragma unroll\n        for (int j = 0; j < {PER_SIDE}; ++j) load_matrix_sync(fb[j], &bs[j * {FRAG}][0], {BK});\n\
         \x20       #pragma unroll\n        for (int i = 0; i < {PER_SIDE}; ++i)\n\
         \x20           #pragma unroll\n            for (int j = 0; j < {PER_SIDE}; ++j)\n\
         \x20               mma_sync(acc[i * {PER_SIDE} + j], fa[i], fb[j], acc[i * {PER_SIDE} + j]);\n"
    )
}

/// 写出：整除时直接 `store_matrix_sync`；否则先落共享内存，再按行列守卫拷回。
fn write_out(y: &str, m: usize, n: usize, exact: bool) -> String {
    let target = if exact {
        format!(
            "d_{y} + ((int)blockIdx.y * {TILE} + i * {FRAG}) * {n} + (int)blockIdx.x * {TILE} + j * {FRAG}"
        )
    } else {
        format!("&cs[i * {PER_SIDE} + j][0][0]")
    };
    let ldm = if exact {
        n.to_string()
    } else {
        FRAG.to_string()
    };
    let stores = format!(
        "    #pragma unroll\n    for (int i = 0; i < {PER_SIDE}; ++i)\n        #pragma unroll\n        for (int j = 0; j < {PER_SIDE}; ++j)\n            store_matrix_sync({target}, acc[i * {PER_SIDE} + j], {ldm}, mem_row_major);\n"
    );
    if exact {
        return stores;
    }
    stores
        + &format!(
            "    __syncthreads();\n    for (int idx = threadIdx.x; idx < {}; idx += {THREADS}) {{\n\
             \x20       int blk = idx / {block};\n\
             \x20       int r = (idx / {FRAG}) % {FRAG}, c = idx % {FRAG};\n\
             \x20       int gr = (int)blockIdx.y * {TILE} + (blk / {PER_SIDE}) * {FRAG} + r;\n\
             \x20       int gc = (int)blockIdx.x * {TILE} + (blk % {PER_SIDE}) * {FRAG} + c;\n\
             \x20       if (gr < {m} && gc < {n}) d_{y}[gr * {n} + gc] = cs[blk][r][c];\n    }}\n",
            PER_SIDE * PER_SIDE * FRAG * FRAG,
            block = FRAG * FRAG,
        )
}

/// 两个只读输入缓冲名（A、B）。
fn input_pair(signature: &Signature) -> Result<(&str, &str)> {
    let reads: Vec<&str> = signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf() && !param.mutable)
        .map(|param| param.name.as_str())
        .collect();
    match reads.as_slice() {
        [a, b] => Ok((a, b)),
        _ => Err(Error::new(
            Code::Invalid,
            format!("WMMA GEMM 需要两个只读缓冲（A, B），得到 {}", reads.len()),
        )
        .with("engine::cuda::input_pair")),
    }
}

/// `K` 必须是 `BK` 的正整数倍：否则一片不齐，由调用方回落通用路径。
fn require_k(k: usize) -> Result<()> {
    if k != 0 && k.is_multiple_of(BK) {
        return Ok(());
    }
    Err(Error::new(
        Code::Unsupported,
        format!("WMMA GEMM 要求 K 是 {BK} 的正整数倍，得到 {k}"),
    )
    .with("engine::cuda::gemm_kernel"))
}

/// `M`/`N` 都整除块边长时载入与写出都不带守卫（编译器把分支整体消掉）。
fn tiles_exactly(m: usize, n: usize) -> bool {
    m.is_multiple_of(TILE) && n.is_multiple_of(TILE)
}

/// 形状 `(M, N, K)`：三个标量的样本值（`gemm` 签名的 `M`/`N`/`K`）。
fn shape(signature: &Signature, sample: &Sample) -> Result<(usize, usize, usize)> {
    let mut values = Vec::with_capacity(3);
    for param in signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buf())
    {
        values.push(scalar_of(param, sample)?);
    }
    match values.as_slice() {
        [m, n, k] => Ok((*m, *n, *k)),
        _ => Err(Error::new(
            Code::Invalid,
            format!("WMMA GEMM 需要三个标量（M, N, K），得到 {}", values.len()),
        )
        .with("engine::cuda::shape")),
    }
}

/// 单个形状标量的样本值：缺取值或不是非负 `usize` 都在这里报，不留给几何算式去越界。
fn scalar_of(param: &Param, sample: &Sample) -> Result<usize> {
    let value = sample.scalar(&param.name).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("WMMA GEMM 的标量 `{}` 没有样本取值", param.name),
        )
        .with("engine::cuda::shape")
    })?;
    usize::try_from(value).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("WMMA GEMM 的标量 `{}` = {value} 不是非负 usize", param.name),
        )
        .with("engine::cuda::shape")
    })
}

// ---------------------------------------------------------------------------
// 分块归一化与扫描 lowering：RMSNorm 两段、Softmax 三段、扫描两段，以及它们的 host 串联驱动
// ---------------------------------------------------------------------------

/// 部分归约 / 部分指数和 / 归一化三段入口：与 Metal 侧同源，直接复用 [`crate::dialect`] 的常量。
pub(crate) use crate::dialect::{EXPSUM, NORM, PARTIAL};

/// 分块 RMSNorm 的设备段：`(部分平方和, 归一化)`。第一段写 `y[g] = Σx²` 的部分和，
/// 主机折叠出 `rms` 后第二段按块回写 `y = gamma * x / rms`。
///
/// # Errors
///
/// 签名不是「两个只读缓冲 + 一个输出 + 一个标量」的归一化形态时返回 `Invalid`。
pub(crate) fn cuda_chunk_rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<Vec<String>> {
    let partial = Cuda::chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        "_cutile_sumsq_f32",
        PARTIAL,
    )?;
    let norm = rmsnorm_map(signature, sample)?;
    Ok(vec![partial, norm])
}

/// 分块 Softmax 的设备段：`(部分最大, 部分指数和, 归一化)`。
///
/// # Errors
///
/// 签名不是「一个只读缓冲 + 一个输出 + 一个标量」的 softmax 形态时返回 `Invalid`。
pub(crate) fn cuda_chunk_softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<Vec<String>> {
    let max = Cuda::chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        "_cutile_max_f32",
        PARTIAL,
    )?;
    let expsum = softmax_expsum_map(signature, sample, chunk)?;
    let norm = softmax_norm_map(signature, sample)?;
    Ok(vec![max, expsum, norm])
}

/// RMSNorm 归一化段：每线程一块，`y[start + j] = gamma[start + j] * x[start + j] / rms`。
fn rmsnorm_map(signature: &Signature, sample: &Sample) -> Result<String> {
    let len = length_scalar(signature, sample)?;
    let reads = readonly_bufs(signature);
    let (Some(x), Some(gamma)) = (reads.first(), reads.get(1)) else {
        return Err(Error::new(
            Code::Invalid,
            "RMSNorm 需要两个只读缓冲（x, gamma）".to_owned(),
        )
        .with("engine::cuda::rmsnorm_map"));
    };
    let y = output_name(signature)?;
    let body = format!(
        "    if (n < {len}) {{\n        d_{y}[n] = d_{gamma}[n] * d_{x}[n] / rms;\n    }}",
        len = len,
        x = x,
        gamma = gamma,
        y = y,
    );
    Ok(point_entry(signature, NORM, "float rms", &body))
}

/// Softmax 第二段：`y[g] = Σ exp(x - max)` 的部分和（`max` 由主机折叠传入）。
fn softmax_expsum_map(signature: &Signature, sample: &Sample, chunk: usize) -> Result<String> {
    let len = length_scalar(signature, sample)?;
    let x = single_read(signature, "Softmax")?;
    let y = output_name(signature)?;
    let body = format!(
        "{head}    d_{y}[g] = _cutile_exp_sum_f32(d_{x} + start, len, 1, max);",
        head = chunk_head(chunk, len, "    "),
        x = x,
        y = y,
    );
    Ok(map_entry(signature, EXPSUM, "float max", &body))
}

/// Softmax 第三段：`y[start + j] = exp(x - max) / sum`。
fn softmax_norm_map(signature: &Signature, sample: &Sample) -> Result<String> {
    let len = length_scalar(signature, sample)?;
    let x = single_read(signature, "Softmax")?;
    let y = output_name(signature)?;
    let body = format!(
        "    if (n < {len}) {{\n        d_{y}[n] = _cutile_exp_f32(d_{x}[n] - max) / sum;\n    }}",
        len = len,
        x = x,
        y = y,
    );
    Ok(point_entry(signature, NORM, "float max, float sum", &body))
}

/// 一个分块 map 入口：签名缓冲形参 + 额外标量（`rms` / `max` / `sum`），入口名由调用方给。
fn map_entry(signature: &Signature, entry: &str, extra: &str, body: &str) -> String {
    let bufs = Cuda::bufs(signature);
    let bufs = if extra.is_empty() {
        bufs
    } else {
        format!("{bufs},\n    {extra}")
    };
    Cuda::kernel_entry("", "", &bufs, ("", Cuda::GRID_STMT), body, entry)
}

/// 一个逐元素网格入口：每线程一个元素，线程下标符号是 `n`（[`Cuda`] 的 [`Dialect::POINT_STMT`]）。
fn point_entry(signature: &Signature, entry: &str, extra: &str, body: &str) -> String {
    let bufs = Cuda::bufs(signature);
    let bufs = if extra.is_empty() {
        bufs
    } else {
        format!("{bufs},\n    {extra}")
    };
    Cuda::kernel_entry("", "", &bufs, ("", Cuda::POINT_STMT), body, entry)
}

/// 唯一只读缓冲名（Softmax 形态）；缺或多都返回 `Invalid`。
fn single_read<'a>(signature: &'a Signature, family: &str) -> Result<&'a str> {
    let reads = readonly_bufs(signature);
    match reads.as_slice() {
        [only] => Ok(*only),
        _ => Err(Error::new(
            Code::Invalid,
            format!("{family} 需要恰好一个只读缓冲，得到 {}", reads.len()),
        )
        .with("engine::cuda")),
    }
}

/// 分块归一化驱动骨架：多段设备源（仅首段带 preamble）→ host 串联程序的几何、主机缓冲、
/// 部分结果脚手架与收尾（回读、释放、`return 0`）只有这一份。
fn driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce: impl FnOnce(&mut String, Stage<'_>),
) -> Result<String> {
    let len = length_usize(signature, sample)?;
    let grid = len.div_ceil(chunk);
    let output = output_name(signature)?;
    let args = buf_args(signature);
    let mut text = program_head(segments);
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_bufs_grid(
        sample,
        signature.params(),
        output,
        len,
    )?);
    text.push_str(&partial_scaffold(grid));
    reduce(
        &mut text,
        Stage {
            len,
            grid,
            output,
            args: &args,
        },
    );
    text.push_str(&cuda_readback(output, len));
    text.push_str(&cuda_release(signature));
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// 驱动中段拿到的几何与实参：`len` 是归约长度，`grid` 是块数，`output` 是输出缓冲名，
/// `args` 是签名的缓冲实参（追加标量由各段自己给 `extra`）。
struct Stage<'a> {
    len: usize,
    grid: usize,
    output: &'a str,
    args: &'a str,
}

/// 一段块级归约的派发与回读：`PARTIAL` 段在块内归约后写进输出缓冲，再拷回主机供下一段折叠。
fn partial_pass(text: &mut String, stage: &Stage<'_>) {
    text.push_str(&launch(PARTIAL, stage.grid, stage.args, ""));
    text.push_str(&partial_copy(stage.output, stage.grid));
}

/// 末段归一化：每元素一线程，标度由前几段在主机折叠得到；`extra` 是带进内核的标量名。
fn normalized_pass(text: &mut String, stage: &Stage<'_>, extra: &str) {
    text.push_str(&launch_threads(
        NORM,
        stage.len.div_ceil(256),
        256,
        stage.args,
        extra,
    ));
}

/// 分块 RMSNorm 的完整 CUDA 程序：段与驱动是同一形态的两半，调用方只调这一个入口。
///
/// # Errors
///
/// 段的生成与驱动的折叠按同一组几何判定，任一步不符即返回（不会先签段再报驱动）。
pub fn cuda_rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    let segments = cuda_chunk_rmsnorm(kernel_source, signature, sample, chunk)?;
    cuda_rmsnorm_driver(&segments, signature, sample, chunk)
}

/// 分块 Softmax 的完整 CUDA 程序：三趟段与主机折叠同批给出，不在这里另拼一份段数。
///
/// # Errors
///
/// 同 [`cuda_rmsnorm`]：哪一步不合几何就在那一步返回。
pub fn cuda_softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    let segments = cuda_chunk_softmax(kernel_source, signature, sample, chunk)?;
    cuda_softmax_driver(&segments, signature, sample, chunk)
}

/// 分块归一化的完整 CUDA 程序：多段设备源（仅首段带 preamble）+ host 串联驱动。
pub fn cuda_rmsnorm_driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    driver(segments, signature, sample, chunk, |text, stage| {
        partial_pass(text, &stage);
        text.push_str(&fold_sum());
        text.push_str(&format!(
            "    float rms = sqrtf(sum / (float){len} + {eps}f);\n",
            len = stage.len,
            eps = RMS_EPS
        ));
        normalized_pass(text, &stage, "rms");
    })
}

/// 分块 Softmax 的完整 CUDA 程序：部分最大 → 折叠 max → 部分指数和 → 折叠 sum → 归一化。
pub fn cuda_softmax_driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    driver(segments, signature, sample, chunk, |text, stage| {
        partial_pass(text, &stage);
        text.push_str(&fold_max());
        text.push_str(&launch(EXPSUM, stage.grid, stage.args, "max"));
        partial_pass(text, &stage);
        text.push_str(&fold_sum());
        normalized_pass(text, &stage, "max, sum");
    })
}

/// 程序头：CUDA include、错误检查宏、多段设备源。
pub(crate) fn program_head(segments: &[String]) -> String {
    let mut text = String::from(
        "#include <cuda_runtime.h>\n#include <cstdio>\n\
         #define CUTILE_CUDA_CHECK() do { cudaError_t error = cudaDeviceSynchronize(); \
         if (error != cudaSuccess) { fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; } } while (0)\n",
    );
    for (idx, segment) in segments.iter().enumerate() {
        let piece = if idx == 0 {
            segment.as_str()
        } else {
            entry_of(segment)
        };
        text.push_str(piece);
        text.push_str("\n\n");
    }
    text
}

/// 后续段的入口部分：从段内最后一个 `__global__` 起截断（preamble 与模板函数只在首段保留）。
fn entry_of(segment: &str) -> &str {
    const MARK: &str = "\n\n__global__ void ";
    segment.rfind(MARK).map_or(segment, |at| &segment[at + 2..])
}

/// host 侧部分结果的脚手架：网格数 `G` 与回读数组。
fn partial_scaffold(grid: usize) -> String {
    format!("    int G = {grid};\n    float h_tmp[{grid}];\n")
}

/// 一次派发 + 同步检查；`extra` 是追加的标量实参（`rms` / `max` / `sum`），空则不追加。
/// （分块段每线程一块，块大小固定为 1。）
pub(crate) fn launch(entry: &str, grid: usize, args: &str, extra: &str) -> String {
    launch_threads(entry, grid, 1, args, extra)
}

/// 带块内线程数的派发（归一化段逐元素，块内 256 线程；kernel 侧自带 `n < N` 守卫）。
pub(crate) fn launch_threads(
    entry: &str,
    blocks: usize,
    threads: usize,
    args: &str,
    extra: &str,
) -> String {
    let extra = if extra.is_empty() {
        String::new()
    } else {
        format!(", {extra}")
    };
    format!("    {entry}<<<{blocks}, {threads}>>>({args}{extra});\n    CUTILE_CUDA_CHECK();\n")
}

/// 把 `y` 前 `G` 个槽的部分结果回读到 `h_tmp`。
pub(crate) fn partial_copy(output: &str, grid: usize) -> String {
    format!("    cudaMemcpy(h_tmp, d_{output}, {grid} * sizeof(float), cudaMemcpyDeviceToHost);\n")
}

/// 主机折叠部分最大值；初值与 `.k` 模板同口径（f32 最小正规数的负数）。
fn fold_max() -> String {
    String::from(
        "    float max = -3.402823466e38f;\n    for (int i = 0; i < G; ++i) { if (h_tmp[i] > max) max = h_tmp[i]; }\n",
    )
}

/// 主机折叠部分指数和。
fn fold_sum() -> String {
    String::from("    float sum = 0.0f;\n    for (int i = 0; i < G; ++i) { sum += h_tmp[i]; }\n")
}

// ---------------------------------------------------------------------------
// 分块扫描 lowering：局部扫描 + 主机携带前缀 + 校正，以及 host 串联驱动
// ---------------------------------------------------------------------------

/// 校正段入口（把携带前缀的衰减贡献加回块内）。
pub(crate) const CORRECT: &str = "kernel_correct";

/// CUDA 分块扫描的设备段：`(局部扫描, 校正)`。
///
/// 局部扫描把模板的时间轴循环改写到块内 `[start, start+len)` 并从 0 起算；校正在每块上叠加
/// `C_g · a^(j+1)`（就地累加）。
///
/// # Errors
///
/// 模板里找不到以长度标量为上界的循环、或签名缺标量时返回说明。
pub(crate) fn cuda_chunk_scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<Vec<String>> {
    let scalar = scalar_name(signature)?;
    let len = length_scalar(signature, sample)?;
    let local = scan_local(kernel_source, signature, sample, chunk, scalar)?;
    let correct = scan_correct(output_name(signature)?, chunk, len, decay);
    Ok(vec![local, correct])
}

/// 局部扫描段：设备源补 `__device__`、追加 `int3 tid`、注入块头、把时间轴循环绑定到块内，
/// `kernel_main` 用 `blockIdx` 装配 `tid` 并透传（签名缓冲与标量都原样透传）。
fn scan_local(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    scalar: &str,
) -> Result<String> {
    let kernel = signature.kernel();
    let mut body = Cuda::prelude(kernel_source, &kernel);
    body = thread_param_typed(&body, &kernel, "int3 tid")?;
    body = inject_chunk_preamble(&body, &kernel, chunk, scalar)?;
    body = bind_chunk_loop(&body, scalar)?;
    let call = format!("{}, tid", launch_args(sample, signature.params())?);
    let bufs = cuda_launch_params(signature);
    let init = "    int3 tid; tid.x = blockIdx.x; tid.y = blockIdx.y; tid.z = blockIdx.z;\n";
    Ok(format!(
        "{body}\n\n__global__ void kernel_main(\n{bufs}) {{\n{init}    {kernel}({call});\n}}"
    ))
}

/// 校正段：每线程 `g` 把 `C_g · a^(j+1)` 加回块内每个元素（就地累加），块头与归一化 map 段
/// 同一口径（`g`/`start`/`rest`/`len`）。
fn scan_correct(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    let head = chunk_head(chunk, len, "    ");
    let body = scan_correct_body(output, decay, "    ");
    format!(
        "__global__ void {CORRECT}(\n    const float* d_c,\n    float* d_{output}) {{\n    int g = blockIdx.x * blockDim.x + threadIdx.x;\n{head}{body}\n}}",
        head = head,
        body = body,
        output = output,
    )
}

/// 分块扫描的完整 CUDA 程序（段 + 驱动一次给全）：调用方不必知道这里是两段。
///
/// # Errors
///
/// 段生成的几何判定与驱动折叠同口径，任一步不符即返回。
pub fn cuda_scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<String> {
    let segments = cuda_chunk_scan(kernel_source, signature, sample, chunk, decay)?;
    cuda_scan_driver(&segments, signature, sample, chunk, decay)
}

/// 分块扫描的完整 CUDA 程序：局部扫描 → 回读整段 → 主机折叠携带前缀 → 校正 → 回读输出。
pub fn cuda_scan_driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<String> {
    let len = length_usize(signature, sample)?;
    let grid = len.div_ceil(chunk);
    let output = output_name(signature)?;
    let args = launch_args(sample, signature.params())?;
    let mut text = program_head(segments);
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_bufs_grid(
        sample,
        signature.params(),
        output,
        len,
    )?);
    text.push_str(&format!(
        "    int G = {grid};\n    float *d_c = 0;\n    cudaMalloc((void **)&d_c, G * sizeof(float));\n    float h_c[{grid}];\n"
    ));
    text.push_str(&launch("kernel_main", grid, &args, ""));
    text.push_str(&format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    ));
    text.push_str(&fold_carry(output, chunk, len as i64, decay));
    text.push_str("    cudaMemcpy(d_c, h_c, G * sizeof(float), cudaMemcpyHostToDevice);\n");
    text.push_str(&format!(
        "    {CORRECT}<<<G, 1>>>(d_c, d_{output});\n    CUTILE_CUDA_CHECK();\n"
    ));
    text.push_str(&cuda_readback(output, len));
    text.push_str("    cudaFree(d_c);\n");
    text.push_str(&cuda_release(signature));
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// 主机折叠（f64）：`C_{g+1} = a^len · C_g + y[start+len-1]`；`a^len` 用逐次乘法。
fn fold_carry(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    carry_fold(
        &format!("h_{output}"),
        "h_c",
        "G",
        chunk,
        len,
        decay,
        "    ",
    )
}
