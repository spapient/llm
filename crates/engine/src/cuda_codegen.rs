//! CUDA 侧的设备源与主机驱动拼装：`__device__` 翻译单元、`<<<grid, block>>>` 发射、
//! `cudaMalloc` / `cudaMemcpy` / `cudaFree` 骨架与回读。
//!
//! 与 [`crate::codegen`] 的分家：那里只放「方言表 [`crate::codegen::Wrap`] + 两端共用的 lowering
//! 原语」，这里只写 CUDA 一侧的字面量与驱动骨架。两条 lowering 从同一份方言派生，同一条 kernel
//! body 只拼一次；`codegen` 把本模块再导出，所以 `crate::codegen::cuda_*` 这条路径与拆分前一致。
//!
//! # 为什么驱动只有一个 `Driver`
//!
//! 六个驱动形态（语义门、逐点网格、循环注入、分块归约、分块归一化、分块扫描）只在「文件头」与
//! 「派发那一行」上不同；同步报错、回读与释放清单必须逐字同源——各写一份时，加一个 include
//! 就会漏掉其中一族。

use diagnostic::{Code, Error, Result};
use op::decl::Grid;
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use cutile::harness::{c_literal, c_type, output_name, output_of};

use crate::codegen::{
    CUDA, bind_grid_loops, chunk_reduce, cuda_buffers, cuda_device, grid_kernel, launch_args,
    length_scalar, thread_param_typed,
};
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

/// `__global__` 包装的形参声明：缓冲是设备指针，用 `d_` 前缀与 host 侧 `cuda_host_buffers` 的
/// `cudaMalloc(&d_<name>)` 同源；标量直接传值。
fn cuda_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            let base = param.ctype.c_name();
            if param.ctype.is_buffer() {
                format!("{base} *d_{}", param.name)
            } else {
                format!("{base} {}", param.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 文件头：CUDA 头、身份注释，以及整体转成 `__device__` 的内核源码。
fn cuda_prelude(kernel_source: &str, kernel: &str, sample: &Sample) -> String {
    let device = cuda_device(kernel_source, kernel);
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
fn cuda_wrapper(kernel: &str, params: &[Param], launch_args: &str) -> String {
    format!(
        "\n__global__ void cutile_launch({}) {{\n    {kernel}({launch_args});\n}}\n",
        cuda_params(params)
    )
}

/// host 驱动的公共骨架：文件头 → `main` → 缓冲分配 → 一次派发 → 回读 → 释放 → 返回。
///
/// 六个驱动（单线程语义门、逐点网格、循环注入、分块归约、分块归一化、分块扫描）只在「文件头」
/// 与「派发那一行」上不同；同步报错、回读与释放清单必须逐字同源——各写一份时，加一个 include
/// 就会漏掉其中一族。
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
        text.push_str(&cuda_host_buffers_grid(
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

/// 结果回读：设备 → 主机，再按 `float` 位模式逐元素打印。
pub(crate) fn cuda_readback(output: &str, len: usize) -> String {
    format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   for (int i = 0; i < {len}; ++i) {{\n\
         \x20       union {{ float f; unsigned int u; }} v; v.f = h_{output}[i];\n\
         \x20       printf(\"%u\\n\", v.u);\n\
         \x20   }}\n"
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
    let emit = cuda_launch(
        "kernel_main",
        &blocks.to_string(),
        1,
        &buffer_args(signature),
    );
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
    let emit = cuda_launch("kernel_main", &grid.to_string(), 1, &buffer_args(signature));
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, output_len).assemble(&head, &emit)
}

/// host 侧缓冲声明、设备分配与只读入向拷贝；输出缓冲长度用显式 `output_len`（分块归约的部分
/// 结果数），而非样本里的单元素。
pub(crate) fn cuda_host_buffers_grid(
    sample: &Sample,
    params: &[Param],
    output: &str,
    output_len: usize,
) -> Result<String> {
    let mut text = String::new();
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        let Some(buffer) = sample.buffer(&param.name) else {
            return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
        };
        text.push_str(&cuda_buffer_host(param, &buffer.values, output, output_len));
    }
    Ok(text)
}

/// 单个缓冲的 host 声明 + 设备分配 + （只读时）入向拷贝；`output` 与 `output_len` 只用于把输出
/// 缓冲的尺寸换成部分结果数，其余缓冲尺寸取样本原值。
fn cuda_buffer_host(param: &Param, values: &Values, output: &str, output_len: usize) -> String {
    let ctype = c_type(values);
    let size = if param.name == output {
        output_len
    } else {
        values.len()
    };
    let mut text = if param.mutable {
        format!("    {ctype} h_{}[{size}];\n", param.name)
    } else {
        format!(
            "    static const {ctype} h_{}[{size}] = {{{}}};\n",
            param.name,
            c_literal(values)
        )
    };
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

/// 生产态 CUDA 逐点网格内核（`grid_kernel` 的 CUDA 方言）。
pub fn cuda_grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    grid_kernel(
        kernel_source,
        signature,
        &CUDA,
        Some(length_scalar(signature, sample)?),
    )
}

/// `kernel_main` 的完整形参表：缓冲 + 标量（模板里的 `M` / `N` / `K` 是运行时值，循环注入与
/// 串行都要把它们透传给内部内核）。
pub(crate) fn cuda_launch_params(signature: &Signature) -> String {
    let scalars = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
        .map(|param| format!("{} {}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let buffers = cuda_buffers(signature);
    if scalars.is_empty() {
        buffers
    } else {
        format!("{buffers},\n    {scalars}")
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
        &buffer_args(signature),
    );
    let head = cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    );
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

/// 生产态 CUDA 循环注入网格内核（`[grid]` 骨架的 CUDA 方言）：gemv 一维、gemm 二维共用。
///
/// 与 Metal 版同一条文本改写（循环头绑到线程下标、正文跨步原样保留），差异只有三处语法：
/// - 内部内核追加的是 `int3 tid`；
/// - `kernel_main` 里用 `blockIdx` 装配 `tid`；
/// - 设备函数用 `__device__` 而非 `device` 地址空间。
///
/// 空网格（`dims` 为空）退化为串行：不追加 `tid`、不绑循环、缓冲与标量原样透传。
///
/// # Errors：模板里找不到声明的循环、签名与样本不一致时返回说明。
pub fn cuda_loop_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    if grid.dims.is_empty() {
        return cuda_serial_kernel(kernel_source, signature, sample);
    }
    cuda_injected_kernel(kernel_source, signature, sample, grid)
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

/// CUDA 串行（零维网格）：不追加 `tid`，一次调用跑完整条循环。
fn cuda_serial_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = cuda_launch_params(signature);
    Ok(format!(
        "{}\n\n__global__ void kernel_main(\n{buffers}) {{\n    {}({call});\n}}",
        cuda_device(kernel_source, &signature.kernel()),
        signature.kernel(),
    ))
}

/// CUDA 循环注入：内部内核追加 `int3 tid`、绑循环头，`kernel_main` 用 `blockIdx` 装配 `tid` 并透传。
fn cuda_injected_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    let mut body = cuda_device(kernel_source, &signature.kernel());
    body = thread_param_typed(&body, &signature.kernel(), "int3 tid")?;
    body = bind_grid_loops(body, grid)?;
    let call = format!("{}, tid", launch_args(sample, signature.params())?);
    let buffers = cuda_launch_params(signature);
    let init = "    int3 tid; tid.x = blockIdx.x; tid.y = blockIdx.y; tid.z = blockIdx.z;\n";
    Ok(format!(
        "{body}\n\n__global__ void kernel_main(\n{buffers}) {{\n{init}    {}({call});\n}}",
        signature.kernel(),
    ))
}

/// 生产态 CUDA 循环注入的 host 驱动：按 `dims` 组成 `dim3` 网格发射 `kernel_main`，
/// 分配/拷贝/回读与 [`cuda_grid_driver`] 同一口径（输出长度取样本值）。
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
    let args = launch_args(sample, signature.params())?;
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
fn cuda_prologue(device_source: &str) -> String {
    cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    )
}

/// `[grid]` 求出的各维大小 → `dim3` 的三维；缺的维保持 1（`dim3` 的缺省就是 1）。
fn cuda_grid(dims: &[usize]) -> [usize; 3] {
    let mut grid = [1usize; 3];
    for (slot, size) in grid.iter_mut().zip(dims) {
        *slot = *size;
    }
    grid
}

/// 启动实参：只传缓冲（形状是编译期常量，不进内核形参）。
pub(crate) fn buffer_args(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// 生产态 CUDA/MUSA 基准驱动：每次采样用 `cudaEvent` 记一对时间戳，热身后按 `repeat` 轮
/// 各跑 16 批，`stderr` 每轮打一行 `ns`。
///
/// # Errors：缺输出缓冲、输出不是 f32，或缓存参数与样本不一致时返回说明。
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
    let output = output_name(signature)?;
    let output_values = sample
        .buffer(output)
        .ok_or_else(|| Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))?;
    if !matches!(&output_values.values, Values::F32(_)) {
        return Err(Error::new(
            Code::Unsupported,
            "设备基准回读当前要求 f32 输出",
        ));
    }
    let mut source = cuda_prologue(device_source);
    if musa {
        source.push_str(
            "#define cudaEvent_t musaEvent_t\n\
             #define cudaEventCreate musaEventCreate\n\
             #define cudaEventRecord musaEventRecord\n\
             #define cudaEventSynchronize musaEventSynchronize\n\
             #define cudaEventElapsedTime musaEventElapsedTime\n\
             #define cudaEventDestroy musaEventDestroy\n\
             #define cudaGetLastError musaGetLastError\n",
        );
    }
    source.push_str(
        r#"
#define CUTILE_CUDA_CHECK(expr) do { \
    cudaError_t cutile_error = (expr); \
    if (cutile_error != cudaSuccess) { \
        fprintf(stderr, "cuda: %s\n", cudaGetErrorString(cutile_error)); \
        return 1; \
    } \
} while (0)
int main() {
"#,
    );
    source.push_str(&cuda_host_buffers_grid(
        sample,
        signature.params(),
        output,
        output_values.values.len(),
    )?);
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        source.push_str(&format!(
            "    if (!d_{}) {{ fprintf(stderr, \"CUDA 缓冲分配失败: {}\\n\"); return 1; }}\n",
            param.name, param.name
        ));
    }
    source.push_str("    CUTILE_CUDA_CHECK(cudaGetLastError());\n");
    source.push_str(&format!(
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
    ));
    source.push_str(&format!(
        "    CUTILE_CUDA_CHECK(cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost));\n\
         \x20   for (int i = 0; i < {len}; ++i) {{\n\
         \x20       union {{ float f; unsigned int u; }} v; v.f = h_{output}[i];\n\
         \x20       printf(\"%u\\n\", v.u);\n\
         \x20   }}\n",
        len = output_values.values.len()
    ));
    source.push_str(&cuda_release(signature));
    source.push_str(
        "    CUTILE_CUDA_CHECK(cudaEventDestroy(start));\n\
         \x20   CUTILE_CUDA_CHECK(cudaEventDestroy(stop));\n\
         \x20   return 0;\n}\n",
    );
    Ok(source)
}

/// 收尾：释放每个缓冲参数对应的设备端指针（只释放真分配过的，标量是值传递）。
pub(crate) fn cuda_release(signature: &Signature) -> String {
    let mut text = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text
}

/// 生产态 CUDA 分块归约内核（`chunk_reduce` 的 CUDA 方言）。
pub fn cuda_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &CUDA)
}
