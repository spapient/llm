use cutile::harness::{c_literal, c_type, output_name, output_of};
use diagnostic::{Code, Error, Result};
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use crate::dialect::{Cuda, Dialect, launch_args, length_scalar, readback_bits};

use super::*;

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

/// C 系 host 侧缓冲声明：只读落静态初值、可写开数组。CUDA 与昇腾逐字同源（`h_` 主机数组 +
/// 设备侧 `d_` 指针），Metal 的 ObjC host 不用这套（见 [`crate::metal`]）。
pub(crate) fn host_buffer_decl(param: &Param, values: &Values, ctype: &str, size: usize) -> String {
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
pub(crate) fn cuda_buffer_host(
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
    let mut text = host_buffer_decl(param, values, ctype, size);
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
        .filter(|param| !param.ctype.is_buffer())
        .map(|param| format!("{} {}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let buffers = Cuda::buffers(signature);
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
pub(crate) fn cuda_params(params: &[Param]) -> String {
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
