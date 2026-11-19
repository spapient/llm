//! 生产态 CUDA lowering：把 `.k` 内核按声明式骨架降成 CUDA 源（逐点网格 / 分块归约），以及
//! verify 车道的单线程 CUDA harness。同一条内核体只拼一次，后端差异从 [`super`] 的方言表取；
//! Metal 方言见 [`super::metal`]，本模块只做 CUDA。

use diagnostic::{Code, Error, Result};
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use super::{chunk_reduce, cuda_device, grid_kernel, launch_args, CUDA};
use cutile::harness::{c_literal, c_type, output_name, output_of};

/// 拼出可在 GPU 上真实执行的 CUDA 程序：`.k` 内核转 `__device__`，单线程 `__global__` 包装在
/// `<<<1,1>>>` 里调用，host `main` 负责分配/拷贝/回读——**语义对拍门**，性能另由 roofline 门承载。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn cuda_harness(kernel_source: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let params = signature.params();
    let launch_args = launch_args(sample, params)?;
    let mut text = cuda_prelude(kernel_source, &kernel, sample);
    text.push_str(&cuda_wrapper(&kernel, params, &launch_args));
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_buffers(sample, params)?);
    // 设备侧报错就打印 `cuda:` 一行并以 1 退出。
    text.push_str(&format!(
        "    cutile_launch<<<1, 1>>>({launch_args});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    ));
    text.push_str(&cuda_readback(output, values.len()));
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text.push_str("    return 0;\n}\n");
    Ok(text)
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
        "\n__global__ void cutile_launch({}) {{\n    {kernel}({launch_args});\n}}\n\n",
        cuda_params(params)
    )
}

/// 结果回读：设备 → 主机，再按 `float` 位模式逐元素打印。
fn cuda_readback(output: &str, len: usize) -> String {
    format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   for (int i = 0; i < {len}; ++i) {{\n\
         \x20       union {{ float f; unsigned int u; }} v; v.f = h_{output}[i];\n\
         \x20       printf(\"%u\\n\", v.u);\n\
         \x20   }}\n"
    )
}

/// host 侧的缓冲声明、设备分配，以及只读缓冲的入向拷贝。
fn cuda_host_buffers(sample: &Sample, params: &[Param]) -> Result<String> {
    let mut text = String::new();
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        let Some(buffer) = sample.buffer(&param.name) else {
            return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
        };
        let (name, ctype, size) = (&param.name, c_type(&buffer.values), buffer.values.len());
        if param.mutable {
            text.push_str(&format!("    {ctype} h_{name}[{size}];\n"));
        } else {
            text.push_str(&format!(
                "    static const {ctype} h_{name}[{size}] = {{{}}};\n",
                c_literal(&buffer.values)
            ));
        }
        text.push_str(&format!(
            "    {ctype} *d_{name} = 0;\n    cudaMalloc((void **)&d_{name}, {size} * sizeof({ctype}));\n"
        ));
        if !param.mutable {
            text.push_str(&format!(
                "    cudaMemcpy(d_{name}, h_{name}, {size} * sizeof({ctype}), cudaMemcpyHostToDevice);\n"
            ));
        }
    }
    Ok(text)
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
    let params = signature.params();
    let launch = params
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let mut text = String::from("#include <cuda_runtime.h>\n#include <cstdio>\n");
    text.push_str(device_source);
    if !device_source.ends_with('\n') {
        text.push('\n');
    }
    text.push_str("\nint main(void) {\n");
    text.push_str(&cuda_host_buffers_grid(sample, params, output, output_len)?);
    text.push_str(&format!(
        "    kernel_main<<<{grid}, 1>>>({launch});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    ));
    text.push_str(&cuda_readback(output, output_len));
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// host 侧缓冲声明、设备分配与只读入向拷贝；输出缓冲长度用显式 `output_len`（分块归约的部分
/// 结果数），而非样本里的单元素。
fn cuda_host_buffers_grid(
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

/// 生产态 CUDA 逐点网格内核（[`grid_kernel`] 的 CUDA 方言）。
pub fn cuda_grid_kernel(kernel_source: &str, signature: &Signature) -> Result<String> {
    grid_kernel(kernel_source, signature, &CUDA)
}

/// 生产态 CUDA 分块归约内核（[`chunk_reduce`] 的 CUDA 方言）。
pub fn cuda_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &CUDA)
}
