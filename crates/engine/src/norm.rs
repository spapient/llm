//! CUDA 分块归一化 lowering：RMSNorm 两段、Softmax 三段，以及它们的 host 串联驱动。
//!
//! 归一化是「跨元素依赖 + 逐元素映射」两件事：部分平方和 / 部分最大值按 `[chunk]` 切块归约
//! （每线程一块，主机折叠），归一化按同一块网格回写。与 Metal 侧由 Rust executor 逐段
//! register 的路径不同——CUDA 的一次执行是**单个 host 程序**，折叠在 C 里做，因此多段设备源
//! 与驱动必须同处一个翻译单元。
//!
//! 多段同文件有两个硬约束：
//! - 每段入口名必须不同（`PARTIAL` / `EXPSUM` / `NORM`）；
//! - 只有第一段携带设备 preamble（软件归约 / exp / sqrt 原语），后续段从 `__global__` 起截断，
//!   否则同一份 preamble 在一个翻译单元里重复定义。
//!
//! 折叠公式与 `.k` 模板逐字同口径：RMSNorm 的 `sqrt(sum_sq / N + 1e-6f)`、Softmax 的
//! `max` 与 `exp(x - max)` 求和；误差口径由同一份标量参考对拍。

use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::Signature;

use cutile::harness::output_name;

use crate::codegen::{
    CUDA, buffer_args, chunk_reduce_entry, cuda_buffers, cuda_host_buffers_grid, cuda_readback,
    cuda_release, kernel_entry, length_scalar, scalar_name,
};

/// 部分归约段入口（部分平方和 / 部分最大值）。
pub(crate) const PARTIAL: &str = "kernel_partial";
/// 部分指数和段入口（Softmax 第二段）。
pub(crate) const EXPSUM: &str = "kernel_expsum";
/// 归一化段入口（RMSNorm / Softmax 的末段）。
pub(crate) const NORM: &str = "kernel_norm";

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
    let partial = chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        "_cutile_sumsq_f32",
        &CUDA,
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
    let max = chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        "_cutile_max_f32",
        &CUDA,
        PARTIAL,
    )?;
    let expsum = softmax_expsum_map(signature, sample, chunk)?;
    let norm = softmax_norm_map(signature, sample)?;
    Ok(vec![max, expsum, norm])
}

/// RMSNorm 归一化段：每线程一块，`y[start + j] = gamma[start + j] * x[start + j] / rms`。
fn rmsnorm_map(signature: &Signature, sample: &Sample) -> Result<String> {
    let len = length_scalar(signature, sample)?;
    let reads = readonly_buffers(signature);
    let (Some(x), Some(gamma)) = (reads.first(), reads.get(1)) else {
        return Err(Error::new(
            Code::Invalid,
            "RMSNorm 需要两个只读缓冲（x, gamma）".to_owned(),
        )
        .with("engine::norm::rmsnorm_map"));
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
        head = chunk_head(chunk, len),
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

/// 分块 map 段的公共头：块起点与块内长度（末块可能不足 `chunk`）。
fn chunk_head(chunk: usize, len: i64) -> String {
    format!(
        "    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n"
    )
}

/// 一个分块 map 入口：签名缓冲形参 + 额外标量（`rms` / `max` / `sum`），入口名由调用方给。
fn map_entry(signature: &Signature, entry: &str, extra: &str, body: &str) -> String {
    let buffers = cuda_buffers(signature);
    let buffers = if extra.is_empty() {
        buffers
    } else {
        format!("{buffers},\n    {extra}")
    };
    kernel_entry(&CUDA, "", "", &buffers, ("", CUDA.grid_stmt), body, entry)
}

/// 一个逐元素网格入口：每线程一个元素，线程下标符号是 `n`（[`CUDA`] 的 `point_stmt`）。
fn point_entry(signature: &Signature, entry: &str, extra: &str, body: &str) -> String {
    let buffers = cuda_buffers(signature);
    let buffers = if extra.is_empty() {
        buffers
    } else {
        format!("{buffers},\n    {extra}")
    };
    kernel_entry(&CUDA, "", "", &buffers, ("", CUDA.point_stmt), body, entry)
}

/// 只读缓冲名，按签名顺序。
fn readonly_buffers(signature: &Signature) -> Vec<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer() && !param.mutable)
        .map(|param| param.name.as_str())
        .collect()
}

/// 唯一只读缓冲名（Softmax 形态）；缺或多都返回 `Invalid`。
fn single_read<'a>(signature: &'a Signature, family: &str) -> Result<&'a str> {
    let reads = readonly_buffers(signature);
    match reads.as_slice() {
        [only] => Ok(*only),
        _ => Err(Error::new(
            Code::Invalid,
            format!("{family} 需要恰好一个只读缓冲，得到 {}", reads.len()),
        )
        .with("engine::norm")),
    }
}

/// 分块归一化的完整 CUDA 程序：多段设备源（仅首段带 preamble）+ host 串联驱动。
///
/// RMSNorm：部分平方和 → 主机折叠 `sqrt(sum/N + 1e-6)` → 归一化。
pub fn cuda_rmsnorm_driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    let len = length_usize(signature, sample)?;
    let grid = len.div_ceil(chunk);
    let output = output_name(signature)?;
    let args = buffer_args(signature);
    let mut text = program_head(segments);
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_buffers_grid(
        sample,
        signature.params(),
        output,
        len,
    )?);
    text.push_str(&partial_scaffold(grid));
    text.push_str(&launch(PARTIAL, grid, &args, ""));
    text.push_str(&partial_copy(output, grid));
    text.push_str("    float sum = 0.0f;\n    for (int i = 0; i < G; ++i) { sum += h_tmp[i]; }\n");
    text.push_str(&format!(
        "    float rms = sqrtf(sum / (float){len} + 1e-6f);\n"
    ));
    text.push_str(&launch_threads(NORM, len.div_ceil(256), 256, &args, "rms"));
    text.push_str(&cuda_readback(output, len));
    text.push_str(&cuda_release(signature));
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// 分块 Softmax 的完整 CUDA 程序：部分最大 → 折叠 max → 部分指数和 → 折叠 sum → 归一化。
pub fn cuda_softmax_driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    let len = length_usize(signature, sample)?;
    let grid = len.div_ceil(chunk);
    let output = output_name(signature)?;
    let args = buffer_args(signature);
    let mut text = program_head(segments);
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_buffers_grid(
        sample,
        signature.params(),
        output,
        len,
    )?);
    text.push_str(&partial_scaffold(grid));
    text.push_str(&launch(PARTIAL, grid, &args, ""));
    text.push_str(&partial_copy(output, grid));
    text.push_str(&fold_max());
    text.push_str(&launch(EXPSUM, grid, &args, "max"));
    text.push_str(&partial_copy(output, grid));
    text.push_str(&fold_sum());
    text.push_str(&launch_threads(
        NORM,
        len.div_ceil(256),
        256,
        &args,
        "max, sum",
    ));
    text.push_str(&cuda_readback(output, len));
    text.push_str(&cuda_release(signature));
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// 长度标量的 `usize` 视图；负值或超长返回 `Invalid`。
pub(crate) fn length_usize(signature: &Signature, sample: &Sample) -> Result<usize> {
    let len = length_scalar(signature, sample)?;
    usize::try_from(len).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!(
                "长度标量 `{}` = {len} 不是非负 usize",
                scalar_name(signature).unwrap_or("?")
            ),
        )
        .with("engine::norm::length_usize")
    })
}

/// 程序头：CUDA include、错误检查宏、多段设备源。
///
/// 设备 preamble（`_cutile_*` 软件原语）在第一段完整出现；后续段从各自的 `__global__`
/// 入口起截断——同一翻译单元里重复定义 preamble 会编译失败。
pub(crate) fn program_head(segments: &[String]) -> String {
    let mut text = String::from(
        "#include <cuda_runtime.h>\n#include <cstdio>\n\
         #define CUTILE_CUDA_CHECK() do { cudaError_t error = cudaDeviceSynchronize(); \
         if (error != cudaSuccess) { fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; } } while (0)\n",
    );
    for (index, segment) in segments.iter().enumerate() {
        let piece = if index == 0 {
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
    match segment.rfind(MARK) {
        Some(at) => {
            let start = at + MARK.len() - "__global__ void ".len();
            &segment[start..]
        }
        None => segment,
    }
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
