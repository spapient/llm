//! 分块归一化族：RMSNorm 两段、Softmax 三段的设备 map 与 host 串联驱动。多段公共装配在
//! [`super::segments`]，这里只写各族的段生成与折叠顺序。

use crate::dialect::{
    Cuda, Dialect, EXPSUM, NORM, PARTIAL, RMS_EPS, chunk_head, length_scalar, readonly_bufs,
};
use cutile::harness::output_name;
use diagnostic::{Code, Error, Result};
use prof::model::Sample;
use prof::sig::Signature;

use super::segments::{
    Stage, driver, fold_max, fold_sum, launch, launch_threads, partial_copy, partial_scaffold_step,
};
use super::shared::buf_args;

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
    );
    Ok(chunk_entry(signature, NORM, "float rms", &body))
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
    Ok(chunk_entry(signature, EXPSUM, "float max", &body))
}

/// Softmax 第三段：`y[start + j] = exp(x - max) / sum`。
fn softmax_norm_map(signature: &Signature, sample: &Sample) -> Result<String> {
    let len = length_scalar(signature, sample)?;
    let x = single_read(signature, "Softmax")?;
    let y = output_name(signature)?;
    let body = format!(
        "    if (n < {len}) {{\n        d_{y}[n] = _cutile_exp_f32(d_{x}[n] - max) / sum;\n    }}",
    );
    Ok(chunk_entry(signature, NORM, "float max, float sum", &body))
}

/// 分块族的 map 入口：签名缓冲形参 + 额外标量（`rms` / `max` / `sum`）。线程下标语句由入口职责定
/// ——归一化段（`NORM`）逐元素改写，其余段（`PARTIAL` / `EXPSUM`）每线程一块——不留给调用方挑。
fn chunk_entry(signature: &Signature, entry: &str, extra: &str, body: &str) -> String {
    let bufs = Cuda::bufs(signature);
    let bufs = if extra.is_empty() {
        bufs
    } else {
        format!("{bufs},\n    {extra}")
    };
    let stmt = if entry == NORM {
        Cuda::POINT_STMT
    } else {
        Cuda::GRID_STMT
    };
    Cuda::kernel_entry("", "", &bufs, ("", stmt), body, entry)
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
    let args = buf_args(signature);
    driver(
        segments,
        signature,
        sample,
        chunk,
        &args,
        partial_scaffold_step,
        |text, stage| {
            partial_pass(text, &stage);
            text.push_str(&fold_sum());
            text.push_str(&format!(
                "    float rms = sqrtf(sum / (float){len} + {eps}f);\n",
                len = stage.len,
                eps = RMS_EPS
            ));
            normalized_pass(text, &stage, "rms");
        },
    )
}

/// 分块 Softmax 的完整 CUDA 程序：部分最大 → 折叠 max → 部分指数和 → 折叠 sum → 归一化。
pub fn cuda_softmax_driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<String> {
    let args = buf_args(signature);
    driver(
        segments,
        signature,
        sample,
        chunk,
        &args,
        partial_scaffold_step,
        |text, stage| {
            partial_pass(text, &stage);
            text.push_str(&fold_max());
            text.push_str(&launch(EXPSUM, stage.grid, stage.args, "max"));
            partial_pass(text, &stage);
            text.push_str(&fold_sum());
            normalized_pass(text, &stage, "max, sum");
        },
    )
}
