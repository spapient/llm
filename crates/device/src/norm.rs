//! CUDA 分块归一化与扫描 lowering：RMSNorm 两段、Softmax 三段、扫描两段，以及它们的 host 串联驱动。

use crate::cuda::{
    buf_args, cuda_host_bufs_grid, cuda_launch_params, cuda_readback, cuda_release,
};
use crate::dialect::{
    Cuda, Dialect, RMS_EPS, bind_chunk_loop, carry_fold, chunk_head, inject_chunk_preamble,
    launch_args, length_scalar, length_usize, readonly_bufs, scalar_name, scan_correct_body,
    thread_param_typed,
};
use cutile::harness::output_name;
use diagnostic::{Code, Error, Result};
use nn::model::Sample;
use nn::sig::Signature;

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
        .with("engine::norm")),
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
// CUDA 分块扫描 lowering：局部扫描 + 主机携带前缀 + 校正，以及 host 串联驱动
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
