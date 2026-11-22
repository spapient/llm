//! CUDA 分块扫描 lowering：局部扫描 + 主机携带前缀 + 校正，以及 host 串联驱动。

use crate::cuda::{cuda_host_bufs_grid, cuda_launch_params, cuda_readback, cuda_release};
use crate::dialect::{
    Cuda, Dialect, bind_chunk_loop, carry_fold, chunk_head, inject_chunk_preamble, launch_args,
    length_scalar, scalar_name, scan_correct_body, thread_param_typed,
};
use crate::norm::{launch, length_usize, program_head};
use cutile::harness::output_name;
use diagnostic::Result;
use op::model::Sample;
use op::sig::Signature;

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
