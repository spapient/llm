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

/// host 侧部分结果的脚手架：网格数 `G`、回读数组，无额外缓冲要释放。
fn partial_scaffold(grid: usize) -> String {
    format!("    int G = {grid};\n    float h_tmp[{grid}];\n")
}

/// [`driver`] 的零额外缓冲脚手架：部分结果直接落在输出缓冲的前 `G` 个槽。
fn partial_scaffold_step(grid: usize) -> (String, &'static str) {
    (partial_scaffold(grid), "")
}

/// 扫描的脚手架：块数 `G`、携带前缀设备缓冲 `d_c` 与主机数组 `h_c`（用完即释放）。
fn carry_scaffold(grid: usize) -> (String, &'static str) {
    (
        format!(
            "    int G = {grid};\n    float *d_c = 0;\n    cudaMalloc((void **)&d_c, G * sizeof(float));\n    float h_c[{grid}];\n"
        ),
        "    cudaFree(d_c);\n",
    )
}

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
    let args = launch_args(sample, signature.params())?;
    driver(
        segments,
        signature,
        sample,
        chunk,
        &args,
        carry_scaffold,
        |text, stage| {
            let (len, grid, output) = (stage.len, stage.grid, stage.output);
            text.push_str(&launch("kernel_main", grid, stage.args, ""));
            text.push_str(&format!(
                "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n"
            ));
            text.push_str(&fold_carry(output, chunk, len as i64, decay));
            text.push_str("    cudaMemcpy(d_c, h_c, G * sizeof(float), cudaMemcpyHostToDevice);\n");
            text.push_str(&format!(
                "    {CORRECT}<<<G, 1>>>(d_c, d_{output});\n    CUTILE_CUDA_CHECK();\n"
            ));
        },
    )
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
