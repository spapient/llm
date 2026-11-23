//! 多段生产态程序的共享 host 脚手架：段程序头、块级派发、部分结果回读与主机折叠、中间段缓冲
//! 脚手架。归一化（RMSNorm / Softmax）与扫描两族只在「每段怎么派发、折叠成什么」上不同，公共
//! 装配与折叠字面只写这一份。

use crate::dialect::length_usize;
use cutile::harness::output_name;
use diagnostic::{Error, Result};
use prof::model::Sample;
use prof::sig::Signature;

use super::shared::{cuda_host_bufs_grid, cuda_readback, cuda_release};

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

/// host 侧部分结果的脚手架：网格数 `G`、回读数组，无额外缓冲要释放。
fn partial_scaffold(grid: usize) -> String {
    format!("    int G = {grid};\n    float h_tmp[{grid}];\n")
}

/// [`super::chunk::driver`] 的零额外缓冲脚手架：部分结果直接落在输出缓冲的前 `G` 个槽。
pub(crate) fn partial_scaffold_step(grid: usize) -> (String, &'static str) {
    (partial_scaffold(grid), "")
}

/// 扫描的脚手架：块数 `G`、携带前缀设备缓冲 `d_c` 与主机数组 `h_c`（用完即释放）。
pub(crate) fn carry_scaffold(grid: usize) -> (String, &'static str) {
    (
        format!(
            "    int G = {grid};\n    float *d_c = 0;\n    cudaMalloc((void **)&d_c, G * sizeof(float));\n    float h_c[{grid}];\n"
        ),
        "    cudaFree(d_c);\n",
    )
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
pub(crate) fn fold_max() -> String {
    String::from(
        "    float max = -3.402823466e38f;\n    for (int i = 0; i < G; ++i) { if (h_tmp[i] > max) max = h_tmp[i]; }\n",
    )
}

/// 主机折叠部分指数和。
pub(crate) fn fold_sum() -> String {
    String::from("    float sum = 0.0f;\n    for (int i = 0; i < G; ++i) { sum += h_tmp[i]; }\n")
}

/// 分块族 host 程序的骨架：多段设备源（仅首段带 preamble）→ `main` → 主机缓冲 → 中间段脚手架 →
/// 各段派发与主机折叠 → 回读 → 额外缓冲释放 → 全部缓冲释放。`scaffold` 给中间段声明与用完即释的
/// 缓冲，`reduce` 写派发与折叠；归一化与扫描的差异全在这两个参数里，骨架只有这一份。
pub(crate) fn driver(
    segments: &[String],
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    args: &str,
    scaffold: impl FnOnce(usize) -> (String, &'static str),
    reduce: impl FnOnce(&mut String, Stage<'_>),
) -> Result<String> {
    let len = length_usize(signature, sample)?;
    let grid = len.div_ceil(chunk);
    let output = output_name(signature)?;
    let mut text = program_head(segments);
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_bufs_grid(
        sample,
        signature.params(),
        output,
        len,
    )?);
    let (decl, free) = scaffold(grid);
    text.push_str(&decl);
    reduce(
        &mut text,
        Stage {
            len,
            grid,
            output,
            args,
        },
    );
    text.push_str(&cuda_readback(output, len, None, false));
    text.push_str(free);
    text.push_str(&cuda_release(signature));
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// 驱动中段拿到的几何与实参：`len` 是归约长度，`grid` 是块数，`output` 是输出缓冲名，
/// `args` 是签名的缓冲实参（追加标量由各段自己给 `extra`）。
pub(crate) struct Stage<'a> {
    pub(crate) len: usize,
    pub(crate) grid: usize,
    pub(crate) output: &'a str,
    pub(crate) args: &'a str,
}
