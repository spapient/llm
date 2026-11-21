//! CUDA / MUSA 后端：`Codegen` 出 `.cu`、`Driver` 跑 `nvcc`。
//!
//! 从 [`crate::backends`] 拆出——三条厂商链之间没有共享状态。共用的基准几何与见证回读留在父模块。

use crate::toolchain::{detect_musa, detect_nvcc, find_cl, musa_device};
use std::path::Path;
use std::process::Command;

use super::*;

fn cuda_bench_geometry(
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<([usize; 3], [usize; 3], String)> {
    match kind {
        Lower::Pointwise => pointwise_wrap(signature, sample),
        Lower::Loop(grid) => loop_wrap(signature, sample, grid),
        Lower::QuantizedGemvI4 { threads } => warp_wrap(signature, sample, *threads),
        _ => Err(Error::new(
            Code::Unsupported,
            "该 lowering 尚无单 kernel 设备计时封装",
        )),
    }
}

/// 逐点族：一条线程一个输出元素，块宽 256。
fn pointwise_wrap(
    signature: &Signature,
    sample: &Sample,
) -> Result<([usize; 3], [usize; 3], String)> {
    let len = benchmark_output_len(signature, sample)?;
    Ok((
        [len.div_ceil(256), 1, 1],
        [256, 1, 1],
        crate::cuda::buffer_args(signature),
    ))
}

/// 多段族：网格就是声明里的网格，一个线程一段。
fn loop_wrap(
    signature: &Signature,
    sample: &Sample,
    grid: &op::decl::Grid,
) -> Result<([usize; 3], [usize; 3], String)> {
    Ok((
        benchmark_grid(grid, sample)?,
        [1, 1, 1],
        crate::dialect::launch_args(sample, signature.params())?,
    ))
}

/// i4 GEMV 一行一个 warp：`grid = [N, 1, 1]`、`block = [threads, 1, 1]`。
fn warp_wrap(
    signature: &Signature,
    sample: &Sample,
    threads: usize,
) -> Result<([usize; 3], [usize; 3], String)> {
    let rows = warp_rows(sample)?;
    Ok((
        [rows, 1, 1],
        [warp_lanes(threads)?, 1, 1],
        crate::dialect::launch_args(sample, signature.params())?,
    ))
}

/// GEMV 的规约按 lane 展开，因此 32-lane warp 是硬条件而非默认值。
fn warp_lanes(threads: usize) -> Result<usize> {
    if threads != 32 {
        return Err(Error::new(Code::Unsupported, "CUDA GEMV 要求 32-lane warp"));
    }
    Ok(threads)
}

/// 输出宽度 `N` 必须存在且为正整数：零行或负行没有可写的输出。
fn warp_rows(sample: &Sample) -> Result<usize> {
    let rows = sample
        .scalar("N")
        .ok_or_else(|| Error::new(Code::Invalid, "gemv_i4 样本缺少输出宽度 N"))?;
    usize::try_from(rows)
        .ok()
        .filter(|rows| *rows > 0)
        .ok_or_else(|| Error::new(Code::Invalid, "gemv_i4 输出宽度 N 必须为正数"))
}

pub(crate) struct CudaBackend;

impl Codegen for CudaBackend {
    fn emit(&self) -> Emit {
        Emit::Cuda
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope; // CUDA 只走设备执行
        cuda_program(profile, dirs, selected)
    }
    fn lower(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
    ) -> Result<Vec<String>> {
        cuda_lower(kernel, signature, sample, kind)
    }
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        Ok(Source::Single(crate::cuda::cuda_harness(
            kernel, signature, sample,
        )?))
    }
    fn parallel(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        launch: &Launch,
    ) -> Result<Source> {
        crate::program::cuda(kernel, signature, sample, launch).map(Source::Single)
    }
    fn benchmark_harness(
        &self,
        profile: &Profile,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
        repeat: usize,
    ) -> Result<Source> {
        let lowered = cuda_lower(kernel, signature, sample, kind)?;
        let [device] = lowered.as_slice() else {
            return Err(Error::new(
                Code::Unsupported,
                "CUDA 基准当前只支持单段生产态 kernel",
            ));
        };
        let (grid, threads, args) = cuda_bench_geometry(signature, sample, kind)?;
        let source = crate::cuda::cuda_benchmark_driver(
            device,
            signature,
            sample,
            grid,
            threads,
            &args,
            repeat,
            profile.meta.tool == "mcc",
        )?;
        Ok(Source::Single(source))
    }
}

impl Driver for CudaBackend {
    /// 可用性按**档案声明的工具名**判，不按厂商判：同一份 `Emit::Cuda` 后端既是 `nvcc` 的落地，
    /// 也是 `mcc` 的落地。`nvcc` 在 Windows 上还要 MSVC 宿主；`mcc` 不调宿主 cl（它自己就是 clang）。
    fn available(&self, profile: &Profile) -> bool {
        match profile.meta.tool.as_str() {
            "nvcc" => nvcc_ready(),
            "mcc" => musa_ready(),
            _ => false,
        }
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        let (stdout, stderr) = compile_run_cuda(profile, dirs, program, "witness")?;
        Ok(driver::readback(&stdout, &stderr))
    }
    /// 缓存指纹里的工具链：按现有工具选一份（同一台机器上不会同时是 CUDA 与 MUSA 的编译机，
    /// 真同时有时取先命中的那份，两个档案的缓存键因此天然分开）。
    fn toolchain(&self) -> Option<Toolchain> {
        detect_nvcc().or_else(detect_musa)
    }
    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Single(text) => compile_run_cuda(profile, dirs, text, label),
            Source::Pair { .. } => Err(Error::new(
                Code::Unsupported,
                "CUDA 后端需要单源（.cu）".to_string(),
            )),
        }
    }
}

impl Backend for CudaBackend {}

/// 把中性的 [`Lower`] 描述降成 CUDA：逐点网格与分块归约已落地，其余骨架尚未覆盖。
fn cuda_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        // 逐块归一化两族与扫描自己就产出多段源码，不经过下面的单文件包装。
        ChunkRmsnorm { chunk } => {
            crate::norm::cuda_chunk_rmsnorm(kernel, signature, sample, *chunk)
        }
        ChunkSoftmax { chunk } => {
            crate::norm::cuda_chunk_softmax(kernel, signature, sample, *chunk)
        }
        ChunkScan { chunk, decay } => {
            crate::scan::cuda_chunk_scan(kernel, signature, sample, *chunk, *decay)
        }
        QuantizedGemvI4 { threads } => {
            crate::cuda::cuda_quantized_gemv_i4(signature, *threads).map(single_file)
        }
        // 其余骨架都只吐一段源码；分派单独一层，是为了让这一层只回答「哪一族」。
        _ => cuda_single_source(kind, kernel, signature, sample).map(single_file),
    }
}

/// 单段源码骨架的分派：pointwise / chunk-reduce / loop / wmma 各产出一段源码。
///
/// 两段族的骨架已在 [`cuda_lower`] 里分派，落到这里的只会有这四族之一；兜底臂不是「偷偷吞掉」，
/// 它回答的是「这个骨架没有单段 CUDA 生产态 lowering」。
fn cuda_single_source(
    kind: &Lower,
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    match kind {
        Pointwise => crate::cuda::cuda_grid_kernel(kernel, signature, sample),
        ChunkReduce { chunk, func } => {
            crate::cuda::cuda_chunk_reduce(kernel, signature, sample, *chunk, func)
        }
        Loop(grid) => crate::cuda::cuda_loop_kernel(kernel, signature, sample, grid),
        WmmaGemm => crate::wmma::gemm_kernel(signature, sample),
        _ => Err(Error::new(
            Code::Internal,
            "该骨架产出多段源码，应在 cuda_lower 分派，不应落到单段兜底".to_string(),
        )),
    }
}

/// 单段源码 → 单元素文件表（CUDA 生产态单段骨架只产出一个翻译单元）。
fn single_file(source: String) -> Vec<String> {
    vec![source]
}

fn cuda_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = cuda_head(profile);
    let files = as_device(driver::witness_files(selected, dirs, driver::C_PARAMS)?);
    driver::push_sources(&mut program, &files);
    push_cuda_kernel(&mut program, MAX_OUT, selected);
    push_cuda_main(&mut program, MAX_OUT, selected.len());
    Ok(program)
}

fn cuda_head(profile: &Profile) -> String {
    let mut head = String::from("#include <cuda_runtime.h>\n#include <cstdio>\n");
    if !profile.preamble.is_empty() {
        head.push_str(&profile.preamble);
        head.push('\n');
    }
    head
}

fn as_device(files: driver::Sources) -> driver::Sources {
    files
        .into_iter()
        .map(|(file, text)| (file, text.replace("int witness", "__device__ int witness")))
        .collect()
}

fn push_cuda_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "__global__ void cutile_witness(float *out, int *counts, int total) {{\n\
         \x20   int tid = blockIdx.x;\n\
         \x20   if (tid >= total) return;\n\
         \x20   float local[{max}];\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n\n");
}

/// 回读口径：逐见证打印位模式，再打一行 `witness N`。
///
/// CUDA 与昇腾的 host 驱动共用这一段字面：同一个 `driver::readback` 在两种产物上必须读出同一种
/// 位模式。
fn push_cuda_main(program: &mut String, max: usize, total: usize) {
    program.push_str(&format!(
        "int main() {{\n\
         \x20   const int total = {total};\n\
         \x20   float *d_out = 0; int *d_counts = 0;\n\
         \x20   cudaMalloc((void **)&d_out, total * {max} * sizeof(float));\n\
         \x20   cudaMalloc((void **)&d_counts, total * sizeof(int));\n\
         \x20   cutile_witness<<<total, 1>>>(d_out, d_counts, total);\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{\n\
         \x20       fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error));\n\
         \x20       return 1;\n\
         \x20   }}\n\
         \x20   float h_out[total * {max}]; int h_counts[total];\n\
         \x20   cudaMemcpy(h_out, d_out, total * {max} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   cudaMemcpy(h_counts, d_counts, total * sizeof(int), cudaMemcpyDeviceToHost);\n"
    ));
    program.push_str(&witness_readback(max));
    program.push_str(
        "\x20   cudaFree(d_out); cudaFree(d_counts);\n\
         \x20   return 0;\n\
         \x20   }\n",
    );
}

/// 编译并运行任意 CUDA 程序，返回 `(stdout, stderr)` 原文：临时目录、nvcc（Windows 用 `-ccbin`
/// 指向 cl）、执行。见证与算子 verify 共用这一条；写、编、跑三段分开。
pub fn compile_run_cuda(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    let dir = driver::scratch(dirs, &format!("{label}-cuda"));
    let outcome = cuda_in_dir(&dir, profile, program, label, dirs);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn cuda_in_dir(
    dir: &Path,
    profile: &Profile,
    program: &str,
    label: &str,
    dirs: &Dirs,
) -> Result<(String, String)> {
    let src = dir.join(format!("{label}.cu"));
    let exe = dir.join(format!("{label}.exe"));
    driver::write_text(&src, program)?;
    cuda_compile(&exe, &src, profile, dirs)?;
    let mut command = Command::new(&exe);
    driver::run_exe(&mut command, &exe, "CUDA 执行失败")
}

fn cuda_compile(exe: &Path, src: &Path, profile: &Profile, dirs: &Dirs) -> Result<()> {
    let tool = profile.meta.tool.as_str();
    let mut command = Command::new(tool);
    // 头文件垫片归档案（`[compile.headers]`）：生成程序里的 `<cuda_runtime.h>` 与工具链实际提供的
    // `<musa_runtime.h>` 之间的映射写在 `devices/musa.toml`，本层只负责把它变成 `-I`。
    command.args(llm_optim::compile_flags(profile, dirs)?);
    cuda_ccbin(tool, &mut command)?;
    command.arg("-o").arg(exe).arg(src);
    driver::run_tool(tool, &mut command, src)
}

/// nvcc 的可用性：编译器在，且 Windows 上还能找到 MSVC 的 `cl.exe`。
fn nvcc_ready() -> bool {
    detect_nvcc().is_some() && (std::env::consts::OS != "windows" || find_cl().is_some())
}

/// MUSA 的可用性：编译器与设备都在。
fn musa_ready() -> bool {
    detect_musa().is_some() && musa_device()
}

/// Windows 上的 nvcc 需要显式指定 MSVC 编译器。
fn cuda_ccbin(tool: &str, command: &mut Command) -> Result<()> {
    if !(tool == "nvcc" && std::env::consts::OS == "windows") {
        return Ok(());
    }
    let Some(cl) = find_cl() else {
        return Err(Error::new(
            Code::Unsupported,
            "找不到 MSVC cl.exe：nvcc 设备编译需要它".to_string(),
        ));
    };
    command.arg("-ccbin").arg(cl);
    Ok(())
}
