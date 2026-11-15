//! CUDA 真机编排：nvcc 把见证编进 `__global__` kernel，在 GPU 上按 tid 逐条执行，再 cudaMemcpy 回读。
//!
//! 与 `CudaFront` 的主机语法自检不同，这里是真设备执行；它**不**证明别的架构或别的 `-arch` 下成立。
//! 片段本体来自 `prim`，本层只做编译编排与回读。共享装载、起进程与回读在 [`crate::device`]。

use crate::device::{self, C_PARAMS, Readback, Sources};
use crate::dirs::Dirs;
use crate::prof::Profile;
use crate::toolchain;
use crate::witness::{MAX_OUT, Selected};
use diagnostic::{Code, Error, Result};
use std::path::Path;
use std::process::Command;

pub(crate) fn cuda_available(_: &Profile) -> bool {
    // nvcc 在，且 Windows 上能找到宿主 cl（Linux 上 nvcc 用系统 gcc，不需要 cl）。
    toolchain::detect_nvcc().is_some()
        && (std::env::consts::OS != "windows" || toolchain::find_cl().is_some())
}

pub(crate) fn cuda_program(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    _scope: crate::witness::Scope,
) -> Result<String> {
    cuda_program_inner::<MAX_OUT>(profile, dirs, selected)
}

fn cuda_program_inner<const OUT: usize>(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
) -> Result<String> {
    let mut program = cuda_head(profile);
    let files = as_device(device::witness_files(selected, dirs, C_PARAMS)?);
    device::push_sources(&mut program, &files);
    push_cuda_kernel(&mut program, OUT, selected);
    push_cuda_main(&mut program, OUT, selected.len());
    Ok(program)
}

pub(crate) fn cuda_run(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    _count: usize,
    _scope: crate::witness::Scope,
) -> Result<Readback> {
    let (stdout, stderr) = compile_run_cuda(profile, dirs, program, "witness")?;
    Ok(device::readback(&stdout, &stderr))
}

/// CUDA 的头：runtime 与 `stdio` 的 include，加上档案的 `[compile].preamble`。非空 preamble 之后固定补一个
/// 换行——统一成"缺了才补"会改掉每一份已缓存的程序文本。
fn cuda_head(profile: &Profile) -> String {
    let mut head = String::from("#include <cuda_runtime.h>\n#include <cstdio>\n");
    if !profile.preamble.is_empty() {
        head.push_str(&profile.preamble);
        head.push('\n');
    }
    head
}

/// 给见证定义加 `__device__`：kernel 要调用它们，而 `int witness` 只出现在定义处。
fn as_device(files: Sources) -> Sources {
    files
        .into_iter()
        .map(|(file, text)| (file, text.replace("int witness", "__device__ int witness")))
        .collect()
}

/// 内核包装：`__global__` 按 `blockIdx.x` 分派，一个 block 一条见证。
fn push_cuda_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "__global__ void cutile_witness(float *out, int *counts, int total) {{\n\
         \x20   int tid = blockIdx.x;\n\
         \x20   if (tid >= total) return;\n\
         \x20   float local[{max}];\n\
         \x20   for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n\
         \x20   int n = 0;\n\
         \x20   switch (tid) {{\n"
    ));
    device::push_dispatch(program, max, selected, "}}\n\n");
}

/// host main：分配、发射、同步、回读——stdout 只按实际写入个数打印位模式（与 CPU harness
/// 同口径），counts 走 stderr。
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
         \x20   cudaMemcpy(h_counts, d_counts, total * sizeof(int), cudaMemcpyDeviceToHost);\n\
         \x20   for (int t = 0; t < total; ++t) {{\n\
         \x20       for (int i = 0; i < h_counts[t]; ++i) {{\n\
         \x20           union {{ float f; unsigned int u; }} v;\n\
         \x20           v.f = h_out[t * {max} + i];\n\
         \x20           printf(\"%u\\n\", v.u);\n\
         \x20       }}\n\
         \x20       fprintf(stderr, \"witness %d\\n\", h_counts[t]);\n\
         \x20   }}\n\
         \x20   cudaFree(d_out); cudaFree(d_counts);\n\
         \x20   return 0;\n\
         }}\n"
    ));
}

/// 编译并运行任意 CUDA 程序，返回 `(stdout, stderr)` 原文：临时目录、nvcc（Windows 用 `-ccbin` 指向 cl）、
/// 执行。见证与算子 verify 共用这一条；写、编、跑三段分开。
pub fn compile_run_cuda(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    let dir = device::scratch(dirs, &format!("{label}-cuda"));
    let outcome = cuda_in_dir(&dir, profile, program, label);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

/// 一次性目录里的一条 CUDA 程序：写源 → nvcc 编 → 跑一遍。
fn cuda_in_dir(
    dir: &Path,
    profile: &Profile,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    let src = dir.join(format!("{label}.cu"));
    let exe = dir.join(format!("{label}.exe"));
    device::write_text(&src, program)?;
    cuda_compile(&exe, &src, profile)?;
    let mut command = Command::new(&exe);
    device::run_exe(&mut command, &exe, "CUDA 执行失败")
}

/// nvcc 编译：Windows 上用 `-ccbin` 显式指向 cl.exe，不依赖 vcvars/PATH。
fn cuda_compile(exe: &Path, src: &Path, profile: &Profile) -> Result<()> {
    let mut command = Command::new("nvcc");
    command.args(&profile.flags);
    if std::env::consts::OS == "windows" {
        let Some(cl) = toolchain::find_cl() else {
            return Err(Error::new(
                Code::Unsupported,
                "找不到 MSVC cl.exe：nvcc 设备编译需要它".to_string(),
            ));
        };
        command.arg("-ccbin").arg(cl);
    }
    command.arg("-o").arg(exe).arg(src);
    device::run_tool("nvcc", &mut command, src)
}
