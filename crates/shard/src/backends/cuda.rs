//! CUDA / MUSA 后端：见证片段装配 + `nvcc` / `mcc` 编译执行回读。
//! 设备算子正文来自 prof 档案片段；本模块不按骨架重写算子算法。

use crate::backend::{Backend, Codegen, Driver, Readback, Source};
use crate::backends::common::{device_fn, witness_head, witness_readback};
use crate::driver;
use crate::platform::{detect_musa, detect_nvcc, find_cl, musa_device};
use crate::toolchain::Toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use cutile::dirs::Dirs;
use diagnostic::{Code, Error, Result};
use prof::prof::{Emit, Profile};
use std::path::Path;
use std::process::Command;

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

fn cuda_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = witness_head("#include <cuda_runtime.h>\n#include <cstdio>\n", profile);
    let files = device_fn(
        driver::witness_files(selected, dirs, driver::C_PARAMS)?,
        "__device__",
    );
    driver::push_sources(&mut program, &files);
    push_cuda_kernel(&mut program, MAX_OUT, selected);
    push_cuda_main(&mut program, MAX_OUT, selected.len());
    Ok(program)
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
         }\n",
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
    let (src, exe) = (
        dir.join(format!("{label}.cu")),
        dir.join(format!("{label}.exe")),
    );
    let outcome = (|| {
        driver::write_text(&src, program)?;
        cuda_compile(&exe, &src, profile, dirs)?;
        let mut command = Command::new(&exe);
        driver::run_exe(&mut command, &exe, "CUDA 执行失败")
    })();
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn cuda_compile(exe: &Path, src: &Path, profile: &Profile, dirs: &Dirs) -> Result<()> {
    let tool = profile.meta.tool.as_str();
    let mut command = Command::new(tool);
    // 头文件垫片归档案（`[compile.headers]`）：生成程序里的 `<cuda_runtime.h>` 与工具链实际提供的
    // `<musa_runtime.h>` 之间的映射写在 `devices/musa.toml`，本层只负责把它变成 `-I`。
    command.args(crate::compile_flags(profile, dirs)?);
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
