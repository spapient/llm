//! Metal / CUDA 设备后端：实现 [`cutile::backend::Backend`]（`Codegen` 出源 + `Driver` 跑源）。
//!
//! 这是 L6 的设备落地：唯一碰 `nvcc` / `xcrun metal` / `metallib` / MSVC `cl.exe` 的地方。
//! cutile（L3 机制层）只认 [`cutile::backend::Backend`] 缝，不认识这里的任何厂商符号；本模块经
//! [`install`] 把 Metal / CUDA 后端注册进 cutile 的进程级表，由 L7（`cli`）在启动时调用。
//!
//! 双源说明：Metal 是「设备源（.metal）+ host 驱动（.m）」两段文本，CUDA 是单段 `.cu`；这个差异
//! 收在 [`cutile::backend::Source`] 里，由各自的 [`Driver::compile_run`] 解包，不外泄到机制层。

use cutile::backend::{Backend, Codegen, Driver, Lower, Readback, Source};
use cutile::dirs::Dirs;
use cutile::driver;
use cutile::prof::{Emit, Profile};
use cutile::toolchain::{Toolchain, parse_version, probe};
use cutile::witness::{MAX_OUT, Scope, Selected};
use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::Signature;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// 厂商工具链探测：nvcc / xcrun metal / MSVC cl.exe。
// ---------------------------------------------------------------------------

static NVCC: OnceLock<Option<Toolchain>> = OnceLock::new();
static METAL: OnceLock<Option<Toolchain>> = OnceLock::new();

/// 探测 nvcc（CUDA 编译器）：`nvcc --version` 只打印版本、不调用宿主 cl，故可独立成功；宿主编译器由
/// 调用方用 `-ccbin` 另行指定。
fn detect_nvcc() -> Option<Toolchain> {
    NVCC.get_or_init(|| probe("nvcc")).clone()
}

/// 探测 Metal 编译器（`xcrun metal --version`）：命令可用、能自报版本时才算数。
fn detect_metal() -> Option<Toolchain> {
    METAL.get_or_init(probe_metal).clone()
}

fn probe_metal() -> Option<Toolchain> {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    Some(Toolchain {
        command: "xcrun metal".to_string(),
        version: parse_version(&text),
        banner: text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())?
            .to_string(),
    })
}

/// 定位 MSVC cl.exe（Windows 上 nvcc 的宿主编译器）：先问 `vswhere`，再退回环境变量登记的两个
/// 安装根，多版本共存时按**数值**取最新。
fn find_cl() -> Option<PathBuf> {
    if std::env::consts::OS != "windows" {
        return None;
    }
    candidates().into_iter().max_by_key(|cl| msvc_version(cl))
}

fn candidates() -> Vec<PathBuf> {
    let mut found = vswhere_cls();
    for key in ["VCINSTALLDIR", "VSINSTALLDIR"] {
        if let Some(root) = std::env::var_os(key) {
            found.extend(msvc_cls(Path::new(&root)));
        }
    }
    found
}

fn vswhere_cls() -> Vec<PathBuf> {
    let Some(program_files) = std::env::var_os("ProgramFiles(x86)") else {
        return Vec::new();
    };
    let vswhere = Path::new(&program_files).join("Microsoft Visual Studio/Installer/vswhere.exe");
    let Ok(output) = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| PathBuf::from(line.trim()))
        .filter(|cl| cl.is_file())
        .collect()
}

fn msvc_cls(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for base in [root.join("Tools/MSVC"), root.join("VC/Tools/MSVC")] {
        let Ok(entries) = std::fs::read_dir(base) else {
            continue;
        };
        found.extend(
            entries
                .flatten()
                .map(|entry| entry.path().join("bin/Hostx64/x64/cl.exe")),
        );
    }
    found.into_iter().filter(|cl| cl.is_file()).collect()
}

fn msvc_version(cl: &Path) -> Vec<u32> {
    let text = cl.to_string_lossy().replace('/', "\\");
    let Some((_, rest)) = text.rsplit_once("MSVC\\") else {
        return Vec::new();
    };
    rest.split('\\')
        .next()
        .map(|version| {
            version
                .split('.')
                .map(|part| part.parse::<u32>().unwrap_or(0))
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Metal 后端
// ---------------------------------------------------------------------------

/// MSL 的形参必须带地址空间：MSL 没有默认地址空间，裸 `float *` 在那里是语法错误。
const MSL_PARAMS: &str = "thread float *";

struct MetalBackend;

impl Codegen for MetalBackend {
    fn emit(&self) -> Emit {
        Emit::Metal
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope; // Metal 只走设备执行，host 自检由 C 后端承担
        metal_program(profile, dirs, selected)
    }
    fn lower(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
    ) -> Result<Vec<String>> {
        metal_lower(kernel, signature, sample, kind)
    }
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        let (device, host) = crate::codegen::metal::metal_harness(kernel, signature, sample)?;
        Ok(Source::Pair { device, host })
    }
}

impl Driver for MetalBackend {
    fn available(&self, _: &Profile) -> bool {
        detect_metal().is_some()
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        run_metal(program, count, &profile.flags, dirs)
    }
    fn toolchain(&self) -> Option<Toolchain> {
        detect_metal()
    }
    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Pair { device, host } => compile_run_metal(profile, dirs, device, host, label),
            Source::Single(_) => Err(Error::new(
                Code::Unsupported,
                "Metal 后端需要双源（MSL + host 驱动）".to_string(),
            )),
        }
    }
}

impl Backend for MetalBackend {}

/// 把中性的 [`Lower`] 描述降成 MSL：逐点/循环走网格或串行，分块族（归约/扫描/归一化）走分块内核。
fn metal_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    use cutile::backend::Lower::*;
    match kind {
        Pointwise => crate::codegen::metal::metal_grid_kernel(kernel, signature).map(|s| vec![s]),
        Loop(grid) => crate::codegen::metal::metal_loop_kernel(kernel, signature, sample, grid)
            .map(|s| vec![s]),
        ChunkReduce { .. } | ChunkScan { .. } | ChunkRmsnorm { .. } | ChunkSoftmax { .. } => {
            metal_chunk_lower(kernel, signature, sample, kind)
        }
    }
}

/// 分块族 lowering：归约单段、扫描/归一化两段、softmax 三段，段序与 [`cutile::execution`] 的
/// `pair`/`triple` 解包一致（按弹出顺序还原）。
fn metal_chunk_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    use cutile::backend::Lower::*;
    match kind {
        ChunkReduce { chunk, func } => {
            crate::codegen::metal::metal_chunk_reduce(kernel, signature, sample, *chunk, func)
                .map(|s| vec![s])
        }
        ChunkScan { chunk, decay } => {
            crate::codegen::metal::metal_chunk_scan(kernel, signature, sample, *chunk, *decay)
                .map(|(local, correct)| vec![local, correct])
        }
        ChunkRmsnorm { chunk } => {
            crate::codegen::metal::metal_chunk_rmsnorm(kernel, signature, sample, *chunk)
                .map(|(partial, map)| vec![partial, map])
        }
        ChunkSoftmax { chunk } => {
            crate::codegen::metal::metal_chunk_softmax(kernel, signature, sample, *chunk)
                .map(|(max, expsum, map)| vec![max, expsum, map])
        }
        Pointwise | Loop(_) => Err(Error::new(Code::Internal, "非分块 lowering 走错入口")),
    }
}

fn metal_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = metal_head(profile);
    driver::push_sources(
        &mut program,
        &driver::witness_files(selected, dirs, MSL_PARAMS)?,
    );
    push_metal_kernel(&mut program, MAX_OUT, selected);
    Ok(program)
}

fn metal_head(profile: &Profile) -> String {
    let mut head = profile.preamble.clone();
    driver::normalize_newline(&mut head);
    head
}

fn push_metal_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "kernel void cutile_witness(device float *out [[buffer(0)]],\n\
         \x20                          device int *counts [[buffer(1)]],\n\
         \x20                          uint tid [[thread_position_in_grid]]) {{\n\
         \x20   thread float local[{max}];\n\
         \x20   for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n\
         \x20   int n = 0;\n\
         \x20   switch (tid) {{\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n");
}

fn run_metal(program: &str, count: usize, flags: &[String], dirs: &Dirs) -> Result<Readback> {
    let dir = driver::scratch(dirs, "witness-metal");
    let outcome = metal_in_dir(&dir, program, count, flags);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn metal_in_dir(dir: &Path, program: &str, count: usize, flags: &[String]) -> Result<Readback> {
    let metalib = metal_kernel(dir, program, flags)?;
    let exe = metal_driver_exe(dir)?;
    metal_readback(&exe, &metalib, count)
}

/// 写 MSL 并编成 `metallib`：`xcrun metal` 管语言层、`metallib` 管链接；诊断里带 `.metal` 路径。
pub fn metal_kernel(dir: &Path, program: &str, flags: &[String]) -> Result<PathBuf> {
    let msl = dir.join("witness.metal");
    let air = dir.join("witness.air");
    let metalib = dir.join("witness.metallib");
    driver::write_text(&msl, program)?;
    let mut compile = xcrun("metal", flags);
    compile.arg("-c").arg(&msl).arg("-o").arg(&air);
    driver::run_tool("xcrun metal", &mut compile, &msl)?;
    let mut link = xcrun("metallib", &[]);
    link.arg(&air).arg("-o").arg(&metalib);
    driver::run_tool("xcrun metallib", &mut link, &msl)?;
    Ok(metalib)
}

/// 编译并执行一次 Metal 生产态内核（设备源 + host 驱动），返回 `(stdout, stderr)` 原文。
/// 与 [`cutile::backend::Driver::compile_run`] 同口径：一次性目录，用完即删。
pub fn compile_run_metal(
    profile: &Profile,
    dirs: &Dirs,
    msl: &str,
    host: &str,
    label: &str,
) -> Result<(String, String)> {
    let dir = driver::scratch(dirs, &format!("{label}-metal"));
    let outcome = metal_kernel_in_dir(&dir, msl, host, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn metal_kernel_in_dir(
    dir: &Path,
    msl: &str,
    driver: &str,
    flags: &[String],
) -> Result<(String, String)> {
    let metalib = metal_kernel(dir, msl, flags)?;
    let exe = metal_driver_exe_from(dir, driver, "kernel_driver", "内核对拍")?;
    let mut command = Command::new(&exe);
    command.arg(&metalib);
    driver::run_exe(&mut command, &exe, "Metal 内核执行失败")
}

fn metal_driver_exe_from(
    dir: &Path,
    driver_source: &str,
    stem: &str,
    role: &str,
) -> Result<PathBuf> {
    let Some(compiler) = cutile::toolchain::detect() else {
        return Err(Error::new(
            Code::Unsupported,
            format!("缺少 host C 编译器：Metal {role}需要一个 host 驱动把内核发射到 GPU"),
        ));
    };
    let driver = dir.join(format!("{stem}.m"));
    let exe = dir.join(stem);
    driver::write_text(&driver, driver_source)?;
    let mut clang = Command::new(&compiler.command);
    clang
        .args(["-fobjc-arc", "-O2", "-o"])
        .arg(&exe)
        .arg(&driver);
    clang.args(["-framework", "Foundation", "-framework", "Metal"]);
    driver::run_tool(&compiler.command, &mut clang, &driver)?;
    Ok(exe)
}

fn xcrun(tool: &str, flags: &[String]) -> Command {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", "macosx"]).arg(tool).args(flags);
    command
}

fn metal_driver_exe(dir: &Path) -> Result<PathBuf> {
    metal_driver_exe_from(dir, &metal_driver(MAX_OUT), "driver", "见证")
}

fn metal_readback(exe: &Path, metalib: &Path, count: usize) -> Result<Readback> {
    let mut command = Command::new(exe);
    command.arg(metalib).arg(count.to_string());
    let (stdout, stderr) = driver::run_exe(&mut command, exe, "Metal 见证执行失败")?;
    Ok(driver::readback(&stdout, &stderr))
}

/// 见证驱动的 host 侧文本：把 `metallib` 里的 `cutile_witness` 发射到 GPU 并回读；缓冲宽度由
/// [`MAX_OUT`] 注入。供测试与 [`metal_driver_exe_from`] 共用，不重复写驱动文本。
pub fn metal_driver(max: usize) -> String {
    format!("#define WITNESS_MAX_OUT {max}\n{METAL_DRIVER}")
}

const METAL_DRIVER: &str = r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: driver <metallib> <count>\n");
            return 2;
        }
        const int count = atoi(argv[2]);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "本机没有 Metal 设备\n");
            return 3;
        }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            fprintf(stderr, "metallib 装载失败：%s\n", error.localizedDescription.UTF8String);
            return 4;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"cutile_witness"];
        if (!function) {
            fprintf(stderr, "metallib 里没有 cutile_witness\n");
            return 5;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String);
            return 6;
        }
        id<MTLBuffer> values = [device newBufferWithLength:(NSUInteger)count * WITNESS_MAX_OUT * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> counts = [device newBufferWithLength:(NSUInteger)count * sizeof(int)
                                                   options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:counts offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)count, 1, 1)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {
            fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String);
            return 7;
        }
        const int *written = (const int *)counts.contents;
        const float *out = (const float *)values.contents;
        for (int i = 0; i < count; ++i) {
            fprintf(stderr, "witness %d\n", written[i]);
        }
        fflush(stderr);
        for (int i = 0; i < count; ++i) {
            int n = written[i];
            if (n < 0) n = 0;
            if (n > WITNESS_MAX_OUT) n = WITNESS_MAX_OUT;
            for (int j = 0; j < n; ++j) {
                union { float f; unsigned int u; } v;
                v.f = out[i * WITNESS_MAX_OUT + j];
                printf("%u\n", v.u);
            }
        }
    }
    return 0;
}
"#;

// ---------------------------------------------------------------------------
// CUDA 后端
// ---------------------------------------------------------------------------

struct CudaBackend;

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
        Ok(Source::Single(crate::codegen::cuda::cuda_harness(
            kernel, signature, sample,
        )?))
    }
}

impl Driver for CudaBackend {
    fn available(&self, _: &Profile) -> bool {
        detect_nvcc().is_some() && (std::env::consts::OS != "windows" || find_cl().is_some())
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
    fn toolchain(&self) -> Option<Toolchain> {
        detect_nvcc()
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
    use cutile::backend::Lower::*;
    match kind {
        Pointwise => crate::codegen::cuda::cuda_grid_kernel(kernel, signature).map(|s| vec![s]),
        ChunkReduce { chunk, func } => {
            crate::codegen::cuda::cuda_chunk_reduce(kernel, signature, sample, *chunk, func)
                .map(|s| vec![s])
        }
        Loop(_) | ChunkScan { .. } | ChunkRmsnorm { .. } | ChunkSoftmax { .. } => Err(Error::new(
            Code::Unsupported,
            "CUDA 生产态 lowering 尚未覆盖该骨架".to_string(),
        )),
    }
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
         \x20   float local[{max}];\n\
         \x20   for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n\
         \x20   int n = 0;\n\
         \x20   switch (tid) {{\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n\n");
}

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
         \x20   }}\n"
    ));
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
    let outcome = cuda_in_dir(&dir, profile, program, label);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn cuda_in_dir(
    dir: &Path,
    profile: &Profile,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    let src = dir.join(format!("{label}.cu"));
    let exe = dir.join(format!("{label}.exe"));
    driver::write_text(&src, program)?;
    cuda_compile(&exe, &src, profile)?;
    let mut command = Command::new(&exe);
    driver::run_exe(&mut command, &exe, "CUDA 执行失败")
}

fn cuda_compile(exe: &Path, src: &Path, profile: &Profile) -> Result<()> {
    let mut command = Command::new("nvcc");
    command.args(&profile.flags);
    if std::env::consts::OS == "windows" {
        let Some(cl) = find_cl() else {
            return Err(Error::new(
                Code::Unsupported,
                "找不到 MSVC cl.exe：nvcc 设备编译需要它".to_string(),
            ));
        };
        command.arg("-ccbin").arg(cl);
    }
    command.arg("-o").arg(exe).arg(src);
    driver::run_tool("nvcc", &mut command, src)
}

// ---------------------------------------------------------------------------
// 装配
// ---------------------------------------------------------------------------

static METAL_BACKEND: MetalBackend = MetalBackend;
static CUDA_BACKEND: CudaBackend = CudaBackend;

/// 把 Metal / CUDA 后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    cutile::backend::register(&METAL_BACKEND);
    cutile::backend::register(&CUDA_BACKEND);
}

#[cfg(test)]
#[path = "../tests/backends_tests.rs"]
mod tests;
