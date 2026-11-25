//! 设备后端契约：cutile 只依赖这条缝，不认识 Metal / CUDA / C 的具体实现。
//!
//! 本缝只保留「见证片段装配 + 编译执行回读」：设备算子正文已由 prof 档案里的设备语言片段给出，
//! 这里不再按骨架重写/降低算子算法。生产态设备执行走 `prof::runtime` FFI 后端岛。

use crate::driver;
use crate::platform::{
    ascend_device, detect_cce, detect_metal, detect_musa, detect_nvcc, find_cl, musa_device,
};
use crate::toolchain::Toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use cutile::dirs::Dirs;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub type Readback = (Vec<i64>, Vec<f32>);

/// 一段要编译并执行的程序源。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// 单源文本（C / CUDA）：编译成可执行文件直接跑。
    Single(String),
    /// 双源文本（Metal）：设备源（.metal）编成 metallib，host 驱动（.m）编成可执行文件并发射。
    Pair {
        /// 设备源（MSL）。
        device: String,
        /// host 驱动（ObjC）。
        host: String,
    },
}

/// 见证程序装配：把已选中的设备见证片段拼成「跑一次全部见证」的程序文本。
/// 不生成算子算法，只组合 prof 档案给出的设备正文。
pub trait Codegen {
    /// 这门语言（[`Emit`] 闭集）。
    fn emit(&self) -> Emit;
    /// 生成「跑一次全部见证」的程序文本；写出上限定在 ABI 常量 [`crate::witness::MAX_OUT`]。
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String>;
}

/// 编译 / 发射 / 回读：唯一碰工具链、设备 FFI 与子进程的一半，由设备后端实现。
pub trait Driver {
    /// 设备是否在本机（工具链 / 驱动就位）。
    fn available(&self, profile: &Profile) -> bool;
    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        max: usize,
        scope: Scope,
    ) -> Result<Readback>;
    /// 这条通道实测的工具链（进产物身份 / 报告）；拿不到（缺编译器）时为 `None`。
    fn toolchain(&self) -> Option<Toolchain>;
    /// 编译并执行一段程序源，返回 `(stdout, stderr)` 原文。
    fn compile_run(
        &self,
        _profile: &Profile,
        _dirs: &Dirs,
        _source: &Source,
        _label: &str,
    ) -> Result<(String, String)> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有编译车道".to_string(),
        ))
    }
}

/// 一台设备的执行后端：`Codegen` 装配见证 + `Driver` 跑源。
/// `Sync` 让后端能被写进进程级注册表（[`register`]）后从任意线程取用。
pub trait Backend: Codegen + Driver + Sync {}

/// 装配一台后端：L7（`cli`）在启动时注入 Metal / CUDA；cutile 已内置 host C 参考后端。
pub fn register(backend: &'static dyn Backend) {
    crate::device::register(backend);
}

// ---- 设备实现：配置表、单源后端与 Metal/OpenCL 执行 ----

/// 单源后端差异表；公共行为全部由 [`NativeBackend`] 复用。
struct NativeSpec {
    emit: Emit,
    tag: &'static str,
    extension: &'static str,
    includes: &'static str,
    device_attribute: &'static str,
    kernel: fn(&mut String, usize, &[Selected<'_>]),
    host: fn(&mut String, usize, usize),
    available: fn(&Profile) -> bool,
    toolchain: fn() -> Option<Toolchain>,
    compile: fn(&Path, &Path, &Profile, &Dirs) -> Result<()>,
    source_error: &'static str,
    run_error: &'static str,
}

struct NativeBackend {
    spec: &'static NativeSpec,
}

impl Codegen for NativeBackend {
    fn emit(&self) -> Emit {
        self.spec.emit
    }

    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> Result<String> {
        native_program(self.spec, profile, dirs, selected)
    }
}

impl Driver for NativeBackend {
    fn available(&self, profile: &Profile) -> bool {
        (self.spec.available)(profile)
    }

    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        let (stdout, stderr) = compile_run_native(self.spec, profile, dirs, program, "witness")?;
        Ok(driver::readback(&stdout, &stderr))
    }

    fn toolchain(&self) -> Option<Toolchain> {
        (self.spec.toolchain)()
    }

    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Single(text) => compile_run_native(self.spec, profile, dirs, text, label),
            Source::Pair { .. } => Err(Error::new(
                Code::Unsupported,
                self.spec.source_error.to_string(),
            )),
        }
    }
}

impl Backend for NativeBackend {}

fn native_program(
    spec: &NativeSpec,
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
) -> Result<String> {
    let mut program = witness_head(spec.includes, profile);
    let files = device_fn(
        driver::witness_files(selected, dirs, driver::C_PARAMS)?,
        spec.device_attribute,
    );
    driver::push_sources(&mut program, &files);
    (spec.kernel)(&mut program, MAX_OUT, selected);
    (spec.host)(&mut program, MAX_OUT, selected.len());
    Ok(program)
}

fn compile_run_native(
    spec: &NativeSpec,
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    driver::with_scratch(dirs, &format!("{label}-{}", spec.tag), |dir| {
        let src = dir.join(format!("{label}.{}", spec.extension));
        let exe = dir.join(format!("{label}.exe"));
        driver::write_text(&src, program)?;
        (spec.compile)(&exe, &src, profile, dirs)?;
        let mut command = Command::new(&exe);
        driver::run_exe(&mut command, &exe, spec.run_error)
    })
}

fn witness_head(includes: &str, profile: &Profile) -> String {
    let mut head = String::from(includes);
    if !profile.preamble.is_empty() {
        head.push_str(&profile.preamble);
        head.push('\n');
    }
    head
}

fn device_fn(files: driver::Sources, prefix: &str) -> driver::Sources {
    files
        .into_iter()
        .map(|(file, text)| {
            (
                file,
                text.replace("int witness", &format!("{prefix} int witness")),
            )
        })
        .collect()
}

fn witness_readback(max: usize) -> String {
    format!(
        "\x20   for (int t = 0; t < total; ++t) {{\n\
         \x20       for (int i = 0; i < h_counts[t]; ++i) {{\n\
         \x20           union {{ float f; unsigned int u; }} v;\n\
         \x20           v.f = h_out[t * {max} + i];\n\
         \x20           printf(\"%u\\n\", v.u);\n\
         \x20       }}\n\
         \x20       fprintf(stderr, \"witness %d\\n\", h_counts[t]);\n\
         \x20   }}\n"
    )
}

static CUDA_BACKEND: NativeBackend = NativeBackend { spec: &CUDA_SPEC };
static ASCEND_BACKEND: NativeBackend = NativeBackend { spec: &ASCEND_SPEC };
static METAL_BACKEND: MetalBackend = MetalBackend;
#[cfg(windows)]
static OPENCL_BACKEND: crate::device::opencl::OpenClBackend = crate::device::opencl::OpenClBackend;

static CUDA_SPEC: NativeSpec = NativeSpec {
    emit: Emit::Cuda,
    tag: "cuda",
    extension: "cu",
    includes: "#include <cuda_runtime.h>\n#include <cstdio>\n",
    device_attribute: "__device__",
    kernel: push_cuda_kernel,
    host: push_cuda_main,
    available: cuda_available,
    toolchain: cuda_toolchain,
    compile: cuda_compile,
    source_error: "CUDA 后端需要单源（.cu）",
    run_error: "CUDA 执行失败",
};

static ASCEND_SPEC: NativeSpec = NativeSpec {
    emit: Emit::Ascendc,
    tag: "ascend",
    extension: "cpp",
    includes: "#include \"acl/acl.h\"\n#include \"kernel_operator.h\"\n#include <cstdio>\n#include <cstdlib>\n",
    device_attribute: "__aicore__",
    kernel: push_ascend_kernel,
    host: push_ascend_main,
    available: |_| ascend_ready(),
    toolchain: detect_cce,
    compile: ascend_compile,
    source_error: "昇腾后端需要单源（host 驱动 + 核函数在同一份 .cpp）",
    run_error: "昇腾执行失败",
};

/// 将 Metal／OpenCL／CUDA／昇腾后端注册进进程级表；重复调用幂等。
pub fn install() {
    register(&METAL_BACKEND);
    #[cfg(windows)]
    register(&OPENCL_BACKEND);
    register(&CUDA_BACKEND);
    register(&ASCEND_BACKEND);
}

fn cuda_available(profile: &Profile) -> bool {
    match profile.meta.tool.as_str() {
        "nvcc" => nvcc_available(),
        "mcc" => detect_musa().is_some() && musa_device(),
        _ => false,
    }
}

/// Windows 的 NVCC 还依赖 MSVC 宿主编译器，其它系统只要求驱动工具链存在。
fn nvcc_available() -> bool {
    detect_nvcc().is_some() && (std::env::consts::OS != "windows" || find_cl().is_some())
}

fn cuda_toolchain() -> Option<Toolchain> {
    detect_nvcc().or_else(detect_musa)
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

fn cuda_compile(exe: &Path, src: &Path, profile: &Profile, dirs: &Dirs) -> Result<()> {
    let tool = profile.meta.tool.as_str();
    let mut command = Command::new(tool);
    command.args(crate::compile_flags(profile, dirs)?);
    if tool == "nvcc" && std::env::consts::OS == "windows" {
        let Some(cl) = find_cl() else {
            return Err(Error::new(
                Code::Unsupported,
                "找不到 MSVC cl.exe：nvcc 设备编译需要它".to_string(),
            ));
        };
        command.arg("-ccbin").arg(cl);
    }
    command.arg("-o").arg(exe).arg(src);
    driver::run_tool(tool, &mut command, src)
}

const ACL_GUARD: &str = "static int cutile_acl(const char *what, aclError code) {\n\
     \x20   if (code == ACL_SUCCESS) return 0;\n\
     \x20   fprintf(stderr, \"acl %s: %d\\n\", what, (int)code);\n\
     \x20   return -1;\n\
     }\n\
     #define CUTILE_ACL(call) do { if (cutile_acl(#call, (call)) != 0) return 1; } while (0)\n";

fn ascend_ready() -> bool {
    detect_cce().is_some() && ascend_device()
}

fn push_ascend_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "extern \"C\" __global__ __aicore__ void cutile_witness(GM_ADDR out_raw, GM_ADDR counts_raw,\n\
         \x20   int total) {{\n\
         \x20   const int tid = (int)AscendC::GetBlockIdx();\n\
         \x20   if (tid >= total) return;\n\
         \x20   __gm__ float *out = (__gm__ float *)out_raw;\n\
         \x20   __gm__ int *counts = (__gm__ int *)counts_raw;\n\
         \x20   float local[{max}];\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n\n");
}

fn push_ascend_main(program: &mut String, max: usize, total: usize) {
    program.push_str(ACL_GUARD);
    program.push_str(&format!(
        "\nint main() {{\n\
         \x20   const int total = {total};\n\
         \x20   const size_t out_bytes = (size_t)total * {max} * sizeof(float);\n\
         \x20   const size_t cnt_bytes = (size_t)total * sizeof(int);\n\
         \x20   CUTILE_ACL(aclInit(nullptr));\n\
         \x20   CUTILE_ACL(aclrtSetDevice(0));\n\
         \x20   aclrtStream stream = nullptr;\n\
         \x20   CUTILE_ACL(aclrtCreateStream(&stream));\n\
         \x20   uint8_t *d_out = nullptr;\n\
         \x20   uint8_t *d_counts = nullptr;\n\
         \x20   CUTILE_ACL(aclrtMalloc((void **)&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n\
         \x20   CUTILE_ACL(aclrtMalloc((void **)&d_counts, cnt_bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n\
         \x20   cutile_witness<<<total, nullptr, stream>>>((GM_ADDR)d_out, (GM_ADDR)d_counts, total);\n\
         \x20   CUTILE_ACL(aclrtSynchronizeStream(stream));\n\
         \x20   float *h_out = (float *)malloc(out_bytes);\n\
         \x20   int *h_counts = (int *)malloc(cnt_bytes);\n\
         \x20   if (h_out == nullptr || h_counts == nullptr) {{\n\
         \x20       fprintf(stderr, \"host malloc\\n\");\n\
         \x20       return 1;\n\
         \x20   }}\n\
         \x20   CUTILE_ACL(aclrtMemcpy(h_out, out_bytes, d_out, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST));\n\
         \x20   CUTILE_ACL(aclrtMemcpy(h_counts, cnt_bytes, d_counts, cnt_bytes, ACL_MEMCPY_DEVICE_TO_HOST));\n"
    ));
    program.push_str(&witness_readback(max));
    program.push_str(
        "\x20   free(h_out); free(h_counts);\n\
         \x20   CUTILE_ACL(aclrtFree(d_out));\n\
         \x20   CUTILE_ACL(aclrtFree(d_counts));\n\
         \x20   CUTILE_ACL(aclrtDestroyStream(stream));\n\
         \x20   CUTILE_ACL(aclrtResetDevice(0));\n\
         \x20   CUTILE_ACL(aclFinalize());\n\
         \x20   return 0;\n\
         }\n",
    );
}

fn ascend_compile(exe: &Path, src: &Path, profile: &Profile, dirs: &Dirs) -> Result<()> {
    let tool = profile.meta.tool.as_str();
    let (links, options): (Vec<String>, Vec<String>) = crate::compile_flags(profile, dirs)?
        .into_iter()
        .partition(|flag| flag.starts_with("-l"));
    let mut command = Command::new(tool);
    command
        .args(options)
        .arg(src)
        .arg("-o")
        .arg(exe)
        .args(links);
    driver::run_tool(tool, &mut command, src)
}

pub(crate) struct MetalBackend;

impl Codegen for MetalBackend {
    fn emit(&self) -> Emit {
        Emit::Metal
    }

    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> Result<String> {
        let mut program = profile.preamble.clone();
        driver::normalize_newline(&mut program);
        driver::push_sources(
            &mut program,
            &driver::witness_files(selected, dirs, "thread float *")?,
        );
        program.push_str(&format!(
            "kernel void cutile_witness(device float *out [[buffer(0)]],\n\
             \x20                          device int *counts [[buffer(1)]],\n\
             \x20                          uint tid [[thread_position_in_grid]]) {{\n\
             \x20   thread float local[{MAX_OUT}];\n"
        ));
        driver::push_dispatch(&mut program, MAX_OUT, selected, "}\n");
        Ok(program)
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

fn run_metal(program: &str, count: usize, flags: &[String], dirs: &Dirs) -> Result<Readback> {
    driver::with_scratch(dirs, "witness-metal", |dir| {
        let metalib = metal_kernel(dir, program, flags)?;
        let exe = metal_driver_exe_from(dir, &metal_driver(MAX_OUT), "driver", "见证")?;
        let mut command = Command::new(&exe);
        command.arg(&metalib).arg(count.to_string());
        let (stdout, stderr) = driver::run_exe(&mut command, &exe, "Metal 见证执行失败")?;
        Ok(driver::readback(&stdout, &stderr))
    })
}

/// 将 MSL 源文本编译为 `metallib`，返回库路径。
///
/// # Errors
///
/// 源文件不可写，或 `metal`／`metallib` 编译失败。
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

/// 编译并执行一次 Metal 双源程序，返回 `(stdout, stderr)`。
///
/// # Errors
///
/// 缺 host 编译器、设备工具链失败、编译失败或执行失败。
pub fn compile_run_metal(
    profile: &Profile,
    dirs: &Dirs,
    msl: &str,
    host: &str,
    label: &str,
) -> Result<(String, String)> {
    driver::with_scratch(dirs, &format!("{label}-metal"), |dir| {
        let metalib = metal_kernel(dir, msl, &profile.flags)?;
        let exe = metal_driver_exe_from(dir, host, "kernel_driver", "内核对拍")?;
        let mut command = Command::new(&exe);
        command.arg(&metalib);
        driver::run_exe(&mut command, &exe, "Metal 内核执行失败")
    })
}

fn metal_driver_exe_from(dir: &Path, source: &str, stem: &str, role: &str) -> Result<PathBuf> {
    let Some(compiler) = crate::toolchain::detect() else {
        return Err(Error::new(
            Code::Unsupported,
            format!("缺少 host C 编译器：Metal {role}需要一个 host 驱动把内核发射到 GPU"),
        ));
    };
    let driver = dir.join(format!("{stem}.m"));
    let exe = dir.join(stem);
    driver::write_text(&driver, source)?;
    let mut clang = Command::new(&compiler.command);
    clang
        .args(["-fobjc-arc", "-O2", "-o"])
        .arg(&exe)
        .arg(&driver)
        .args(["-framework", "Foundation", "-framework", "Metal"]);
    driver::run_tool(&compiler.command, &mut clang, &driver)?;
    Ok(exe)
}

fn xcrun(tool: &str, flags: &[String]) -> Command {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", "macosx"]).arg(tool).args(flags);
    command
}

pub use crate::device::metal_driver;

#[cfg(test)]
#[path = "../tests/backends.rs"]
mod tests;
