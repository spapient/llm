//! 具体设备：C / Metal / CUDA 的程序生成与工具链执行。

use crate::backend;
use crate::build;
use crate::prof::Profile;
use crate::witness::source;
use crate::witness::{Device, Language, MAX_OUT, Scope, Selected};
use diagnostic::{Code, Error};
use llm_config::Settings;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// C 前端：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
///
/// 语言身份（`Emit::C` / `c` / `witness`）由 `#[frontend]` 生成，`EXT` 与 `Emit::ext()` 的
/// 一致性由注解展开出的 `const _` 断言在编译期检查；本实现只写"怎么跑"。
#[annotation::frontend(emit = "c", ext = "c", tag = "witness")]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn available(_profile: &Profile) -> bool {
        // `c` 的设备就是本机 CPU：它永远在。
        true
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        settings: &Settings,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> diagnostic::Result<String> {
        let (preamble, _) = c_compile_inputs(profile, scope);
        let mut sources: BTreeMap<String, String> = BTreeMap::new();
        for (prim, _, witness) in selected {
            if sources.contains_key(&witness.file) {
                continue;
            }
            let text =
                source(witness, settings).map_err(|error| error.with(format!("`{prim}` 的")))?;
            if !text.contains(&format!("{}(", witness.func)) {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "`{prim}` 的见证 `{}` 里没有函数 `{}(float *)`",
                        witness.file, witness.func
                    ),
                ));
            }
            sources.insert(witness.file.clone(), text);
        }

        let mut program = preamble;
        if !program.is_empty() && !program.ends_with('\n') {
            program.push('\n');
        }
        program.push_str("#include <stdio.h>\n\n");
        // 自检用 host C 前端：MSL 的地址空间限定符在 C 里没有对应物，`thread` 必须能被忽略。
        // 只影响自检——设备前端走真正的 `xcrun metal`，不经过这一行。
        if scope == Scope::Host
            && selected
                .iter()
                .any(|(_, _, witness)| witness.file.ends_with(".metal"))
        {
            program.push_str("#define thread\n");
        }
        for text in sources.values() {
            program.push_str(text);
            if !program.ends_with('\n') {
                program.push('\n');
            }
            program.push('\n');
        }
        program.push_str(&harness::<OUT>(selected));
        Ok(program)
    }

    fn run(
        profile: &Profile,
        settings: &Settings,
        program: &str,
        _count: usize,
        scope: Scope,
    ) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
        let (_, flags) = c_compile_inputs(profile, scope);
        // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
        let tag = match scope {
            Scope::Device => Self::TAG,
            Scope::Host => "self",
        };
        // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
        let (outcome, _artifact) =
            build::run_program(backend::detect().as_ref(), &flags, tag, program, settings)?;
        Ok((parse_counts(&outcome.stderr), outcome.bits))
    }
}

/// Metal 前端：MSL → `metallib` → host 驱动发射到 GPU → 回读。
///
/// 为什么不能用 host C 自检代替：MSL 没有默认地址空间，`float *out` 在 `.metal` 里是语法错误，
/// 而 `cc` 只因为源码是纯 C 才放行——验证的是"这段 C 能编译"，不是"这条 Metal 见证存在"。
/// 本实现把同一个文件交给真正的 Metal 编译器，并让它在 GPU 上跑一遍。
///
/// 语言身份（`Emit::Metal` / `metal` / `witness`）由 `#[frontend]` 生成，本实现只写"怎么跑"。
#[annotation::frontend(emit = "metal", ext = "metal", tag = "witness")]
pub struct MetalDevice;

impl Device for MetalDevice {
    fn available(_profile: &Profile) -> bool {
        // 工具链在，就说明这台机器有 Metal 编译器；再往下才是"有没有 GPU"。
        metal_toolchain().is_some()
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        settings: &Settings,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> diagnostic::Result<String> {
        metal_program::<OUT>(profile, settings, selected)
    }

    fn run(
        profile: &Profile,
        settings: &Settings,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
        // 设备参数来自档案：`-std=metal3.1` 之类决定了这段 MSL 按哪个语言版本编。
        run_metal(program, count, &profile.flags, settings)
    }
}

/// MSL 源：档案的 `[compile].preamble` + 见证源码 + 一个按 `tid` 分派的 kernel。
///
/// 每条见证一个 thread：`tid` 是见证在本次执行里的下标，写出的个数与值分别落到 `counts` 与
/// `out[tid * OUT ..]`。`OUT` 是 const 泛型参数，生成的 MSL 里所有缓冲区尺寸都是字面量。
fn metal_program<const OUT: usize>(
    profile: &Profile,
    settings: &Settings,
    selected: &[Selected<'_>],
) -> diagnostic::Result<String> {
    let max = OUT;
    let mut program = profile.preamble.clone();
    if !program.is_empty() && !program.ends_with('\n') {
        program.push('\n');
    }
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for (prim, _, witness) in selected {
        if files.contains_key(&witness.file) {
            continue;
        }
        let text = source(witness, settings).map_err(|error| error.with(format!("`{prim}` 的")))?;
        if !text.contains(&format!("{}(", witness.func)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 里没有函数 `{}(thread float *)`",
                    witness.file, witness.func
                ),
            ));
        }
        files.insert(witness.file.clone(), text);
    }
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }

    program.push_str(&format!(
        "kernel void cutile_witness(device float *out [[buffer(0)]],\n\
         \x20                          device int *counts [[buffer(1)]],\n\
         \x20                          uint tid [[thread_position_in_grid]]) {{\n\
         \x20   thread float local[{max}];\n\
         \x20   for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n\
         \x20   int n = 0;\n\
         \x20   switch (tid) {{\n"
    ));
    push_dispatch(&mut program, max, selected, "}\n");
    Ok(program)
}

/// 两个后端 kernel 共享的分派段：按 `tid` 的 `switch`、cap 截断与回写循环。
///
/// `close` 是 kernel 收尾文本（Metal 与 CUDA 的换行数不同）；`local` 声明因 Metal 带
/// `thread`、CUDA 不带而留在各自函数里，这里只生成两者逐字相同的部分。
fn push_dispatch(program: &mut String, max: usize, selected: &[Selected<'_>], close: &str) {
    for (index, (_, _, witness)) in selected.iter().enumerate() {
        program.push_str(&format!(
            "        case {index}: n = {}(local); break;\n",
            witness.func
        ));
    }
    program.push_str(&format!(
        "        default: break;\n\
         \x20   }}\n\
         \x20   int cap = n < 0 ? 0 : (n > {max} ? {max} : n);\n\
         \x20   counts[tid] = cap;\n\
         \x20   for (int i = 0; i < {max}; ++i) out[tid * {max} + i] = i < cap ? local[i] : 0.0f;\n\
         {close}"
    ));
}

/// 一个新的一次性目录：`<work_dir>/<tag>-<pid>-<序号>`，调用方负责删除。
///
/// pid 隔开进程、序号隔开同进程内的多次调用：同时跑的两个 `cutile` 不会互相覆盖对方的中间产物。
pub(crate) fn scratch(settings: &Settings, tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    settings.work_dir().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// 编译并执行一次 Metal 见证程序，返回 `(每条写入个数, 按顺序回读的值)`。
///
/// 三处调用都要看得见失败原因：`xcrun metal`（语言层）、`metallib`（链接）、`clang`（驱动）；
/// 驱动自身再分"没有 Metal 设备"与"运行失败"两种。诊断带上完整输出与源文件路径，否则
/// `float *out` 这种错误只会以一句 "compile failed" 出现。
fn run_metal(
    program: &str,
    count: usize,
    flags: &[String],
    settings: &Settings,
) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
    let dir = scratch(settings, "witness-metal");
    let outcome = (|| {
        std::fs::create_dir_all(&dir).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", dir.display()))
        })?;
        let msl = dir.join("witness.metal");
        let air = dir.join("witness.air");
        let metalib = dir.join("witness.metallib");
        let driver = dir.join("driver.m");
        let exe = dir.join("driver");
        std::fs::write(&msl, program)
            .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", msl.display())))?;

        run_tool(
            "xcrun metal",
            std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metal"])
                .args(flags)
                .arg("-c")
                .arg(&msl)
                .arg("-o")
                .arg(&air),
            &msl,
        )?;
        run_tool(
            "xcrun metallib",
            std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metallib"])
                .arg(&air)
                .arg("-o")
                .arg(&metalib),
            &msl,
        )?;

        let compiler = backend::detect().ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                "缺少 host C 编译器：Metal 见证需要一个 host 驱动把内核发射到 GPU".to_string(),
            )
        })?;
        std::fs::write(&driver, metal_driver(MAX_OUT)).map_err(|error| {
            Error::new(Code::Io, format!("写 {} 失败：{error}", driver.display()))
        })?;
        run_tool(
            &compiler.command,
            std::process::Command::new(&compiler.command)
                .args(["-fobjc-arc", "-O2", "-o"])
                .arg(&exe)
                .arg(&driver)
                .args(["-framework", "Foundation", "-framework", "Metal"]),
            &driver,
        )?;

        let run = std::process::Command::new(&exe)
            .arg(&metalib)
            .arg(count.to_string())
            .output()
            .map_err(|error| {
                Error::new(Code::Io, format!("运行 {} 失败：{error}", exe.display()))
            })?;
        if !run.status.success() {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "Metal 见证执行失败（退出码 {:?}）：{}",
                    run.status.code(),
                    String::from_utf8_lossy(&run.stderr).trim()
                ),
            ));
        }
        let stderr = String::from_utf8_lossy(&run.stderr).to_string();
        Ok((
            parse_counts(&stderr),
            backend::parse_bits(&String::from_utf8_lossy(&run.stdout)),
        ))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

/// CUDA 真机前端：nvcc 把见证编进 `__global__` kernel，在 GPU 上按 tid 逐条执行，
/// 再 cudaMemcpy 回读。与 `CudaFront`（借 clang 的主机语法自检）不同，这里是真设备执行。
#[annotation::frontend(emit = "cuda", ext = "cu", tag = "witness")]
pub struct CudaDevice;

impl Device for CudaDevice {
    fn available(_profile: &Profile) -> bool {
        // nvcc 在，且 Windows 上能找到宿主 cl（Linux 上 nvcc 用系统 gcc，不需要 cl）。
        backend::detect_nvcc().is_some()
            && (std::env::consts::OS != "windows" || find_cl().is_some())
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        settings: &Settings,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> diagnostic::Result<String> {
        cuda_program::<OUT>(profile, settings, selected)
    }

    fn run(
        profile: &Profile,
        settings: &Settings,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
        run_cuda(profile, settings, program, count)
    }
}

/// 定位 MSVC cl.exe（Windows 上 nvcc 的宿主编译器）：先问 `vswhere`，再退回环境变量登记的两个安装根。
///
/// 先前这里写死了五个安装根（`D:\BuildTools` 之类），那只对某台开发机成立：标准安装
/// （`C:\Program Files\Microsoft Visual Studio\2022\Community`）反而找不到，而 `sort()` + `pop()`
/// 按字符串序会把 `9.29…` 排到 `17.10…` 之后，多版本共存时选中更旧的一套。
pub(crate) fn find_cl() -> Option<PathBuf> {
    if std::env::consts::OS != "windows" {
        return None;
    }
    candidates().into_iter().max_by_key(|cl| msvc_version(cl))
}

/// 全部候选 cl.exe：`vswhere`（微软官方的安装发现器）枚举的，加环境变量里两个安装根下的。
fn candidates() -> Vec<PathBuf> {
    let mut found = vswhere_cls();
    for key in ["VCINSTALLDIR", "VSINSTALLDIR"] {
        if let Some(root) = std::env::var_os(key) {
            found.extend(msvc_cls(Path::new(&root)));
        }
    }
    found
}

/// `vswhere -find` 枚举的全部 cl.exe：它知道安装在哪，不需要本文件猜盘符。
fn vswhere_cls() -> Vec<PathBuf> {
    let Some(program_files) = std::env::var_os("ProgramFiles(x86)") else {
        return Vec::new();
    };
    let vswhere = Path::new(&program_files)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let Ok(output) = std::process::Command::new(vswhere)
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

/// 一个安装根下的全部 cl.exe（`<root>/Tools/MSVC/<版本>/bin/Hostx64/x64/cl.exe`）。
///
/// 两个根形状都看：`VCINSTALLDIR` 直接指到 `…\VC`，`VSINSTALLDIR` 指到安装根（下面还有 `VC`）。
fn msvc_cls(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for base in [
        root.join("Tools").join("MSVC"),
        root.join("VC").join("Tools").join("MSVC"),
    ] {
        let Ok(entries) = std::fs::read_dir(base) else {
            continue;
        };
        found.extend(entries.flatten().map(|entry| {
            entry
                .path()
                .join("bin")
                .join("Hostx64")
                .join("x64")
                .join("cl.exe")
        }));
    }
    found.into_iter().filter(|cl| cl.is_file()).collect()
}

/// cl.exe 路径里 `MSVC\<版本>\` 的那一段，按**数值**比较。
///
/// 按字符串找标记而不是 `Path::components`：`\` 只在 Windows 上是分隔符，而本文件的单元测试
/// 跑在别的平台上（那里整条路径只有一段），用组件切什么都验不到。
///
/// 解析不出版本号时返回空表（在 `Vec` 的字典序里最小），于是带版本号的候选总是优先。
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

/// CUDA 源：runtime 头 + 档案 preamble + `__device__` 见证函数 + `__global__` 分派 kernel + host main。
fn cuda_program<const OUT: usize>(
    profile: &Profile,
    settings: &Settings,
    selected: &[Selected<'_>],
) -> diagnostic::Result<String> {
    let max = OUT;
    let total = selected.len();
    let mut program = String::new();
    program.push_str("#include <cuda_runtime.h>\n#include <cstdio>\n");
    if !profile.preamble.is_empty() {
        program.push_str(&profile.preamble);
        program.push('\n');
    }
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for (prim, _, witness) in selected {
        if files.contains_key(&witness.file) {
            continue;
        }
        let text = source(witness, settings).map_err(|error| error.with(format!("`{prim}` 的")))?;
        if !text.contains(&format!("{}(", witness.func)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 里没有函数 `{}(float *)`",
                    witness.file, witness.func
                ),
            ));
        }
        // 见证函数要在 kernel 里调用：给定义加 `__device__`（"int witness" 只出现在定义处）。
        files.insert(
            witness.file.clone(),
            text.replace("int witness", "__device__ int witness"),
        );
    }
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
    program.push_str(&format!(
        "__global__ void cutile_witness(float *out, int *counts, int total) {{\n\
         \x20   int tid = blockIdx.x;\n\
         \x20   if (tid >= total) return;\n\
         \x20   float local[{max}];\n\
         \x20   for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n\
         \x20   int n = 0;\n\
         \x20   switch (tid) {{\n"
    ));
    push_dispatch(&mut program, max, selected, "}}\n\n");
    // stdout 只按实际写入个数打印位模式（与 CPU harness 同口径），counts 走 stderr。
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
    Ok(program)
}

/// 编译并运行任意 CUDA 程序，返回 `(stdout, stderr)` 原文：临时目录、nvcc（Windows 用
/// `-ccbin` 指向 cl）、执行。见证与算子 verify 共用这一条，避免两套 nvcc 调用。
pub fn compile_run_cuda(
    profile: &Profile,
    settings: &Settings,
    program: &str,
    label: &str,
) -> diagnostic::Result<(String, String)> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = settings.work_dir().join(format!(
        "{label}-cuda-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let outcome = (|| {
        std::fs::create_dir_all(&dir).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", dir.display()))
        })?;
        let src = dir.join(format!("{label}.cu"));
        let exe = dir.join(format!("{label}.exe"));
        std::fs::write(&src, program)
            .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", src.display())))?;
        let mut command = std::process::Command::new("nvcc");
        command.args(&profile.flags);
        if std::env::consts::OS == "windows" {
            let cl = find_cl().ok_or_else(|| {
                Error::new(
                    Code::Unsupported,
                    "找不到 MSVC cl.exe：nvcc 设备编译需要它".to_string(),
                )
            })?;
            command.arg("-ccbin").arg(&cl);
        }
        command.arg("-o").arg(&exe).arg(&src);
        run_tool("nvcc", &mut command, &src)?;
        let run = std::process::Command::new(&exe).output().map_err(|error| {
            Error::new(Code::Io, format!("运行 {} 失败：{error}", exe.display()))
        })?;
        if !run.status.success() {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "CUDA 执行失败（退出码 {:?}）：{}",
                    run.status.code(),
                    String::from_utf8_lossy(&run.stderr).trim()
                ),
            ));
        }
        Ok((
            String::from_utf8_lossy(&run.stdout).to_string(),
            String::from_utf8_lossy(&run.stderr).to_string(),
        ))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

/// 编译并执行 CUDA 见证：Windows 上用 `-ccbin` 显式指向 cl.exe，不依赖 vcvars/PATH。
fn run_cuda(
    profile: &Profile,
    settings: &Settings,
    program: &str,
    _count: usize,
) -> diagnostic::Result<(Vec<i64>, Vec<f32>)> {
    let (stdout, stderr) = compile_run_cuda(profile, settings, program, "witness")?;
    Ok((parse_counts(&stderr), backend::parse_bits(&stdout)))
}

/// 跑一条工具命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
fn run_tool(
    name: &str,
    command: &mut std::process::Command,
    source: &Path,
) -> diagnostic::Result<()> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("调用 `{name}` 失败：{error}")))?;
    if output.status.success() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "`{name}` 执行失败：\n{}\n{}\n源文件：{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
            source.display()
        ),
    ))
}

/// 从 stderr 读出每条见证声明的写入个数（按调用顺序）。
pub(crate) fn parse_counts(stderr: &str) -> Vec<i64> {
    stderr
        .lines()
        .filter_map(|line| line.trim().strip_prefix("witness "))
        .filter_map(|value| value.trim().parse::<i64>().ok())
        .collect()
}

/// 把 `metallib` 里的 `cutile_witness` 发射到 GPU，回读个数与值。
///
/// 计数走 stderr、值走 stdout——与 C harness 同一口径，`compare` 因此只认一条路径。
/// 缓冲区用 `MTLResourceStorageModeShared`：要看的是值，不是带宽。
fn metal_driver(max: usize) -> String {
    format!("#define WITNESS_MAX_OUT {max}\n{METAL_DRIVER}")
}

/// 驱动主体：每条见证的缓冲宽度用 `WITNESS_MAX_OUT`，由 [`metal_driver`] 注入。
///
/// 这个数字曾经在 MSL kernel 里是 const 泛型 `{max}`、在这里是三个裸字面量 `8`——跨 FFI 的
/// 唯一一处失去编译期保护的重复：`Witness::MAX_OUT` 一旦不是 8，kernel 按 `tid * MAX_OUT + i`
/// 写、驱动按 `i * 8 + j` 读，直接错位且不会有任何编译错误。
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

/// 本机的 clang：先问 `clang`，再退到 `cc`（macOS 与多数发行版的 `cc` 就是 clang）。
///
/// 判据是“这个程序是不是 clang”，不是“有没有叫 clang 的文件”：`-x cuda` / `-x cl` 是
/// clang 的前端，gcc 没有。
pub(crate) fn clang() -> Option<backend::Toolchain> {
    backend::probe("clang")
        .or_else(|| backend::detect().filter(|toolchain| toolchain.banner.contains("clang")))
}

/// `xcrun metal` 的实际工具链：命令可用、能自报版本时才算数。
///
/// 不用 `[meta].tool` 判断：那是人写的登记值（`xcrun metal`），而这里要回答的是
/// "这台机器现在能不能把 `.metal` 编成 `metallib`"。
pub(crate) fn metal_toolchain() -> Option<backend::Toolchain> {
    let output = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    Some(backend::Toolchain {
        command: "xcrun metal".to_string(),
        version: backend::parse_version(&text),
        banner: text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())?
            .to_string(),
    })
}

/// 自检用的 host C 参数。
///
/// 设备档案的 `[compile].flags` 在这台机器上要么不存在（`#include <cuda_fp16.h>`），要么
/// 只对真机编译器有意义（`-arch=sm_100a`），所以自检固定用这一套。它与设备执行的那一套
/// 是两个不同的编译参数集，报告的 `[证据]` 段因此分开写。
pub(crate) const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// C 前端的编译输入：前置声明 + 参数。
///
/// 自检只走 host C ABI：设备的 `[compile]` 前置声明与参数在这台机器上要么不存在
/// （`#include <cuda_fp16.h>`），要么只对真机编译器有意义（`-arch=sm_100a`）。
pub(crate) fn c_compile_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (
            String::new(),
            HOST_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        ),
    }
}

/// 生成 `main()`：逐条调用见证函数，值走 stdout，个数走 stderr。
///
/// 个数必须单独走一条通道：stdout 只放 f32 位模式，`backend::parse_bits` 才能原样解码；
/// 把计数混进 stdout 会让"写了几个"与"值是多少"互相污染。
pub(crate) fn harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    // `OUT` 是 const 泛型参数：`max` 在这个实例里就是一个常量，生成的代码里所有上限都是字面量。
    let max = OUT;
    let mut text = format!("\nint main(void) {{\n    float out[{max}];\n    int n;\n");
    for (_, _, witness) in selected {
        text.push_str("    n = ");
        text.push_str(&witness.func);
        text.push_str("(out);\n");
        text.push_str("    fprintf(stderr, \"witness %d\\n\", n);\n");
        text.push_str("    if (n < 0) n = 0;\n");
        text.push_str(&format!("    if (n > {max}) n = {max};\n"));
        text.push_str("    for (int i = 0; i < n; ++i) {\n");
        text.push_str("        union { float f; unsigned int u; } v;\n");
        text.push_str("        v.f = out[i];\n");
        text.push_str("        printf(\"%u\\n\", v.u);\n");
        text.push_str("    }\n");
    }
    text.push_str("    return 0;\n}\n");
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_msvc_wins_by_version_not_by_string_order() {
        let candidates = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\9.29.30133\bin\Hostx64\x64\cl.exe",
            r"D:\BuildTools\VC\Tools\MSVC\17.10.0\bin\Hostx64\x64\cl.exe",
        ];
        let newest = candidates
            .into_iter()
            .max_by_key(|cl| msvc_version(Path::new(cl)))
            .expect("候选非空");
        assert!(
            newest.contains("17.10.0"),
            "字符串序会把 9.29 排到 17.10 之后：{newest}"
        );
    }

    #[test]
    fn a_path_without_a_version_loses_to_one_with_it() {
        assert!(msvc_version(Path::new(r"C:\msvc\cl.exe")).is_empty());
        assert_eq!(
            msvc_version(Path::new(
                r"C:\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
            )),
            vec![14, 29, 30133]
        );
    }

    #[test]
    fn the_metal_driver_takes_its_width_from_the_witness_limit() {
        let driver = metal_driver(MAX_OUT);
        assert!(driver.contains(&format!("#define WITNESS_MAX_OUT {MAX_OUT}")));
        // 一处定义 + 四处使用（上限判断那行出现两次）：每个宽度用点都必须走宏。
        assert_eq!(
            driver.matches("WITNESS_MAX_OUT").count(),
            5,
            "驱动里还有写死的宽度"
        );
        for literal in ["count * 8", "out[i * 8", "(n > 8)"] {
            assert!(!driver.contains(literal), "仍是字面量：{literal}");
        }
    }

    #[test]
    fn scratch_directories_do_not_collide() {
        let settings =
            Settings::<llm_config::Cutile>::defaults(&llm_config::Roots::new(std::env::temp_dir()));
        let first = scratch(&settings, "gate");
        let second = scratch(&settings, "gate");
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(settings.work_dir().as_path()));
    }
}
