//! 见证设备：一台统一的执行器，按 [`Emit`] 在描述符表里分发到对应编排行。
//!
//! 片段本体来自 `prim`、能力档案来自 `prof`，本层只消费：把见证源码拼进模板、编译、起子进程、
//! 回读对拍。各语言的差异全部是表内函数指针行（探测 / 程序生成 / 执行回读），没有按厂商各写一套
//! Rust 前端类型；新增设备只在 [`SPECS`] 加一行，不改任何 `match`。
//!
//! 共享的装载、命名、起进程与回读各只有一份；编译器探测在 [`crate::toolchain`]，编译/执行原语在
//! [`crate::runner`]。C / CUDA / Metal 三段编排在本文件内各占一节私有函数；用到设备时只按档案里的
//! [`Emit`] 选表行，不在调用侧再写一套后端代码。

use crate::atom::{Prim, Witness};
use crate::build;
use crate::dirs::Dirs;
use crate::prof::{Emit, Profile};
use crate::toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use diagnostic::{Code, Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 见证函数在 C 侧（C 与 CUDA 前端）的形参文本。
pub(crate) const C_PARAMS: &str = "float *";

/// 本次要拼进程序的见证源码：文件名 → 文本。
pub(crate) type Sources = BTreeMap<String, String>;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub(crate) type Readback = (Vec<i64>, Vec<f32>);

/// 一台设备的执行描述：差异全是表内的函数指针，运行期只多一次“按 emit 找行”，没有枚举匹配、没有虚表。
pub(crate) struct Spec {
    /// 这门语言。
    pub(crate) emit: Emit,
    /// 设备是否在本机。
    pub(crate) available: fn(&Profile) -> bool,
    /// 生成“跑一次全部见证”的程序文本；写出上限定在 ABI 常量 [`MAX_OUT`]。
    pub(crate) program: fn(&Profile, &Dirs, &[Selected<'_>], Scope) -> Result<String>,
    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    pub(crate) run: fn(&Profile, &Dirs, &str, usize, Scope) -> Result<Readback>,
}

/// 全部设备：**加设备就加一行**，不需要改任何 `match`。
const SPECS: &[Spec] = &[
    Spec {
        emit: Emit::C,
        available: c_available,
        program: c_program,
        run: c_run,
    },
    Spec {
        emit: Emit::Metal,
        available: metal_available,
        program: metal_program,
        run: metal_run,
    },
    Spec {
        emit: Emit::Cuda,
        available: cuda_available,
        program: cuda_program,
        run: cuda_run,
    },
];

/// 按 `emit` 找设备描述；找不到就是“本机没有这门语言的执行前端”。
pub(crate) fn spec(emit: Emit) -> Option<&'static Spec> {
    SPECS.iter().find(|entry| entry.emit == emit)
}

/// 已登记的全部执行语言：错误报告里列出“本机接了哪些”。
pub(crate) fn specs() -> &'static [Spec] {
    SPECS
}

// ---------------------------------------------------------------------------
// 共享装载、命名、起进程与回读
// ---------------------------------------------------------------------------

/// 一份见证源文件：读出来，并确认里面有这个前端能调用的函数（签名由编译器回答）。
/// 错误里带原语名，是因为同一个文件被多条见证共用时，光有文件名定位不了是哪一条。
fn witness_source(prim: &Prim, witness: &Witness, dirs: &Dirs, params: &str) -> Result<String> {
    let text = crate::witness::source(witness, dirs)
        .map_err(|error| error.with(format!("`{prim}` 的")))?;
    if !text.contains(&format!("{}(", witness.func)) {
        let message = format!(
            "`{prim}` 的见证 `{}` 里没有函数 `{}({params})`",
            witness.file, witness.func
        );
        return Err(Error::new(Code::Invalid, message));
    }
    Ok(text)
}

/// 本次执行的见证源码：按文件去重，同一文件里的多条见证只出现一次。
pub(crate) fn witness_files(
    selected: &[Selected<'_>],
    dirs: &Dirs,
    params: &str,
) -> Result<Sources> {
    let mut files = Sources::new();
    for (prim, _, witness) in selected {
        if files.contains_key(&witness.file) {
            continue;
        }
        files.insert(
            witness.file.clone(),
            witness_source(prim, witness, dirs, params)?,
        );
    }
    Ok(files)
}

/// 源码段接在头之后：每段补齐换行再空一行，段与段之间因此总有分隔。
pub(crate) fn push_sources(program: &mut String, files: &Sources) {
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
}

/// 让一段文本以换行结尾（空文本保持为空）：下一段可以直接接上。
pub(crate) fn normalize_newline(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// 写一个文件，并把它的上层目录建出来：一次性构建目录都只有一层。
pub(crate) fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", parent.display()))
        })?;
    }
    std::fs::write(path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))
}

/// 跑一条编译器命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
pub(crate) fn run_tool(name: &str, command: &mut Command, source: &Path) -> Result<()> {
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

/// 跑一个已经编好的可执行文件，返回 `(stdout, stderr)` 原文。非零退出码报 [`Code::Unsupported`]：编得过却跑不起来说明这台机器现在跑不了，不是见证写错了。
pub(crate) fn run_exe(
    command: &mut Command,
    path: &Path,
    failure: &str,
) -> Result<(String, String)> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("运行 {} 失败：{error}", path.display())))?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "{failure}（退出码 {:?}）：{}",
                output.status.code(),
                stderr.trim()
            ),
        ));
    }
    Ok((String::from_utf8_lossy(&output.stdout).to_string(), stderr))
}

/// 从 stderr 读出每条见证声明的写入个数（按调用顺序）。
pub(crate) fn parse_counts(stderr: &str) -> Vec<i64> {
    stderr
        .lines()
        .filter_map(|line| line.trim().strip_prefix("witness "))
        .filter_map(|value| value.trim().parse::<i64>().ok())
        .collect()
}

/// 回读解码：个数走 stderr（`witness N`），值走 stdout 的 f32 位模式；两条通道必须分开，
/// 否则 [`crate::runner::parse_bits`] 会把计数当成值解出来。
pub(crate) fn readback(stdout: &str, stderr: &str) -> Readback {
    (parse_counts(stderr), crate::runner::parse_bits(stdout))
}

/// 一个新的一次性目录：`<work_dir>/<tag>-<pid>-<序号>`，调用方负责删除。pid 隔开进程、序号
/// 隔开同进程内的多次调用：同时跑的两个 `cutile` 不会互相覆盖对方的中间产物。
pub(crate) fn scratch(dirs: &Dirs, tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    dirs.work().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// 两个后端 kernel 共享的分派段：按 `tid` 的 `switch`、cap 截断与回写循环。`close` 是 kernel
/// 收尾文本（Metal 与 CUDA 的换行数不同）：`local` 声明留在各自函数里，这里只拼逐字相同的部分。
pub(crate) fn push_dispatch(
    program: &mut String,
    max: usize,
    selected: &[Selected<'_>],
    close: &str,
) {
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

// ---------------------------------------------------------------------------
// C（host CPU）编排：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
// 它证明“这段 C 在这个 ABI 下编得过、跑得出 `expect`”，不证明设备具备该能力。
// ---------------------------------------------------------------------------

/// 自检用的 host C 参数：设备档案的 `[compile].flags` 在本机要么指向不存在的头，要么只对真机编译器有意义；它与设备执行是两个参数集。
pub(crate) const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// C 编排的编译输入：前置声明 + 参数。自检只走 host C ABI，理由见 [`HOST_FLAGS`]。
fn c_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (
            String::new(),
            HOST_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        ),
    }
}

fn c_available(_: &Profile) -> bool {
    // `c` 的设备就是本机 CPU：它永远在。
    true
}

fn c_program(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    scope: Scope,
) -> Result<String> {
    let mut program = c_head(profile, scope, selected);
    push_sources(&mut program, &witness_files(selected, dirs, C_PARAMS)?);
    program.push_str(&c_harness::<MAX_OUT>(selected));
    Ok(program)
}

fn c_run(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    _count: usize,
    scope: Scope,
) -> Result<Readback> {
    let (_, flags) = c_inputs(profile, scope);
    // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
    let tag = match scope {
        Scope::Device => "witness",
        Scope::Host => "self",
    };
    // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
    let (outcome, _artifact) =
        build::run_program(toolchain::detect().as_ref(), &flags, tag, program, dirs)?;
    // stdout 已由 `runner` 解码成 f32，这里只补上 stderr 里的写入个数。
    Ok((parse_counts(&outcome.stderr), outcome.bits))
}

/// C 前端的头：档案前置声明 + `stdio.h`；自检时再吞掉 MSL 的地址空间限定符。`#define thread` 只服务
/// 于自检，设备前端走真正的 `xcrun metal`，所以这里的放宽不会被当成设备证据。
fn c_head(profile: &Profile, scope: Scope, selected: &[Selected<'_>]) -> String {
    let (mut head, _) = c_inputs(profile, scope);
    let metal = selected
        .iter()
        .any(|(_, _, witness)| witness.file.ends_with(".metal"));
    normalize_newline(&mut head);
    head.push_str("#include <stdio.h>\n\n");
    if scope == Scope::Host && metal {
        head.push_str("#define thread\n");
    }
    head
}

/// C 前端的内核包装与主函数：`main()` 逐条调用见证函数，值走 stdout，个数走 stderr。
fn c_harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    let mut text = format!("\nint main(void) {{\n    float out[{OUT}];\n    int n;\n");
    // 每条见证是一句互不相同的调用语句，顺序由见证清单决定：这里只能逐条生成，不能批处理。
    for (_, _, witness) in selected {
        push_witness(&mut text, &witness.func, OUT);
    }
    text.push_str("    return 0;\n}\n");
    text
}

/// 一次见证调用：算一遍、报个数、按写出的个数打印位模式；`max` 是 const 泛型传下来的字面量，截断在生成的代码里是常数比较。
fn push_witness(text: &mut String, func: &str, max: usize) {
    text.push_str("    n = ");
    text.push_str(func);
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

// ---------------------------------------------------------------------------
// CUDA 真机编排：nvcc 把见证编进 `__global__` kernel，在 GPU 上按 tid 逐条执行，再 cudaMemcpy 回读。
//
// 与语法门 `GATES` 里 CUDA 行的主机语法自检不同，这里是真设备执行；它**不**证明别的架构或别的 `-arch` 下成立。
// 片段本体来自 `prim`，本层只做编译编排与回读。
// ---------------------------------------------------------------------------

fn cuda_available(_: &Profile) -> bool {
    // nvcc 在，且 Windows 上能找到宿主 cl（Linux 上 nvcc 用系统 gcc，不需要 cl）。
    toolchain::detect_nvcc().is_some()
        && (std::env::consts::OS != "windows" || toolchain::find_cl().is_some())
}

fn cuda_program(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    _scope: Scope,
) -> Result<String> {
    cuda_program_inner::<MAX_OUT>(profile, dirs, selected)
}

fn cuda_program_inner<const OUT: usize>(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
) -> Result<String> {
    let mut program = cuda_head(profile);
    let files = as_device(witness_files(selected, dirs, C_PARAMS)?);
    push_sources(&mut program, &files);
    push_cuda_kernel(&mut program, OUT, selected);
    push_cuda_main(&mut program, OUT, selected.len());
    Ok(program)
}

fn cuda_run(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    _count: usize,
    _scope: Scope,
) -> Result<Readback> {
    let (stdout, stderr) = compile_run_cuda(profile, dirs, program, "witness")?;
    Ok(readback(&stdout, &stderr))
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
    push_dispatch(program, max, selected, "}}\n\n");
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
         \x20   }}\n"
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
    let dir = scratch(dirs, &format!("{label}-cuda"));
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
    write_text(&src, program)?;
    cuda_compile(&exe, &src, profile)?;
    let mut command = Command::new(&exe);
    run_exe(&mut command, &exe, "CUDA 执行失败")
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
    run_tool("nvcc", &mut command, src)
}

// ---------------------------------------------------------------------------
// Metal 编排：MSL → `metallib` → host 驱动发射到 GPU → 回读。
//
// 不能用 host C 自检代替：MSL 没有默认地址空间，`float *out` 在 `.metal` 里是语法错误。
// 证明"这条见证在本机 Metal 设备上跑得出 `expect`"，**不**证明它在别的芯片、别的 `-std` 下同样成立。
// 片段本体来自 `prim`，本层只做两段编译、驱动发射与回读。
// ---------------------------------------------------------------------------

/// MSL 的形参必须带地址空间：MSL 没有默认地址空间，裸 `float *` 在那里是语法错误。
const MSL_PARAMS: &str = "thread float *";

fn metal_available(_: &Profile) -> bool {
    // 工具链在，就说明这台机器有 Metal 编译器；再往下才是"有没有 GPU"。
    toolchain::detect_metal().is_some()
}

fn metal_program(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    _scope: Scope,
) -> Result<String> {
    metal_program_inner::<MAX_OUT>(profile, dirs, selected)
}

fn metal_program_inner<const OUT: usize>(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
) -> Result<String> {
    let mut program = metal_head(profile);
    push_sources(&mut program, &witness_files(selected, dirs, MSL_PARAMS)?);
    push_metal_kernel(&mut program, OUT, selected);
    Ok(program)
}

fn metal_run(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    count: usize,
    _scope: Scope,
) -> Result<Readback> {
    // 设备参数来自档案：`-std=metal3.1` 之类决定了这段 MSL 按哪个语言版本编。
    run_metal(program, count, &profile.flags, dirs)
}

/// MSL 的头：档案的 `[compile].preamble`，补齐结尾换行后直接接见证源码。
fn metal_head(profile: &Profile) -> String {
    let mut head = profile.preamble.clone();
    normalize_newline(&mut head);
    head
}

/// 内核包装：`cutile_witness` 按 `tid` 分派到一条见证，把个数与值写回缓冲区。`local` 声明留在
/// 这里而不是公共分派段：Metal 的 `thread` 地址空间在 CUDA 里没有对应物。
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
    push_dispatch(program, max, selected, "}\n");
}

/// 编译并执行一次 Metal 见证程序，返回 `(每条写入个数, 按顺序回读的值)`。写、起子进程、回读三步分开；
/// 编不过与跑不起来都带完整输出，否则只会以一句 "compile failed" 出现。
fn run_metal(program: &str, count: usize, flags: &[String], dirs: &Dirs) -> Result<Readback> {
    let dir = scratch(dirs, "witness-metal");
    let outcome = metal_in_dir(&dir, program, count, flags);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

/// 一次性目录里的一条：写 MSL → 编成 `metallib` → 编驱动 → 跑 → 回读。
fn metal_in_dir(dir: &Path, program: &str, count: usize, flags: &[String]) -> Result<Readback> {
    let metalib = metal_kernel(dir, program, flags)?;
    let exe = metal_driver_exe(dir)?;
    metal_readback(&exe, &metalib, count)
}

/// 写 MSL 并编成 `metallib`：`xcrun metal` 管语言层、`metallib` 管链接；诊断里带 `.metal` 路径。
fn metal_kernel(dir: &Path, program: &str, flags: &[String]) -> Result<PathBuf> {
    let msl = dir.join("witness.metal");
    let air = dir.join("witness.air");
    let metalib = dir.join("witness.metallib");
    write_text(&msl, program)?;
    let mut compile = xcrun("metal", flags);
    compile.arg("-c").arg(&msl).arg("-o").arg(&air);
    run_tool("xcrun metal", &mut compile, &msl)?;
    let mut link = xcrun("metallib", &[]);
    link.arg(&air).arg("-o").arg(&metalib);
    run_tool("xcrun metallib", &mut link, &msl)?;
    Ok(metalib)
}

/// 一条 `xcrun -sdk macosx <tool> <flags>` 命令：Metal 的编译与链接都从这里起。
fn xcrun(tool: &str, flags: &[String]) -> Command {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", "macosx"]).arg(tool).args(flags);
    command
}

/// host 驱动：写 `driver.m`，用本机 clang 编成可执行文件（链接 Foundation 与 Metal）。驱动只负责
/// 把 `metallib` 发射出去并回读，不判断"算得对不对"；模板逐字不变是设备证据的前提。
fn metal_driver_exe(dir: &Path) -> Result<PathBuf> {
    let Some(compiler) = toolchain::detect() else {
        return Err(Error::new(
            Code::Unsupported,
            "缺少 host C 编译器：Metal 见证需要一个 host 驱动把内核发射到 GPU".to_string(),
        ));
    };
    let driver = dir.join("driver.m");
    let exe = dir.join("driver");
    write_text(&driver, &metal_driver(MAX_OUT))?;
    let mut clang = Command::new(&compiler.command);
    clang
        .args(["-fobjc-arc", "-O2", "-o"])
        .arg(&exe)
        .arg(&driver);
    clang.args(["-framework", "Foundation", "-framework", "Metal"]);
    run_tool(&compiler.command, &mut clang, &driver)?;
    Ok(exe)
}

/// 起驱动子进程并回读：`metallib` 与条数是它的两个参数。
fn metal_readback(exe: &Path, metalib: &Path, count: usize) -> Result<Readback> {
    let mut command = Command::new(exe);
    command.arg(metalib).arg(count.to_string());
    let (stdout, stderr) = run_exe(&mut command, exe, "Metal 见证执行失败")?;
    Ok(readback(&stdout, &stderr))
}

/// 把 `metallib` 里的 `cutile_witness` 发射到 GPU，回读个数与值。计数走 stderr、值走 stdout
/// ——与 C harness 同一口径，`compare` 因此只认一条路径；缓冲区用共享存储，看的是值不是带宽。
fn metal_driver(max: usize) -> String {
    format!("#define WITNESS_MAX_OUT {max}\n{METAL_DRIVER}")
}

/// 驱动主体：每条见证的缓冲宽度用 `WITNESS_MAX_OUT`，由 [`metal_driver`] 注入。这个数字曾经是 kernel 里的
/// const 泛型 `{max}` 与这里的三个裸字面量 `8`——跨 FFI 唯一失去编译期保护的重复：`MAX_OUT` 一旦不是 8，
/// 写入与回读就错位且不会有任何编译错误。
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

#[cfg(test)]
#[path = "device_tests.rs"]
mod tests;
