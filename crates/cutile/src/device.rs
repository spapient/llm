//! 具体设备：C / Metal / CUDA 的程序生成与工具链执行。
//!
//! 每个前端分三段：**头**（前置声明）、**内核包装**（分派与回读）、**主函数**（拼成文本）；
//! 三段共用的装载、命名、起进程与回读各只有一份，算不算证据由 [`crate::witness`] 决定。

use crate::atom::{Prim, Witness};
use crate::backend;
use crate::build;
use crate::dirs::Dirs;
use crate::prof::Profile;
use crate::witness::source;
use crate::witness::{Device, Language, MAX_OUT, Scope, Selected};
use diagnostic::{Code, Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// 见证函数在 C 侧（C 与 CUDA 前端）的形参文本。
const C_PARAMS: &str = "float *";

/// MSL 的形参必须带地址空间：MSL 没有默认地址空间，裸 `float *` 在那里是语法错误。
const MSL_PARAMS: &str = "thread float *";

/// 本次要拼进程序的见证源码：文件名 → 文本。
type Sources = BTreeMap<String, String>;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
type Readback = (Vec<i64>, Vec<f32>);

/// 一份见证源文件：读出来，并确认里面有这个前端能调用的函数（签名由编译器回答）。
/// 错误里带原语名，是因为同一个文件被多条见证共用时，光有文件名定位不了是哪一条。
fn witness_source(prim: &Prim, witness: &Witness, dirs: &Dirs, params: &str) -> Result<String> {
    let text = source(witness, dirs).map_err(|error| error.with(format!("`{prim}` 的")))?;
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
fn witness_files(selected: &[Selected<'_>], dirs: &Dirs, params: &str) -> Result<Sources> {
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
fn push_sources(program: &mut String, files: &Sources) {
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
}

/// 让一段文本以换行结尾（空文本保持为空）：下一段可以直接接上。
fn normalize_newline(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// 写一个文件，并把它的上层目录建出来：一次性构建目录都只有一层。
fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", parent.display()))
        })?;
    }
    std::fs::write(path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))
}

/// 跑一条编译器命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
fn run_tool(name: &str, command: &mut Command, source: &Path) -> Result<()> {
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
fn run_exe(command: &mut Command, path: &Path, failure: &str) -> Result<(String, String)> {
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
/// 否则 [`backend::parse_bits`] 会把计数当成值解出来。
fn readback(stdout: &str, stderr: &str) -> Readback {
    (parse_counts(stderr), backend::parse_bits(stdout))
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

/// C 前端：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。它证明"这段 C 在这个
/// ABI 下编得过、跑得出 `expect`"，**不**证明设备具备该能力。身份由 `#[frontend]` 生成。
#[annotation::frontend(emit = "c", ext = "c", tag = "witness")]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn available(_profile: &Profile) -> bool {
        // `c` 的设备就是本机 CPU：它永远在。
        true
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let mut program = c_head(profile, scope, selected);
        push_sources(&mut program, &witness_files(selected, dirs, C_PARAMS)?);
        program.push_str(&harness::<OUT>(selected));
        Ok(program)
    }

    fn run(
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        scope: Scope,
    ) -> Result<Readback> {
        let (_, flags) = c_compile_inputs(profile, scope);
        // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
        let tag = match scope {
            Scope::Device => Self::TAG,
            Scope::Host => "self",
        };
        // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
        let (outcome, _artifact) =
            build::run_program(backend::detect().as_ref(), &flags, tag, program, dirs)?;
        // stdout 已由 `backend` 解码成 f32，这里只补上 stderr 里的写入个数。
        Ok((parse_counts(&outcome.stderr), outcome.bits))
    }
}

/// C 前端的头：档案前置声明 + `stdio.h`；自检时再吞掉 MSL 的地址空间限定符。`#define thread` 只服务
/// 于自检，设备前端走真正的 `xcrun metal`，所以这里的放宽不会被当成设备证据。
fn c_head(profile: &Profile, scope: Scope, selected: &[Selected<'_>]) -> String {
    let (mut head, _) = c_compile_inputs(profile, scope);
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
pub(crate) fn harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    let mut text = format!("\nint main(void) {{\n    float out[{OUT}];\n    int n;\n");
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

/// Metal 前端：MSL → `metallib` → host 驱动发射到 GPU → 回读。不能用 host C 自检代替：MSL 没有默认
/// 地址空间，`float *out` 在 `.metal` 里是语法错误。证明"这条见证在本机 Metal 设备上跑得出 `expect`"，
/// **不**证明它在别的芯片、别的 `-std` 下同样成立。
#[annotation::frontend(emit = "metal", ext = "metal", tag = "witness")]
pub struct MetalDevice;

impl Device for MetalDevice {
    fn available(_profile: &Profile) -> bool {
        // 工具链在，就说明这台机器有 Metal 编译器；再往下才是"有没有 GPU"。
        metal_toolchain().is_some()
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> Result<String> {
        metal_program::<OUT>(profile, dirs, selected)
    }

    fn run(
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        // 设备参数来自档案：`-std=metal3.1` 之类决定了这段 MSL 按哪个语言版本编。
        run_metal(program, count, &profile.flags, dirs)
    }
}

/// MSL 源：头 + 见证源码 + 按 `tid` 分派的 kernel。`tid` 是见证在本次执行里的下标，写出的个数与
/// 值分别落到 `counts` 与 `out[tid * OUT ..]`；`OUT` 是 const 泛型参数，缓冲区尺寸都是字面量。
fn metal_program<const OUT: usize>(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
) -> Result<String> {
    let mut program = metal_head(profile);
    push_sources(&mut program, &witness_files(selected, dirs, MSL_PARAMS)?);
    push_metal_kernel(&mut program, OUT, selected);
    Ok(program)
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

/// 两个后端 kernel 共享的分派段：按 `tid` 的 `switch`、cap 截断与回写循环。`close` 是 kernel
/// 收尾文本（Metal 与 CUDA 的换行数不同）：`local` 声明留在各自函数里，这里只拼逐字相同的部分。
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
    let Some(compiler) = backend::detect() else {
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

/// `xcrun metal` 的实际工具链：命令可用、能自报版本时才算数。不用 `[meta].tool` 判断：那是人写的
/// 登记值，而这里要回答的是"这台机器现在能不能把 `.metal` 编成 `metallib`"。
///
/// 结果在本进程内缓存，与 [`backend::detect`](crate::backend::detect) 同一口径：
/// 工具链是进程级快照，不重复起 `xcrun` 子进程。
pub(crate) fn metal_toolchain() -> Option<backend::Toolchain> {
    static METAL: OnceLock<Option<backend::Toolchain>> = OnceLock::new();
    METAL.get_or_init(probe_metal).clone()
}

fn probe_metal() -> Option<backend::Toolchain> {
    let output = Command::new("xcrun")
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

/// CUDA 真机前端：nvcc 把见证编进 `__global__` kernel，在 GPU 上按 tid 逐条执行，再 cudaMemcpy 回读
/// （与 `CudaFront` 的主机语法自检不同，这里是真设备执行）。它**不**证明别的架构或别的 `-arch` 下成立。
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
        dirs: &Dirs,
        selected: &[Selected<'_>],
        _scope: Scope,
    ) -> Result<String> {
        cuda_program::<OUT>(profile, dirs, selected)
    }

    fn run(
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        let (stdout, stderr) = compile_run_cuda(profile, dirs, program, "witness")?;
        Ok(readback(&stdout, &stderr))
    }
}

/// CUDA 源：头 + `__device__` 见证 + `__global__` 分派 kernel + host main；见证只在 kernel 里调用。
fn cuda_program<const OUT: usize>(
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
        let Some(cl) = find_cl() else {
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

/// 定位 MSVC cl.exe（Windows 上 nvcc 的宿主编译器）：先问 `vswhere`，再退回环境变量登记的两个安装根，多版本共存时按**数值**取最新。
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

/// 一个安装根下的全部 cl.exe（`<root>/Tools/MSVC/<版本>/bin/Hostx64/x64/cl.exe`）。两个根形状都
/// 看：`VCINSTALLDIR` 直接指到 `…\VC`，`VSINSTALLDIR` 指到安装根（下面还有 `VC`）。
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

/// cl.exe 路径里 `MSVC\<版本>\` 的那一段，按**数值**比较。按字符串找标记而不是 `Path::components`：`\` 只在 Windows 上是分隔符，而单元测试跑在别的平台。
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

/// 本机的 clang：先问 `clang`，再退到 `cc`（macOS 与多数发行版的 `cc` 就是 clang）。判据是
/// "这个程序是不是 clang"：`-x cuda` / `-x cl` 是 clang 的前端，gcc 没有。
pub(crate) fn clang() -> Option<backend::Toolchain> {
    backend::probe("clang")
        .or_else(|| backend::detect().filter(|toolchain| toolchain.banner.contains("clang")))
}

/// 自检用的 host C 参数：设备档案的 `[compile].flags` 在本机要么指向不存在的头，要么只对真机编译器有意义；它与设备执行是两个参数集。
pub(crate) const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// C 前端的编译输入：前置声明 + 参数。自检只走 host C ABI，理由见 [`HOST_FLAGS`]。
pub(crate) fn c_compile_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (
            String::new(),
            HOST_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成各前端的见证源码。默认只断言生成非空；`CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本
    /// 写到 `/tmp/cutile-golden` 供人工比对，不常驻产物。
    #[test]
    fn golden_capture() {
        let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
        let dir = std::path::PathBuf::from("/tmp/cutile-golden");
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-golden-work"));
        let catalogue = crate::builtin::Catalogue::try_builtin().expect("内置数据");
        for (vendor, file) in [
            ("cpu", "witness_c.txt"),
            ("metal", "witness_metal.txt"),
            ("nvidia", "witness_cuda.txt"),
        ] {
            let profile = catalogue.profile(vendor).cloned().expect("档案");
            let selected = crate::witness::declared_witnesses(&profile).expect("见证");
            if selected.is_empty() {
                continue;
            }
            let text = match vendor {
                "cpu" => CpuDevice::program::<MAX_OUT>(&profile, &dirs, &selected, Scope::Device),
                "metal" => {
                    MetalDevice::program::<MAX_OUT>(&profile, &dirs, &selected, Scope::Device)
                }
                _ => CudaDevice::program::<MAX_OUT>(&profile, &dirs, &selected, Scope::Device),
            }
            .expect("可生成");
            assert!(!text.is_empty(), "{vendor} 生成文本不能为空");
            if capture {
                std::fs::create_dir_all(&dir).expect("目录");
                std::fs::write(dir.join(file), text).expect("写");
            }
        }
        let metal = catalogue.profile("metal").cloned().expect("档案");
        let selected = crate::witness::declared_witnesses(&metal).expect("见证");
        let text =
            CpuDevice::program::<MAX_OUT>(&metal, &dirs, &selected, Scope::Host).expect("可生成");
        assert!(!text.is_empty(), "metal 档案的 host 自检文本不能为空");
        let driver = metal_driver(MAX_OUT);
        assert!(!driver.is_empty(), "Metal 驱动包装不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_metal_host.txt"), text).expect("写");
            std::fs::write(dir.join("metal_driver.txt"), driver).expect("写");
        }
    }

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
        let dirs = Dirs::under(std::env::temp_dir());
        let first = scratch(&dirs, "gate");
        let second = scratch(&dirs, "gate");
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(dirs.work()));
    }
}
