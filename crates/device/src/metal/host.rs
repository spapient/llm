//! Metal 主机侧胶水：`xcrun metal` 编设备源、`clang` 链接主机驱动、执行与回读。
//!
//! 从 [`crate::backends`] 拆出：那里同时装 CUDA 与 Metal 两套厂商工具链，行数已过单文件上限，
//! 而两套胶水之间没有共享状态——拆开后每条链各自一屏读得完。`backends` 仍是 [`llm_optim::backend`]
//! 的 `Backend` 实现，并通过 re-export 保持这些入口的原路径不变。
//!
//! 多段驱动（[`MetalStage`]）同住这里：段与段之间只隔一段主机折叠代码，拼装口径与单段模板逐字同源。

use super::*;

/// 多段驱动的一段：入口名、按 `buffer(i)` 顺序的缓冲变量名（带 `b_` 前缀）、派发几何、段后主机代码。
///
/// `fold` 是这一段落定之后、下一段之前的主机折叠（部分和 → 总和、部分平方和 → `rms`、
/// 携带前缀推进），空串表示这一段不需要折叠。几何按「工作项数」给，线程组恒为 `(1,1,1)`：
/// 分块族是每线程一块，块内串行走完。
pub(crate) struct MetalStage<'a> {
    pub entry: &'a str,
    pub buffers: &'a [&'a str],
    pub grid: [usize; 3],
    pub fold: &'a str,
}

/// 多段并行驱动：装载一次 metallib，按段依次发射，段间插入主机折叠，最后回读输出。
///
/// `capacity` 是**输出缓冲按元素开的长度**：分块族的部分结果先写进输出缓冲（`y[g]`），
/// 块数可能大于最终输出长度（归约是 1 个标量），因此分配与回读分开给。
/// `scratch` 是折叠用的中间缓冲（`rms` / `max` / `sum` / 携带态），一律 f32。
///
/// # Errors
///
/// 签名与样本不一致、或某段找不到入口函数时返回说明（入口函数缺失由 host 侧返回码 5 报出）。
pub(crate) fn metal_multistage_driver(
    signature: &Signature,
    sample: &Sample,
    capacity: usize,
    read: usize,
    scratch: &[(&str, usize)],
    stages: &[MetalStage<'_>],
) -> Result<String> {
    let output = output_name(signature)?;
    let (data, allocs) = metal_buffers(signature, sample, capacity, scratch)?;
    let body = metal_stages(stages);
    Ok(format!(
        r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>

{data}
int main(int argc, const char **argv) {{
    @autoreleasepool {{
        if (argc < 2) {{ fprintf(stderr, "usage: driver <metallib>\n"); return 2; }}
{library}{allocs}        id<MTLCommandQueue> queue = [device newCommandQueue];
{body}        const float *out = (const float *)b_{output}.contents;
{readback}
    }}
    return 0;
}}
"#,
        data = data,
        library = host_library(),
        allocs = allocs,
        body = body,
        output = output,
        readback = readback_bits("out", read, "        "),
    ))
}

/// 缓冲装载：只读缓冲落字面量、可写缓冲按 `capacity` 开满，折叠用的中间缓冲另开 f32。
fn metal_buffers(
    signature: &Signature,
    sample: &Sample,
    capacity: usize,
    scratch: &[(&str, usize)],
) -> Result<(String, String)> {
    let mut data = String::new();
    let mut allocs = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        let (literal, alloc) = metal_param(param, sample, capacity)?;
        data.push_str(&literal);
        allocs.push_str(&alloc);
    }
    for (name, len) in scratch {
        allocs.push_str(&output_alloc(name, "float", *len));
    }
    Ok((data, allocs))
}

/// 段体：每段一份发射 + 该段自己的主机折叠（折叠串由调用方给，不在驱动里拼第二份算式）。
fn metal_stages(stages: &[MetalStage<'_>]) -> String {
    let mut body = String::new();
    for (index, stage) in stages.iter().enumerate() {
        body.push_str(&metal_stage(index, stage));
        body.push_str(stage.fold);
    }
    body
}

/// 一个缓冲参数的两段：只读的落地为字面量，可写的按 `capacity` 开满。
fn metal_param(param: &Param, sample: &Sample, capacity: usize) -> Result<(String, String)> {
    let base = param.ctype.c_name();
    if param.mutable {
        return Ok((String::new(), output_alloc(&param.name, base, capacity)));
    }
    let buffer = sample
        .buffer(&param.name)
        .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
    Ok((
        literal_decl(&param.name, base, &buffer.values),
        input_alloc(&param.name),
    ))
}

/// 单段发射：取入口函数、建管线、设缓冲、派发、提交并等待。
fn metal_stage(index: usize, stage: &MetalStage<'_>) -> String {
    let mut sets = String::new();
    for (slot, name) in stage.buffers.iter().enumerate() {
        sets.push_str(&format!(
            "            [encoder{index} setBuffer:b_{name} offset:0 atIndex:{slot}];\n"
        ));
    }
    format!(
        r#"        {{
            id<MTLFunction> function{index} = [library newFunctionWithName:@"{entry}"];
            if (!function{index}) {{ fprintf(stderr, "metallib 里没有 {entry}\n"); return 5; }}
            id<MTLComputePipelineState> pipeline{index} = [device newComputePipelineStateWithFunction:function{index} error:&error];
            if (!pipeline{index}) {{ fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String); return 6; }}
            id<MTLCommandBuffer> buffer{index} = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder{index} = [buffer{index} computeCommandEncoder];
            [encoder{index} setComputePipelineState:pipeline{index}];
{sets}            [encoder{index} dispatchThreads:MTLSizeMake({gx}, {gy}, {gz}) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder{index} endEncoding];
            [buffer{index} commit];
            [buffer{index} waitUntilCompleted];
            if (buffer{index}.error) {{ fprintf(stderr, "内核执行失败：%s\n", buffer{index}.error.localizedDescription.UTF8String); return 7; }}
        }}
"#,
        entry = stage.entry,
        sets = sets,
        gx = stage.grid[0],
        gy = stage.grid[1],
        gz = stage.grid[2],
    )
}

/// 一轮调度：建 command buffer、设缓冲与管线、按 16 批发射、提交并等待。
///
/// 预热与计时两段只在尾巴上分叉（错误文本与返回码），所以这一轮本身只写一份：
/// 占位符在插入后由同一批 [`str::replace`] 补齐（`@ROUND@` 先于 `@SETS@` 等）。
/// 驱动正文模板：只拼字面与已算好的片段，不含循环与错误处理（那些留在 `metal_driver`）。
#[allow(clippy::too_many_arguments)]
pub(super) fn metal_driver_template(
    data: &str,
    allocs: &str,
    sets: &str,
    output: &str,
    out_len: usize,
    entry: &str,
    grid: [usize; 3],
    block: [usize; 3],
) -> String {
    format!(
        r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

{data}
int main(int argc, const char **argv) {{
    @autoreleasepool {{
        if (argc < 2) {{ fprintf(stderr, "usage: driver <metallib>\n"); return 2; }}
{prologue}{allocs}        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
{sets}        [encoder dispatchThreads:MTLSizeMake({gx}, {gy}, {gz}) threadsPerThreadgroup:MTLSizeMake({bx}, {by}, {bz})];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {{ fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String); return 7; }}
        const float *out = (const float *)b_{output}.contents;
{readback}
    }}
    return 0;
}}
"#,
        data = data,
        allocs = allocs,
        prologue = host_prologue(entry),
        sets = sets,
        output = output,
        readback = readback_bits("out", out_len, "        "),
        gx = grid[0],
        gy = grid[1],
        gz = grid[2],
        bx = block[0],
        by = block[1],
        bz = block[2]
    )
}

use cutile::dirs::Dirs;
use cutile::prof::Profile;
use diagnostic::{Code, Error, Result};
use llm_optim::backend::Readback;
use llm_optim::driver;
use llm_optim::witness::MAX_OUT;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 跑一次见证：一次性目录里编 `metallib`、链接见证驱动、回读位模式。
pub(crate) fn run_metal(
    program: &str,
    count: usize,
    flags: &[String],
    dirs: &Dirs,
) -> Result<Readback> {
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
/// 与 [`llm_optim::backend::Driver::compile_run`] 同口径：一次性目录，用完即删。
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
    let Some(compiler) = llm_optim::toolchain::detect() else {
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
/// [`MAX_OUT`] 注入。供测试与 `metal_driver_exe_from` 共用，不重复写驱动文本。
pub fn metal_driver(max: usize) -> String {
    format!(
        "#define WITNESS_MAX_OUT {max}\n{METAL_HEAD}{prologue}{METAL_TAIL}",
        prologue = crate::metal::host_prologue("cutile_witness")
    )
}

/// 见证驱动的 head：入口、参数与见证宽度用法；与 [`crate::metal`] 的基准驱动共用同一段前导。
const METAL_HEAD: &str = r#"#import <Foundation/Foundation.h>
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
"#;

/// 见证驱动的 tail：两条缓冲、一次发射与「见证宽度」回读。
const METAL_TAIL: &str = r#"        id<MTLBuffer> values = [device newBufferWithLength:(NSUInteger)count * WITNESS_MAX_OUT * sizeof(float)
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
