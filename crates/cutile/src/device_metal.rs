//! Metal 前端：MSL → `metallib` → host 驱动发射到 GPU → 回读。
//!
//! 不能用 host C 自检代替：MSL 没有默认地址空间，`float *out` 在 `.metal` 里是语法错误。
//! 证明"这条见证在本机 Metal 设备上跑得出 `expect`"，**不**证明它在别的芯片、别的 `-std` 下同样成立。
//! 共享装载、起进程与回读在 [`crate::device`]。

use crate::dirs::Dirs;
use crate::prof::Profile;
use crate::toolchain;
use crate::witness::{Device, MAX_OUT, Selected};
use diagnostic::{Code, Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// MSL 的形参必须带地址空间：MSL 没有默认地址空间，裸 `float *` 在那里是语法错误。
const MSL_PARAMS: &str = "thread float *";

/// Metal 前端：MSL → `metallib` → host 驱动发射到 GPU → 回读。不能用 host C 自检代替：MSL 没有默认
/// 地址空间，`float *out` 在 `.metal` 里是语法错误。证明"这条见证在本机 Metal 设备上跑得出 `expect`"，
/// **不**证明它在别的芯片、别的 `-std` 下同样成立。
#[annotation::frontend(emit = "metal", ext = "metal", tag = "witness")]
pub struct MetalDevice;

impl Device for MetalDevice {
    fn available(_profile: &Profile) -> bool {
        // 工具链在，就说明这台机器有 Metal 编译器；再往下才是"有没有 GPU"。
        toolchain::detect_metal().is_some()
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        _scope: crate::witness::Scope,
    ) -> Result<String> {
        metal_program::<OUT>(profile, dirs, selected)
    }

    fn run(
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        count: usize,
        _scope: crate::witness::Scope,
    ) -> Result<crate::device::Readback> {
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
    crate::device::push_sources(
        &mut program,
        &crate::device::witness_files(selected, dirs, MSL_PARAMS)?,
    );
    push_metal_kernel(&mut program, OUT, selected);
    Ok(program)
}

/// MSL 的头：档案的 `[compile].preamble`，补齐结尾换行后直接接见证源码。
fn metal_head(profile: &Profile) -> String {
    let mut head = profile.preamble.clone();
    crate::device::normalize_newline(&mut head);
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
    crate::device::push_dispatch(program, max, selected, "}\n");
}

/// 编译并执行一次 Metal 见证程序，返回 `(每条写入个数, 按顺序回读的值)`。写、起子进程、回读三步分开；
/// 编不过与跑不起来都带完整输出，否则只会以一句 "compile failed" 出现。
fn run_metal(
    program: &str,
    count: usize,
    flags: &[String],
    dirs: &Dirs,
) -> Result<crate::device::Readback> {
    let dir = crate::device::scratch(dirs, "witness-metal");
    let outcome = metal_in_dir(&dir, program, count, flags);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

/// 一次性目录里的一条：写 MSL → 编成 `metallib` → 编驱动 → 跑 → 回读。
fn metal_in_dir(
    dir: &Path,
    program: &str,
    count: usize,
    flags: &[String],
) -> Result<crate::device::Readback> {
    let metalib = metal_kernel(dir, program, flags)?;
    let exe = metal_driver_exe(dir)?;
    metal_readback(&exe, &metalib, count)
}

/// 写 MSL 并编成 `metallib`：`xcrun metal` 管语言层、`metallib` 管链接；诊断里带 `.metal` 路径。
fn metal_kernel(dir: &Path, program: &str, flags: &[String]) -> Result<PathBuf> {
    let msl = dir.join("witness.metal");
    let air = dir.join("witness.air");
    let metalib = dir.join("witness.metallib");
    crate::device::write_text(&msl, program)?;
    let mut compile = xcrun("metal", flags);
    compile.arg("-c").arg(&msl).arg("-o").arg(&air);
    crate::device::run_tool("xcrun metal", &mut compile, &msl)?;
    let mut link = xcrun("metallib", &[]);
    link.arg(&air).arg("-o").arg(&metalib);
    crate::device::run_tool("xcrun metallib", &mut link, &msl)?;
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
    crate::device::write_text(&driver, &metal_driver(MAX_OUT))?;
    let mut clang = Command::new(&compiler.command);
    clang
        .args(["-fobjc-arc", "-O2", "-o"])
        .arg(&exe)
        .arg(&driver);
    clang.args(["-framework", "Foundation", "-framework", "Metal"]);
    crate::device::run_tool(&compiler.command, &mut clang, &driver)?;
    Ok(exe)
}

/// 起驱动子进程并回读：`metallib` 与条数是它的两个参数。
fn metal_readback(exe: &Path, metalib: &Path, count: usize) -> Result<crate::device::Readback> {
    let mut command = Command::new(exe);
    command.arg(metalib).arg(count.to_string());
    let (stdout, stderr) = crate::device::run_exe(&mut command, exe, "Metal 见证执行失败")?;
    Ok(crate::device::readback(&stdout, &stderr))
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
mod tests {
    use super::*;

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
    fn metal_driver_wrapper_is_nonempty() {
        let driver = metal_driver(MAX_OUT);
        assert!(!driver.is_empty());
    }
}
