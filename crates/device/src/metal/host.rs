//! Metal host 驱动骨架：缓冲装配（初值 / 分配 / 绑定）、host 程序模板、单段派发与基准两轮正文。
//! 单段语义门、网格派发、基准驱动、多段驱动只在「前导 / 缓冲 / 正文」三块数据上不同，骨架只有这一份。

use cutile::harness::{c_literal, output_of};
use diagnostic::{Code, Error, Result};
use prof::model::{Sample, Values};
use prof::sig::Signature;

use crate::dialect::{f32_output_shape, fill, readback_bits};

/// 一条缓冲在 host 侧的三份文本：初值字面量、`MTLBuffer` 分配、`setBuffer` 绑定。
///
/// 单段驱动、基准驱动与多段驱动只差三个数据位——输出容量（`None` 取样本原长）、分配失败是否立即
/// 返回（语义门不查，基准/多段查）、绑定行缩进——拼装逻辑因此只有这一份。
#[derive(Default)]
pub(crate) struct Wiring {
    pub(crate) data: String,
    pub(crate) allocs: String,
    pub(crate) sets: String,
    /// 唯一可写输出缓冲的元素数（与 `capacity` 同口径）。
    pub(crate) out_len: usize,
}

impl Wiring {
    /// 按签名序走一遍缓冲参数，`capacity` 换掉输出缓冲的样本长度（部分结果数）。
    pub(crate) fn of(
        signature: &Signature,
        sample: &Sample,
        capacity: Option<usize>,
        checked: bool,
        indent: usize,
    ) -> Result<Self> {
        let mut wiring = Self::default();
        let pad = " ".repeat(indent);
        for (slot, param) in signature
            .params()
            .iter()
            .filter(|param| param.ctype.is_buf())
            .enumerate()
        {
            let base = param.ctype.c_name();
            if param.mutable {
                wiring.emit_output(&param.name, base, sample, capacity, checked)?;
            } else {
                wiring.emit_input(&param.name, base, sample, checked)?;
            }
            wiring.sets.push_str(&format!(
                "{pad}[encoder setBuffer:b_{} offset:0 atIndex:{slot}];\n",
                param.name
            ));
        }
        Ok(wiring)
    }

    /// 可变输出缓冲：容量取调用方给定值，否则回看样本；分配一次开满。
    fn emit_output(
        &mut self,
        name: &str,
        base: &str,
        sample: &Sample,
        capacity: Option<usize>,
        checked: bool,
    ) -> Result<()> {
        let len = match capacity {
            Some(capacity) => capacity,
            None => sample
                .buf(name)
                .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?
                .values
                .len(),
        };
        self.out_len = len;
        self.allocs
            .push_str(&output_alloc(name, base, len, checked));
        Ok(())
    }

    /// 只读输入缓冲：声明静态初值数组并分配只读缓冲。
    fn emit_input(&mut self, name: &str, base: &str, sample: &Sample, checked: bool) -> Result<()> {
        let buf = sample
            .buf(name)
            .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
        self.data.push_str(&literal_decl(name, base, &buf.values));
        self.allocs.push_str(&input_alloc(name, checked));
        Ok(())
    }
}

/// 只读输入缓冲的初值静态数组声明。
fn literal_decl(name: &str, base: &str, values: &Values) -> String {
    format!(
        "static const {base} init_{name}[] = {{{}}};\n",
        c_literal(values)
    )
}

/// 输出缓冲的分配：可变、按元素数一次开满；`checked` 决定分配失败是否立即返回。
pub(crate) fn output_alloc(name: &str, base: &str, len: usize, checked: bool) -> String {
    let alloc = format!(
        "        id<MTLBuffer> b_{name} = [device newBufferWithLength:{len} * sizeof({base}) options:MTLResourceStorageModeShared];\n"
    );
    if checked {
        return format!(
            "{alloc}        if (!b_{name}) {{ fprintf(stderr, \"Metal 输出缓冲分配失败\\n\"); return 7; }}\n"
        );
    }
    alloc
}

/// 输入缓冲的分配：按静态初值的长度原样拷入；`checked` 决定分配失败是否立即返回。
fn input_alloc(name: &str, checked: bool) -> String {
    let alloc = format!(
        "        id<MTLBuffer> b_{name} = [device newBufferWithBytes:init_{name} length:sizeof(init_{name}) options:MTLResourceStorageModeShared];\n"
    );
    if checked {
        return format!(
            "{alloc}        if (!b_{name}) {{ fprintf(stderr, \"Metal 输入缓冲分配失败\\n\"); return 7; }}\n"
        );
    }
    alloc
}

/// import 段的三档：语义门与基准按需取 `stdlib.h`，多段驱动的 `sqrt` 要 `math.h`。
pub(crate) const IMPORTS_STDLIB: &str = "#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n";
pub(crate) const IMPORTS_STDIO: &str =
    "#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdio.h>\n\n";
pub(crate) const IMPORTS_MATH: &str = "#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdio.h>\n#include <math.h>\n\n";

/// host 程序模板：`@autoreleasepool` 包住取设备、装载、分配、发射与回读。
const HOST_PROGRAM: &str = r#"@imports@data
int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: driver <metallib>\n"); return 2; }
@prologue@allocs@body        const float *out = (const float *)b_@output.contents;
@readback
    }
    return 0;
}
"#;

/// 按数据模板装配一个 host 程序；`body` 是 `@autoreleasepool` 块里除回读之外的正文。
pub(crate) fn host_program(
    imports: &str,
    data: &str,
    prologue: &str,
    allocs: &str,
    body: &str,
    output: &str,
    out_len: usize,
) -> String {
    fill(
        HOST_PROGRAM,
        &[
            ("imports", imports),
            ("data", data),
            ("prologue", prologue),
            ("allocs", allocs),
            ("body", body),
            ("output", output),
            ("readback", &readback_bits("out", out_len, "        ")),
        ],
    )
}

/// 单段派发正文：建命令队列、一次发射、等完成。
const DISPATCH_BODY: &str = r#"        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
@sets        [encoder dispatchThreads:MTLSizeMake(@gx, @gy, @gz) threadsPerThreadgroup:MTLSizeMake(@bx, @by, @bz)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) { fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String); return 7; }
"#;

/// 单段派发正文的填空：绑定段 + 两套几何。
fn dispatch_body(sets: &str, grid: [usize; 3], block: [usize; 3]) -> String {
    fill(
        DISPATCH_BODY,
        &[
            ("sets", sets),
            ("gx", &grid[0].to_string()),
            ("gy", &grid[1].to_string()),
            ("gz", &grid[2].to_string()),
            ("bx", &block[0].to_string()),
            ("by", &block[1].to_string()),
            ("bz", &block[2].to_string()),
        ],
    )
}

/// host 驱动共用的前导：取默认设备、装载 metallib、取入口函数、建管线。
pub(crate) fn host_prologue(entry: &str) -> String {
    format!(
        r#"{library}        id<MTLFunction> function = [library newFunctionWithName:@"{entry}"];
        if (!function) {{ fprintf(stderr, "metallib 里没有 {entry}\n"); return 5; }}
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {{ fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String); return 6; }}
"#,
        library = host_library()
    )
}

/// host 前导的公共半截：取默认设备、装载 metallib。单段驱动取一次入口函数，多段驱动每段各取一次，
/// 因此这一半单独成文，返回码与诊断文本只有这一处。
pub(crate) fn host_library() -> &'static str {
    r#"        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "本机没有 Metal 设备\n"); return 3; }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) { fprintf(stderr, "metallib 装载失败：%s\n", error.localizedDescription.UTF8String); return 4; }
"#
}

/// host 驱动（单线程语义门）：把样本数据作为只读缓冲的初值，发射 `cutile_launch`，回读唯一可写
/// 输出的位模式——验证「算得对不对」，不回答「跑得多快」。
pub(crate) fn metal_harness_driver(
    signature: &Signature,
    sample: &Sample,
    output: &str,
) -> Result<String> {
    let wiring = Wiring::of(signature, sample, None, false, 8)?;
    let body = dispatch_body(&wiring.sets, [1; 3], [1; 3]);
    Ok(host_program(
        IMPORTS_STDLIB,
        &wiring.data,
        &host_prologue("cutile_launch"),
        &wiring.allocs,
        &body,
        output,
        wiring.out_len,
    ))
}

/// host 驱动（网格发射）：初值与绑定同语义门那份，入口名与几何由调用方给——逐点与循环注入
/// 的入口是 `kernel_main`，线程组与总线程数分别由 `block` / `grid` 给出。
///
/// # Errors：缺输出缓冲、签名与样本不一致时返回说明。
pub fn metal_dispatch_driver(
    signature: &Signature,
    sample: &Sample,
    entry: &str,
    grid: [usize; 3],
    block: [usize; 3],
) -> Result<String> {
    let (output, _) = output_of(signature, sample)?;
    let wiring = Wiring::of(signature, sample, None, false, 8)?;
    let body = dispatch_body(&wiring.sets, grid, block);
    Ok(host_program(
        IMPORTS_STDLIB,
        &wiring.data,
        &host_prologue(entry),
        &wiring.allocs,
        &body,
        output,
        wiring.out_len,
    ))
}

/// 基准驱动的一轮调度：建 command buf、设缓冲与管线、按 16 批发射、提交并等待。
///
/// 预热与计时两段只在尾巴上分叉（错误文本与返回码），所以这一轮本身只写一份数据。
const METAL_ROUND: &str = r#"            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
@sets            for (int batch = 0; batch < 16; ++batch) {
                [encoder dispatchThreads:MTLSizeMake(@grid_x, @grid_y, @grid_z)
                    threadsPerThreadgroup:MTLSizeMake(@thread_x, @thread_y, @thread_z)];
            }
            [encoder endEncoding];
            [buffer commit];
            [buffer waitUntilCompleted];
"#;

/// 基准驱动正文：预热两轮后按 `repeat` 轮各跑 16 批，每轮报一次 GPU 时间戳。
const BENCH_BODY: &str = r#"        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { fprintf(stderr, "Metal 命令队列创建失败\n"); return 7; }
        for (int warm = 0; warm < 2; ++warm) {
@round            if (buffer.error) { fprintf(stderr, "内核预热失败：%s\n", buffer.error.localizedDescription.UTF8String); return 8; }
        }
        for (int sample = 0; sample < @repeat; ++sample) {
@round            if (buffer.error) { fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String); return 9; }
            double elapsed = (buffer.GPUEndTime - buffer.GPUStartTime) * 1.0e9;
            if (!(elapsed > 0.0)) { fprintf(stderr, "Metal GPU 时间戳不可用\n"); return 10; }
            fprintf(stderr, "ns %.3f\n", elapsed / 16.0);
        }
"#;

/// 生产态 Metal 基准驱动：host 只负责输入上传与派发，样本计时取自 command buf 的 GPU 时间戳。
pub fn metal_benchmark_driver(
    signature: &Signature,
    sample: &Sample,
    grid: [usize; 3],
    threads: [usize; 3],
    repeat: usize,
) -> Result<String> {
    let (output, out_len) = f32_output_shape(signature, sample)?;
    let wiring = Wiring::of(signature, sample, None, true, 12)?;
    // 两层模板各填各的：一轮调度先落成文本，再嵌进基准正文——填空不跨模板递归。
    let round = fill(
        METAL_ROUND,
        &[
            ("sets", &wiring.sets),
            ("grid_x", &grid[0].to_string()),
            ("grid_y", &grid[1].to_string()),
            ("grid_z", &grid[2].to_string()),
            ("thread_x", &threads[0].to_string()),
            ("thread_y", &threads[1].to_string()),
            ("thread_z", &threads[2].to_string()),
        ],
    );
    let body = fill(
        BENCH_BODY,
        &[("round", &round), ("repeat", &repeat.to_string())],
    );
    Ok(host_program(
        IMPORTS_STDIO,
        &wiring.data,
        &host_prologue("kernel_main"),
        &wiring.allocs,
        &body,
        &output,
        out_len,
    ))
}
