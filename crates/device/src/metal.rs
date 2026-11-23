//! Metal(MSL) 侧的设备源与 host 驱动拼装：`device` 地址空间、`[[buf(slot)]]` 缓冲、compute
//! kernel 发射、`MTLBuffer` 分配 / 回读骨架。
//!
//! 后端无关的 lowering 原语住在 [`crate::dialect::Dialect`] trait 的**默认方法**里，这里只写
//! Metal 一侧的 vendor 驱动骨架（`xcrun metal` 的 MSL 文本 + clang 的 ObjC driver 文本）。同一条
//! kernel body 由 [`Metal`](crate::dialect::Metal) 方言落地，与 CUDA 侧的 [`crate::cuda`] 是
//! 同层的一件事：只换方言，不改语义，回读口径（`float` 位模式逐行打印）必须逐字一致。

use cutile::dirs::Dirs;
use cutile::harness::{c_literal, output_name, output_of};
use cutile::prof::Profile;
use diagnostic::{Code, Error, Result};
use llm_optim::backend::{Readback, Source};
use llm_optim::driver;
use llm_optim::witness::MAX_OUT;
use nn::model::{Sample, Values};
use nn::sig::Signature;
use prof::decl::Grid;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::dialect::{
    Dialect, EXPSUM, Metal, NORM, PARTIAL, RMS_EPS, SCAN_CORRECT, SCAN_LOCAL, carry_fold,
    chunk_head, f32_output_shape, fill, launch_args, length_scalar, length_usize, readback_bits,
    readonly_bufs, scalar_name, scan_correct_body,
};

// ---------------------------------------------------------------------------
// Metal(MSL) lowering：`[grid]`/`[chunk]` 声明式骨架 → 设备源，只用 `Metal` 方言。
// ---------------------------------------------------------------------------

/// 拼出可在本机 GPU 上真实执行的 Metal 程序：`.k` 内核补 `device` 地址空间后，`cutile_launch`
/// 把它包成 compute kernel（`device` 缓冲 ↔ 线程局部数组），host 驱动负责把样本发射到 GPU、
/// 回读输出。返回 `(msl, driver)` 两段文本：`msl` 交给 `xcrun metal`，`driver` 交给 clang。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn metal_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<(String, String)> {
    let (output, _) = output_of(signature, sample)?;
    let msl = format!(
        "{}\n{}",
        Metal::address_text(kernel_source),
        metal_launch(&signature.kernel(), signature, sample)?
    );
    Ok((msl, metal_harness_driver(signature, sample, output)?))
}

/// 生产态 Metal 逐点网格内核（[`crate::dialect::Dialect::grid_kernel`] 的 Metal 方言）。
///
/// 带长度守卫：`dispatchThreads` 会把总线程数向上取整到线程组的整数倍，末组多出来的线程必须
/// 自己跳过——否则越界线程会往缓冲尾部之外写，把别的缓冲改掉而结果看起来「只是有点错」。
///
/// # Errors：不是逐点签名（标量数 ≠ 1）、模板缺内核函数，或样本缺长度标量。
pub fn metal_grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    Metal::grid_kernel(
        kernel_source,
        signature,
        Some(length_scalar(signature, sample)?),
    )
}

/// 生产态循环注入网格内核：把模板里的串行循环头绑定到线程下标（N 维网格，`tid.x` / `tid.y` …）。
///
/// 网格绑定来自 `[grid]` 声明（[`Grid`]），不是对模板的索引分析：每个网格维把模板里的串行循环
/// `for (int t = 0; t < S; ++t)` 的头绑定到线程下标（`t = tid.x`、循环只跑一次），正文里的跨步
/// （`i*K`、`p*N`、`i*D`）原样保留、标量取样本原值。这比「压缩标量 + 缓冲偏移」更泛：gemm 右
/// 操作数的规约维跨步 `p*N` 不会因压缩 `N` 而错位，注意力里 `i*D` 的行跨步同样保留。`tid` 声明在
/// `kernel_main`，而循环在内部函数里，故先给内部函数追加 `uint3 tid` 形参、调用点透传。gemv 一维、
/// gemm 二维、attention 按查询行一维共用这一条 lowering，差异全在声明里。
///
/// 空网格（`dims` 为空）退化为**串行**：不追加 `tid`、不绑循环、缓冲原样透传、标量取样本值——
/// 串行就是循环注入的零维特例，不再单列一条 lowering。
///
/// # Errors：网格维超过 3、模板里找不到声明的循环、标量缺样本取值时返回说明。
pub fn metal_loop_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    if grid.dims.is_empty() {
        return Metal::serial_kernel(kernel_source, signature, sample);
    }
    Metal::injected_kernel(kernel_source, signature, sample, grid)
}

/// `cutile_launch` compute kernel：按签名顺序声明 `device` 缓冲并原样透传给内核；标量直接取样本值。
fn metal_launch(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let bufs = Metal::bufs(signature);
    Ok(format!(
        "kernel void cutile_launch({bufs},\n                          uint tid [[thread_position_in_grid]]) {{\n    if (tid != 0) return;\n    {kernel}({call});\n}}"
    ))
}

/// 一条缓冲在 host 侧的三份文本：初值字面量、`MTLBuffer` 分配、`setBuffer` 绑定。
///
/// 单段驱动、基准驱动与多段驱动只差三个数据位——输出容量（`None` 取样本原长）、分配失败是否立即
/// 返回（语义门不查，基准/多段查）、绑定行缩进——拼装逻辑因此只有这一份。
#[derive(Default)]
struct Wiring {
    data: String,
    allocs: String,
    sets: String,
    /// 唯一可写输出缓冲的元素数（与 `capacity` 同口径）。
    out_len: usize,
}

impl Wiring {
    /// 按签名序走一遍缓冲参数，`capacity` 换掉输出缓冲的样本长度（部分结果数）。
    fn of(
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
            let buf = sample
                .buf(&param.name)
                .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
            let base = param.ctype.c_name();
            let len = capacity.unwrap_or(buf.values.len());
            if param.mutable {
                wiring.out_len = len;
                wiring.allocs.push_str(&output_alloc(&param.name, base, len, checked));
            } else {
                wiring.data.push_str(&literal_decl(&param.name, base, &buf.values));
                wiring.allocs.push_str(&input_alloc(&param.name, checked));
            }
            wiring.sets.push_str(&format!(
                "{pad}[encoder setBuffer:b_{} offset:0 atIndex:{slot}];\n",
                param.name
            ));
        }
        Ok(wiring)
    }
}

/// host 驱动（单线程语义门）：把样本数据作为只读缓冲的初值，发射 `cutile_launch`，回读唯一可写
/// 输出的位模式——验证「算得对不对」，不回答「跑得多快」。
fn metal_harness_driver(signature: &Signature, sample: &Sample, output: &str) -> Result<String> {
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

/// host 驱动（网格发射）：初值与绑定同 `metal_driver`，入口名与几何由调用方给——逐点与循环注入
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

/// 单个 Metal host 程序的骨架：import 段 + 静态初值 + `main` +「前导 / 缓冲 / 正文」三块数据 + 回读。
/// 单段派发、基准、多段三族只在这三块数据上不同，骨架只有这一份。
const HOST_PROGRAM: &str = r#"@imports@data
int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: driver <metallib>\n"); return 2; }
@prologue@@allocs@@body@        const float *out = (const float *)b_@output@.contents;
@readback@
    }
    return 0;
}
"#;

/// import 段的三档：语义门与基准按需取 `stdlib.h`，多段驱动的 `sqrt` 要 `math.h`。
const IMPORTS_STDLIB: &str = "#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n";
const IMPORTS_STDIO: &str = "#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdio.h>\n\n";
const IMPORTS_MATH: &str = "#import <Foundation/Foundation.h>\n#import <Metal/Metal.h>\n#include <stdio.h>\n#include <math.h>\n\n";

/// 按数据模板装配一个 host 程序；`body` 是 `@autoreleasepool` 块里除回读之外的正文。
fn host_program(
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
@sets@        [encoder dispatchThreads:MTLSizeMake(@gx, @gy, @gz) threadsPerThreadgroup:MTLSizeMake(@bx, @by, @bz)];
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
///
/// 没有设备时的返回码与三段诊断文本是 host 侧约定，[`crate::backends`] 的见证驱动与本节
/// 生成的基准驱动因此逐字一致：两份宿主程序只有尾巴（预热与计时 / 读回）不同。
fn literal_decl(name: &str, base: &str, values: &Values) -> String {
    format!(
        "static const {base} init_{name}[] = {{{}}};\n",
        c_literal(values)
    )
}

/// 输出缓冲的分配：可变、按元素数一次开满；`checked` 决定分配失败是否立即返回。
fn output_alloc(name: &str, base: &str, len: usize, checked: bool) -> String {
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

/// host 驱动共用的前导：取默认设备、装载 metallib、取入口函数、建管线。
///
/// 没有设备时的返回码与三段诊断文本是 host 侧约定，[`crate::backends`] 的见证驱动与本节
/// 生成的基准驱动因此逐字一致：两份宿主程序只有尾巴（预热与计时 / 读回）不同。
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

// ---------------------------------------------------------------------------
// Metal 分块族内核：归约、扫描、`rmsnorm`、`softmax` 的设备源拼装，以及计时基准驱动。
// ---------------------------------------------------------------------------

/// 生产态分块扫描内核：把时间轴串行递推切成 `chunk` 元素一块，返回两段内核源 `(局部扫描, 校正)`。
///
/// 1. 局部扫描：每线程 `g` 对块 `[g·chunk, min(chunk, N-g·chunk))` 做基 0 扫描，写 `y`；
/// 2. 主机携带前缀：`C_0 = 0`、`C_{g+1} = a^len·C_g + y[start+len-1]`；
/// 3. 校正：每线程 `g` 把携带态的衰减贡献 `C_g·a^(j+1)` 加回 `y[start+j]`（就地累加）。
pub fn metal_chunk_scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<(String, String)> {
    let output = output_name(signature)?;
    let scalar = scalar_name(signature)?;
    let len = length_scalar(signature, sample)?;
    let local = Metal::chunk_scan_local(kernel_source, signature, sample, chunk, scalar)?;
    let correct = chunk_scan_correct(output, chunk, len, decay);
    Ok((local, correct))
}

/// 分块 map 内核：每线程 `g` 处理一块 `[start, start+len)`，块头（`g`/`start`/`rest`/`len`）与
/// `kernel_main` 包裹只写这一份；扫描校正、RMSNorm 归一化、Softmax 指数和/归一化四段共用，
/// 差异只在缓冲声明与块内正文。
fn chunk_map_kernel(
    preamble: &str,
    entry: &str,
    bufs: &str,
    chunk: usize,
    len: i64,
    body: &str,
) -> String {
    let head = if preamble.is_empty() {
        String::new()
    } else {
        format!("{preamble}\n\n")
    };
    let chunk_lines = chunk_head(chunk, len, "    ");
    format!(
        "{head}kernel void {entry}(\n{bufs},\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n{chunk_lines}{body}\n}}"
    )
}

/// 校正内核：每线程 `g` 把携带态 `d_c[g]` 的衰减几何级数 `C·a^(j+1)` 加回 `y`（就地累加）。
/// 槽位按「输入后输出」：`d_c` 是输入在 `buf(0)`、`d_y` 是输出在 `buf(1)`。
fn chunk_scan_correct(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    let bufs =
        format!("    device float* d_c [[buffer(0)]],\n    device float* d_{output} [[buffer(1)]]");
    let body = scan_correct_body(output, decay, "    ");
    chunk_map_kernel("", SCAN_CORRECT, &bufs, chunk, len, &body)
}

/// 生产态分块 RMSNorm：两段内核 + 主机折叠。
///
/// 1. **部分平方和**：复用 [`Metal::chunk_reduce`]，把 `x` 切成块、每线程用 `_cutile_sumsq_f32`
///    归约一块、部分和写 `y[g]`；
/// 2. **归一化**：每线程 `g` 用主机折叠出的 `rms`（`sqrt(Σx²/N + eps)`）归一化一块。
pub fn metal_chunk_rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<(String, String)> {
    let partial = Metal::chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        "_cutile_sumsq_f32",
        PARTIAL,
    )?;
    let len = length_scalar(signature, sample)?;
    let map = chunk_map_kernel(
        &Metal::address_text(kernel_source),
        NORM,
        "    device float* d_rms [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_gamma [[buffer(2)]],\n    device float* d_y [[buffer(3)]]",
        chunk,
        len,
        "    float rms = d_rms[0];\n    for (int j = 0; j < len; ++j) {{\n        d_y[start + j] = d_gamma[start + j] * d_x[start + j] / rms;\n    }}",
    );
    Ok((partial, map))
}

/// 生产态分块 Softmax：三段内核 + 两次主机折叠。
///
/// 1. **部分最大值**：复用 [`Metal::chunk_reduce`]，`_cutile_max_f32` 归约一块、部分最大写 `y[g]`；
/// 2. **部分指数和**：每线程 `g` 用主机折叠出的 `max` 做 `_cutile_exp_sum_f32(x, …, max)`，写 `y[g]`；
/// 3. **归一化**：每线程 `g` 用 `max`/`sum` 做 `_cutile_exp_norm_f32`，写 `y` 全行。
pub fn metal_chunk_softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<(String, String, String)> {
    let max = Metal::chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        "_cutile_max_f32",
        PARTIAL,
    )?;
    let len = length_scalar(signature, sample)?;
    let preamble = Metal::address_text(kernel_source);
    let expsum = chunk_map_kernel(
        &preamble,
        EXPSUM,
        "    device float* d_max [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_y [[buffer(2)]]",
        chunk,
        len,
        "    d_y[g] = _cutile_exp_sum_f32(d_x + start, len, 1, d_max[0]);",
    );
    let map = chunk_map_kernel(
        &preamble,
        NORM,
        "    device float* d_max [[buffer(0)]],\n    device float* d_sum [[buffer(1)]],\n    device float* d_x [[buffer(2)]],\n    device float* d_y [[buffer(3)]]",
        chunk,
        len,
        "    _cutile_exp_norm_f32(d_y + start, d_x + start, len, 1, d_max[0], d_sum[0]);",
    );
    Ok((max, expsum, map))
}

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
    let body = fill(
        BENCH_BODY,
        &[
            ("round", METAL_ROUND),
            ("repeat", &repeat.to_string()),
            ("sets", &wiring.sets),
            ("grid_x", &grid[0].to_string()),
            ("grid_y", &grid[1].to_string()),
            ("grid_z", &grid[2].to_string()),
            ("thread_x", &threads[0].to_string()),
            ("thread_y", &threads[1].to_string()),
            ("thread_z", &threads[2].to_string()),
        ],
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

// ---------------------------------------------------------------------------
// Metal 主机侧胶水：`xcrun metal` 编设备源、`clang` 链接主机驱动、执行与回读。
// ---------------------------------------------------------------------------

/// 多段驱动的一段：入口名、按 `buf(i)` 顺序的缓冲变量名（带 `b_` 前缀）、派发几何、段后主机代码。
///
/// `fold` 是这一段落定之后、下一段之前的主机折叠（部分和 → 总和、部分平方和 → `rms`、
/// 携带前缀推进），空串表示这一段不需要折叠。几何按「工作项数」给，线程组恒为 `(1,1,1)`：
/// 分块族是每线程一块，块内串行走完。
pub(crate) struct MetalStage<'a> {
    pub entry: &'a str,
    pub bufs: &'a [&'a str],
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
    let wiring = Wiring::of(signature, sample, Some(capacity), true, 12)?;
    let mut allocs = wiring.allocs;
    for (name, len) in scratch {
        allocs.push_str(&output_alloc(name, "float", *len, true));
    }
    let body = format!(
        "        id<MTLCommandQueue> queue = [device newCommandQueue];\n{}",
        metal_stages(stages)
    );
    Ok(host_program(
        IMPORTS_MATH,
        &wiring.data,
        host_library(),
        &allocs,
        &body,
        &output,
        read,
    ))
}

/// 段体：每段一份发射 + 该段自己的主机折叠（折叠串由调用方给，不在驱动里拼第二份算式）。
fn metal_stages(stages: &[MetalStage<'_>]) -> String {
    let mut body = String::new();
    for (idx, stage) in stages.iter().enumerate() {
        body.push_str(&metal_stage(idx, stage));
        body.push_str(stage.fold);
    }
    body
}

/// 单段发射：取入口函数、建管线、设缓冲、派发、提交并等待。
fn metal_stage(idx: usize, stage: &MetalStage<'_>) -> String {
    let mut sets = String::new();
    for (slot, name) in stage.bufs.iter().enumerate() {
        sets.push_str(&format!(
            "            [encoder{idx} setBuffer:b_{name} offset:0 atIndex:{slot}];\n"
        ));
    }
    format!(
        r#"        {{
            id<MTLFunction> function{idx} = [library newFunctionWithName:@"{entry}"];
            if (!function{idx}) {{ fprintf(stderr, "metallib 里没有 {entry}\n"); return 5; }}
            id<MTLComputePipelineState> pipeline{idx} = [device newComputePipelineStateWithFunction:function{idx} error:&error];
            if (!pipeline{idx}) {{ fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String); return 6; }}
            id<MTLCommandBuffer> buffer{idx} = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder{idx} = [buffer{idx} computeCommandEncoder];
            [encoder{idx} setComputePipelineState:pipeline{idx}];
{sets}            [encoder{idx} dispatchThreads:MTLSizeMake({gx}, {gy}, {gz}) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder{idx} endEncoding];
            [buffer{idx} commit];
            [buffer{idx} waitUntilCompleted];
            if (buffer{idx}.error) {{ fprintf(stderr, "内核执行失败：%s\n", buffer{idx}.error.localizedDescription.UTF8String); return 7; }}
        }}
"#,
        entry = stage.entry,
        sets = sets,
        gx = stage.grid[0],
        gy = stage.grid[1],
        gz = stage.grid[2],
    )
}

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
        prologue = host_prologue("cutile_witness")
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

// ---------------------------------------------------------------------------
// Metal 多段并行程序：分块族（归约 / 归一化 / 扫描）的「多段发射 + 主机折叠」。
// ---------------------------------------------------------------------------

/// 折叠的一步：与 `.k` 模板的 `reduce.op` 同口径，`sum` / `max` 两个闭集表达。
const SUM_STEP: &str = "acc += parts[g];";
const MAX_STEP: &str = "if (parts[g] > acc) { acc = parts[g]; }";

/// 一段程序的公共几何：长度、块数、输出缓冲名、全部缓冲名。
struct Prep<'a> {
    len: usize,
    blocks: usize,
    output: &'a str,
    names: Vec<&'a str>,
}

impl<'a> Prep<'a> {
    fn of(signature: &'a Signature, sample: &Sample, chunk: usize) -> Result<Self> {
        let len = length_usize(signature, sample)?;
        Ok(Self {
            len,
            blocks: len.div_ceil(chunk.max(1)),
            output: output_name(signature)?,
            names: buf_names(signature),
        })
    }

    /// 输出缓冲的容量：它同时当「部分结果数组」与「整行结果」用，取大者开满，否则局部扫描 /
    /// 归一化写回整行时越界（Metal 不做边界检查，错的是内存不是报告）。
    fn span(&self) -> usize {
        self.blocks.max(self.len)
    }
}

/// 分块归约程序：一次部分归约段 + 主机折叠（`sum` / `max` 由 `reduce.op` 派生）。
///
/// # Errors
///
/// 长度标量缺失、签名与样本不一致，或设备段生成失败。
pub fn reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    func: &str,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    // `sum` / `max` 取哪一步由 `.k` 模板的 `reduce.op` 派生，折叠本身只有一份。
    let device = Metal::chunk_reduce(kernel_source, signature, sample, chunk, func)?;
    let step = if func.contains("max") {
        MAX_STEP
    } else {
        SUM_STEP
    };
    let fold = fold(&format!("b_{}", prep.output), None, prep.blocks, step);
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        1,
        &[],
        &[MetalStage {
            entry: "kernel_main",
            bufs: &prep.names,
            grid: [prep.blocks, 1, 1],
            fold: &fold,
        }],
    )?;
    Ok(Source::Pair { device, host })
}

/// 分块 RMSNorm 程序：部分平方和段 → `rms` 折叠 → 归一化段。
///
/// # Errors
///
/// 长度标量缺失、签名不是「两个只读缓冲 + 输出 + 标量」的归一化形态，或设备段生成失败。
pub fn rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    let (partial, norm) = metal_chunk_rmsnorm(kernel_source, signature, sample, chunk)?;
    let device = concat(&partial, &[(norm.as_str(), NORM)])?;
    let (x, gamma) = normalization_reads(signature)?;
    let fold = rms_fold(prep.output, prep.blocks, prep.len);
    let map = ["rms", x, gamma, prep.output];
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        prep.len,
        &[("rms", 1)],
        &[
            MetalStage {
                entry: PARTIAL,
                bufs: &prep.names,
                grid: [prep.blocks, 1, 1],
                fold: &fold,
            },
            MetalStage {
                entry: NORM,
                bufs: &map,
                grid: [prep.blocks, 1, 1],
                fold: "",
            },
        ],
    )?;
    Ok(Source::Pair { device, host })
}

/// 分块 Softmax 程序：部分最大段 → `max` 折叠 → 部分指数和段 → `sum` 折叠 → 归一化段。
///
/// # Errors
///
/// 长度标量缺失、签名不是「一个只读缓冲 + 输出 + 标量」的 softmax 形态，或设备段生成失败。
pub fn softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    // `max` 与 `sum` 各落一个单独缓冲，留给下一段当入口参数。
    let (max, expsum, norm) = metal_chunk_softmax(kernel_source, signature, sample, chunk)?;
    let device = concat(&max, &[(expsum.as_str(), EXPSUM), (norm.as_str(), NORM)])?;
    let x = only_read(signature)?;
    let parts = format!("b_{}", prep.output);
    let fold_max = fold(&parts, Some("b_max"), prep.blocks, MAX_STEP);
    let fold_sum = fold(&parts, Some("b_sum"), prep.blocks, SUM_STEP);
    let exp_bufs = ["max", x, prep.output];
    let norm_bufs = ["max", "sum", x, prep.output];
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        prep.len,
        &[("max", 1), ("sum", 1)],
        &[
            MetalStage {
                entry: PARTIAL,
                bufs: &prep.names,
                grid: [prep.blocks, 1, 1],
                fold: &fold_max,
            },
            MetalStage {
                entry: EXPSUM,
                bufs: &exp_bufs,
                grid: [prep.blocks, 1, 1],
                fold: &fold_sum,
            },
            MetalStage {
                entry: NORM,
                bufs: &norm_bufs,
                grid: [prep.blocks, 1, 1],
                fold: "",
            },
        ],
    )?;
    Ok(Source::Pair { device, host })
}

/// 分块扫描程序：局部扫描段 → 携带前缀折叠（f64）→ 校正段。
///
/// # Errors
///
/// 长度标量缺失、模板里找不到以长度标量为上界的循环，或设备段生成失败。
pub fn scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<Source> {
    let prep = Prep::of(signature, sample, chunk)?;
    // 携带前缀的 `a^len` 用平方乘，块长最多 `chunk` 次乘法。
    let (local, correct) = metal_chunk_scan(kernel_source, signature, sample, chunk, decay)?;
    let device = concat(&local, &[(correct.as_str(), SCAN_CORRECT)])?;
    let fold = fold_carry(prep.output, chunk, prep.len, decay);
    let correct_bufs = ["c", prep.output];
    let host = metal_multistage_driver(
        signature,
        sample,
        prep.span(),
        prep.len,
        &[("c", prep.blocks)],
        &[
            MetalStage {
                entry: SCAN_LOCAL,
                bufs: &prep.names,
                grid: [prep.blocks, 1, 1],
                fold: &fold,
            },
            MetalStage {
                entry: SCAN_CORRECT,
                bufs: &correct_bufs,
                grid: [prep.blocks, 1, 1],
                fold: "",
            },
        ],
    )?;
    Ok(Source::Pair { device, host })
}

/// 签名里全部缓冲的名字（按 `buf(i)` 顺序）。
fn buf_names(signature: &Signature) -> Vec<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .map(|param| param.name.as_str())
        .collect()
}

/// 归一化的两个只读缓冲（`x` 与 `gamma`）：少一个就做不了 RMSNorm，在入口拒绝。
fn normalization_reads(signature: &Signature) -> Result<(&str, &str)> {
    let reads = readonly_bufs(signature);
    reads
        .first()
        .zip(reads.get(1))
        .map(|(first, second)| (*first, *second))
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                "RMSNorm 需要两个只读缓冲（x, gamma）".to_owned(),
            )
            .with("engine::metal::rmsnorm")
        })
}

/// Softmax 的唯一只读缓冲（`x`）：分布定义在整行上，多给一个就说不清算谁。
fn only_read(signature: &Signature) -> Result<&str> {
    readonly_bufs(signature).first().copied().ok_or_else(|| {
        Error::new(Code::Invalid, "Softmax 需要一个只读缓冲（x）".to_owned())
            .with("engine::metal::softmax")
    })
}

/// `sqrt(Σx²/N + ε)` 折叠：f64 求和、写回前收窄 f32（与 `.k` 模板同口径）。
fn rms_fold(output: &str, blocks: usize, len: usize) -> String {
    format!(
        "        {{\n            const float *parts = (const float *)b_{output}.contents;\n            double acc = 0.0;\n            for (int g = 0; g < {blocks}; ++g) {{ acc += (double)parts[g]; }}\n            ((float *)b_rms.contents)[0] = (float)sqrt(acc / (double){len} + (double){eps}f);\n        }}\n",
        eps = RMS_EPS
    )
}

/// 多段设备源拼接：第一段带整份 preamble，后续段只取自己的 `kernel void <entry>` 之后——
/// 同一翻译单元里 preamble 出现一次即可，重复会撞重定义。
///
/// # Errors
///
/// 生成的段里找不到该入口函数（生成器与入口名漂移）时返回 `Internal`。
fn concat(first: &str, more: &[(&str, &str)]) -> Result<String> {
    let mut text = first.to_string();
    for (segment, entry) in more {
        let marker = format!("kernel void {entry}(");
        let at = segment.find(&marker).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!("段 `{entry}` 不在生成的设备源里（漂移）"),
            )
            .with("engine::metal::concat")
        })?;
        text.push_str("\n\n");
        text.push_str(&segment[at..]);
    }
    Ok(text)
}

/// 部分结果折叠：把 `parts[0..blocks]` 折成一个值落地。`target` 为 `None` 时就在 `parts[0]` 就地写
/// （归约）；`Some(buf)` 则写给留给下一段的独立缓冲（Softmax 的 `max` / `sum`）。
fn fold(parts: &str, target: Option<&str>, blocks: usize, step: &str) -> String {
    let (read, write) = match target {
        None => (
            format!("float *parts = (float *){parts}.contents;"),
            "parts[0] = acc;".to_string(),
        ),
        Some(target) => (
            format!("const float *parts = (const float *)({parts}).contents;"),
            format!("((float *)({target}).contents)[0] = acc;"),
        ),
    };
    format!(
        "        {{\n            {read}\n            float acc = parts[0];\n            for (int g = 1; g < {blocks}; ++g) {{ {step} }}\n            {write}\n        }}\n"
    )
}

/// 携带前缀折叠（f64）：`C_0 = 0`、`C_{g+1} = a^len·C_g + y[start+len-1]`，`a^len` 用平方乘。
fn fold_carry(output: &str, chunk: usize, len: usize, decay: f32) -> String {
    let blocks = len.div_ceil(chunk.max(1));
    format!(
        "        {{\n            const float *y = (const float *)b_{output}.contents;\n            float *c = (float *)b_c.contents;\n{fold}        }}\n",
        fold = carry_fold(
            "y",
            "c",
            &blocks.to_string(),
            chunk,
            len as i64,
            decay,
            "            "
        ),
    )
}
