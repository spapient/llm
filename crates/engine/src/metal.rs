//! Metal(MSL) 侧的设备源与 host 驱动拼装：`device` 地址空间、`[[buffer(slot)]]` 缓冲、compute
//! kernel 发射、`MTLBuffer` 分配 / 回读骨架。
//!
//! 后端无关的 lowering 原语住在 [`crate::dialect::Dialect`] trait 的**默认方法**里，这里只写
//! Metal 一侧的 vendor 驱动骨架（`xcrun metal` 的 MSL 文本 + clang 的 ObjC driver 文本）。同一条
//! kernel body 由 [`Metal`](crate::dialect::Metal) 方言落地，与 CUDA 侧的 [`crate::cuda`] 是
//! 同层的一件事：只换方言，不改语义，回读口径（`float` 位模式逐行打印）必须逐字一致。

use diagnostic::{Code, Error, Result};
use op::decl::Grid;
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use cutile::harness::{c_literal, output_name, output_of};

use crate::dialect::{
    Dialect, EXPSUM, Metal, NORM, PARTIAL, SCAN_CORRECT, chunk_head, f32_output_shape, launch_args,
    length_scalar, readback_bits, scalar_name, scan_correct_body,
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
    Ok((msl, metal_driver(signature, sample, output)?))
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

/// 生产态 Metal 分块归约内核（[`crate::dialect::Dialect::chunk_reduce`] 的 Metal 方言）。
pub fn metal_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    Metal::chunk_reduce(kernel_source, signature, sample, chunk, reduce_func)
}

/// [`metal_chunk_reduce`] 的具名版本：多段程序里部分归约段要与其他段区分入口名。
pub fn metal_chunk_reduce_entry(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
    entry: &str,
) -> Result<String> {
    Metal::chunk_reduce_entry(kernel_source, signature, sample, chunk, reduce_func, entry)
}

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
    buffers: &str,
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
        "{head}kernel void {entry}(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n{chunk_lines}{body}\n}}"
    )
}

/// 校正内核：每线程 `g` 把携带态 `d_c[g]` 的衰减几何级数 `C·a^(j+1)` 加回 `y`（就地累加）。
/// 槽位按「输入后输出」：`d_c` 是输入在 `buffer(0)`、`d_y` 是输出在 `buffer(1)`。
fn chunk_scan_correct(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    let buffers =
        format!("    device float* d_c [[buffer(0)]],\n    device float* d_{output} [[buffer(1)]]");
    let body = scan_correct_body(output, decay, "    ");
    chunk_map_kernel("", SCAN_CORRECT, &buffers, chunk, len, &body)
}

/// 生产态分块 RMSNorm：两段内核 + 主机折叠。
///
/// 1. **部分平方和**：复用 [`metal_chunk_reduce`]，把 `x` 切成块、每线程用 `_cutile_sumsq_f32`
///    归约一块、部分和写 `y[g]`；
/// 2. **归一化**：每线程 `g` 用主机折叠出的 `rms`（`sqrt(Σx²/N + eps)`）归一化一块。
pub fn metal_chunk_rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<(String, String)> {
    let partial = metal_chunk_reduce_entry(
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
/// 1. **部分最大值**：复用 [`metal_chunk_reduce`]，`_cutile_max_f32` 归约一块、部分最大写 `y[g]`；
/// 2. **部分指数和**：每线程 `g` 用主机折叠出的 `max` 做 `_cutile_exp_sum_f32(x, …, max)`，写 `y[g]`；
/// 3. **归一化**：每线程 `g` 用 `max`/`sum` 做 `_cutile_exp_norm_f32`，写 `y` 全行。
pub fn metal_chunk_softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<(String, String, String)> {
    let max = metal_chunk_reduce_entry(
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

/// `cutile_launch` compute kernel：按签名顺序声明 `device` 缓冲并原样透传给内核；标量直接取样本值。
fn metal_launch(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = Metal::buffers(signature);
    Ok(format!(
        "kernel void cutile_launch({buffers},\n                          uint tid [[thread_position_in_grid]]) {{\n    if (tid != 0) return;\n    {kernel}({call});\n}}"
    ))
}

/// 缓冲绑定段：只读输入的初值、`MTLBuffer` 分配、`setBuffer` 槽位与可写输出的元素数，按签名序
/// 一次生成。单线程语义门与网格发射只在「入口名 + 发射几何」上不同，绑定与回读必须逐字同源。
fn metal_bindings(
    signature: &Signature,
    sample: &Sample,
) -> Result<(String, String, String, usize)> {
    let mut data = String::new();
    let mut allocs = String::new();
    let mut sets = String::new();
    let mut out_len = 0usize;
    for (slot, param) in signature
        .params()
        .iter()
        .filter(|p| p.ctype.is_buffer())
        .enumerate()
    {
        let buffer = sample
            .buffer(&param.name)
            .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
        let base = param.ctype.c_name();
        let len = buffer.values.len();
        if param.mutable {
            out_len = len;
            allocs.push_str(&format!(
                "        id<MTLBuffer> b_{name} = [device newBufferWithLength:{len} * sizeof({base}) options:MTLResourceStorageModeShared];\n",
                name = param.name,
                len = len,
                base = base
            ));
        } else {
            data.push_str(&format!(
                "static const {base} init_{name}[] = {{{lit}}};\n",
                base = base,
                name = param.name,
                lit = c_literal(&buffer.values)
            ));
            allocs.push_str(&format!(
                "        id<MTLBuffer> b_{name} = [device newBufferWithBytes:init_{name} length:sizeof(init_{name}) options:MTLResourceStorageModeShared];\n",
                name = param.name
            ));
        }
        sets.push_str(&format!(
            "        [encoder setBuffer:b_{name} offset:0 atIndex:{slot}];\n",
            name = param.name,
            slot = slot
        ));
    }
    Ok((data, allocs, sets, out_len))
}

/// host 驱动（单线程语义门）：把样本数据作为只读缓冲的初值，发射 `cutile_launch`，回读唯一可写
/// 输出的位模式——验证「算得对不对」，不回答「跑得多快」。
fn metal_driver(signature: &Signature, sample: &Sample, output: &str) -> Result<String> {
    let (data, allocs, sets, out_len) = metal_bindings(signature, sample)?;
    Ok(metal_driver_template(
        &data,
        &allocs,
        &sets,
        output,
        out_len,
        "cutile_launch",
        [1; 3],
        [1; 3],
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
    let (data, allocs, sets, out_len) = metal_bindings(signature, sample)?;
    Ok(metal_driver_template(
        &data, &allocs, &sets, output, out_len, entry, grid, block,
    ))
}

/// 生产态 Metal 基准驱动：host 只负责输入上传与派发，样本计时取自 command buffer 的 GPU 时间戳。
pub fn metal_benchmark_driver(
    signature: &Signature,
    sample: &Sample,
    grid: [usize; 3],
    threads: [usize; 3],
    repeat: usize,
) -> Result<String> {
    let (output, out_len) = f32_output_shape(signature, sample)?;
    let (data, allocs, sets) = bench_wiring(signature, sample)?;
    let source = r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

@DATA@
int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: driver <metallib>\n"); return 2; }
@PROLOGUE@@ALLOCS@        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { fprintf(stderr, "Metal 命令队列创建失败\n"); return 7; }
        for (int warm = 0; warm < 2; ++warm) {
@ROUND@            if (buffer.error) { fprintf(stderr, "内核预热失败：%s\n", buffer.error.localizedDescription.UTF8String); return 8; }
        }
        for (int sample = 0; sample < @REPEAT@; ++sample) {
@ROUND@            if (buffer.error) { fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String); return 9; }
            double elapsed = (buffer.GPUEndTime - buffer.GPUStartTime) * 1.0e9;
            if (!(elapsed > 0.0)) { fprintf(stderr, "Metal GPU 时间戳不可用\n"); return 10; }
            fprintf(stderr, "ns %.3f\n", elapsed / 16.0);
        }
        const float *out = (const float *)b_@OUTPUT@.contents;
@READBACK@
    }
    return 0;
}
"#
    .replace("@DATA@", &data)
    .replace("@ALLOCS@", &allocs)
    .replace("@ROUND@", METAL_ROUND)
    .replace("@PROLOGUE@", &host_prologue("kernel_main"))
    .replace("@SETS@", &sets)
    .replace("@GRID_X@", &grid[0].to_string())
    .replace("@GRID_Y@", &grid[1].to_string())
    .replace("@GRID_Z@", &grid[2].to_string())
    .replace("@THREAD_X@", &threads[0].to_string())
    .replace("@THREAD_Y@", &threads[1].to_string())
    .replace("@THREAD_Z@", &threads[2].to_string())
    .replace("@REPEAT@", &repeat.to_string())
    .replace("@OUTPUT@", &output)
    .replace("@READBACK@", &readback_bits("out", out_len, "        "));
    Ok(source)
}

/// 按签名顺序给每条缓冲写「声明 / 分配 / 绑定」三份文本：可变的是输出，不可变的先落成静态初值。
fn bench_wiring(signature: &Signature, sample: &Sample) -> Result<(String, String, String)> {
    let mut data = String::new();
    let mut allocs = String::new();
    let mut sets = String::new();
    for (slot, param) in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .enumerate()
    {
        let buffer = sample
            .buffer(&param.name)
            .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
        let base = param.ctype.c_name();
        let name = param.name.as_ref();
        let len = buffer.values.len();
        if param.mutable {
            allocs.push_str(&output_alloc(name, base, len));
        } else {
            data.push_str(&literal_decl(name, base, &buffer.values));
            allocs.push_str(&input_alloc(name));
        }
        sets.push_str(&format!(
            "            [encoder setBuffer:b_{name} offset:0 atIndex:{slot}];\n",
        ));
    }
    Ok((data, allocs, sets))
}

/// 输入缓冲的静态初值：一份数值表，与参数同名的数组。
fn literal_decl(name: &str, base: &str, values: &Values) -> String {
    format!(
        "static const {base} init_{name}[] = {{{}}};\n",
        c_literal(values)
    )
}

/// 输出缓冲的分配：可变、按元素数一次开满。
fn output_alloc(name: &str, base: &str, len: usize) -> String {
    format!(
        "        id<MTLBuffer> b_{name} = [device newBufferWithLength:{len} * sizeof({base}) options:MTLResourceStorageModeShared];\n\
         \x20       if (!b_{name}) {{ fprintf(stderr, \"Metal 输出缓冲分配失败\\n\"); return 7; }}\n",
    )
}

/// 输入缓冲的分配：按静态初值的长度原样拷入。
fn input_alloc(name: &str) -> String {
    format!(
        "        id<MTLBuffer> b_{name} = [device newBufferWithBytes:init_{name} length:sizeof(init_{name}) options:MTLResourceStorageModeShared];\n\
         \x20       if (!b_{name}) {{ fprintf(stderr, \"Metal 输入缓冲分配失败\\n\"); return 7; }}\n",
    )
}

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
const METAL_ROUND: &str = r#"            id<MTLCommandBuffer> buffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
@SETS@            for (int batch = 0; batch < 16; ++batch) {
                [encoder dispatchThreads:MTLSizeMake(@GRID_X@, @GRID_Y@, @GRID_Z@)
                    threadsPerThreadgroup:MTLSizeMake(@THREAD_X@, @THREAD_Y@, @THREAD_Z@)];
            }
            [encoder endEncoding];
            [buffer commit];
            [buffer waitUntilCompleted];
"#;

/// 驱动正文模板：只拼字面与已算好的片段，不含循环与错误处理（那些留在 `metal_driver`）。
#[allow(clippy::too_many_arguments)]
fn metal_driver_template(
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
