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

pub mod chunk;
#[cfg(target_os = "macos")]
pub mod executor;
pub mod fold;
pub mod host;

pub use chunk::{
    metal_benchmark_driver, metal_chunk_reduce, metal_chunk_reduce_entry, metal_chunk_rmsnorm,
    metal_chunk_scan, metal_chunk_softmax,
};
use host::metal_driver_template;
pub(crate) use host::{MetalStage, metal_multistage_driver};

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
