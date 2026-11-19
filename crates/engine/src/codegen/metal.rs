//! 生产态 Metal(MSL) lowering：把一份 `.k` 内核按声明式 `[grid]`/`[chunk]` 降成设备源，
//! 同一条内核体只拼一次，后端差异从 [`super`] 的方言表取。host 侧计量与基准封装在 `cutile::harness`；
//! CUDA 方言与其逐点/归约 lowering 见 [`super::cuda`]，本模块只做 Metal。

use diagnostic::{Code, Error, Result};
use op::decl::Grid;
use op::model::Sample;
use op::sig::{Param, Signature};

use super::{buffer_decls, chunk_reduce, grid_kernel, launch_args, length_scalar, msl_address_space, scalar_name, METAL};
use cutile::harness::{c_literal, output_name, output_of};

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
        msl_address_space(kernel_source),
        metal_launch(&signature.kernel(), signature, sample)?
    );
    Ok((msl, metal_driver(signature, sample, output)?))
}

/// 生产态 Metal 逐点网格内核（[`grid_kernel`] 的 Metal 方言）。
pub fn metal_grid_kernel(kernel_source: &str, signature: &Signature) -> Result<String> {
    grid_kernel(kernel_source, signature, &METAL)
}

/// 串行实参：缓冲取 `d_{name}`（原样透传，单线程处理全部），标量取样本里的实际值。
fn serial_args(params: &[Param], sample: &Sample) -> Result<String> {
    params
        .iter()
        .map(|param| {
            if param.ctype.is_buffer() {
                Ok(format!("d_{}", param.name))
            } else {
                let value = sample.scalar(&param.name).ok_or_else(|| {
                    Error::new(
                        Code::Invalid,
                        format!("串行内核标量 `{}` 没有样本取值", param.name),
                    )
                    .with("engine::codegen::metal::metal_loop_kernel")
                })?;
                Ok(value.to_string())
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(|args| args.join(", "))
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
        return serial_kernel(kernel_source, signature, sample);
    }
    injected_kernel(kernel_source, signature, sample, grid)
}

/// 串行（零维网格）特例：不追加 `tid`、不绑循环，缓冲原样透传、标量取样本值。
fn serial_kernel(kernel_source: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = serial_args(signature.params(), sample)?;
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{}\n\nkernel void kernel_main(\n{buffers}) {{\n    {}({call});\n}}",
        msl_address_space(kernel_source),
        signature.kernel(),
    ))
}

/// 循环注入（非零网格）：给内部内核追加 `uint3 tid`、绑循环头到线程下标、调用点透传 `tid`。
fn injected_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    let mut body = msl_address_space(kernel_source);
    body = thread_param(&body, &signature.kernel())?;
    body = bind_grid_loops(body, grid)?;
    let call = format!("{}, tid", serial_args(signature.params(), sample)?);
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{body}\n\nkernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    {}({call});\n}}",
        signature.kernel(),
    ))
}

/// 生产态 Metal 分块归约内核（[`chunk_reduce`] 的 Metal 方言）。
pub fn metal_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &METAL)
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
    let local = chunk_scan_local(kernel_source, signature, sample, chunk, scalar)?;
    let correct = chunk_scan_correct(output, chunk, len, decay);
    Ok((local, correct))
}

/// 局部扫描内核：把模板里的时间轴循环头改写到块内 `[start, start+len)`，`tid` 透传进内部内核。
fn chunk_scan_local(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    scalar: &str,
) -> Result<String> {
    let kernel = signature.kernel();
    let mut body = msl_address_space(kernel_source);
    body = thread_param(&body, &kernel)?;
    body = inject_chunk_preamble(&body, &kernel, chunk, scalar)?;
    body = bind_chunk_loop(&body, scalar)?;
    let call = format!("{}, tid", serial_args(signature.params(), sample)?);
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{body}\n\nkernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    {kernel}({call});\n}}"
    ))
}

/// 校正内核：每线程 `g` 把携带态 `d_c[g]` 的衰减几何级数 `C·a^(j+1)` 加回 `y`（就地累加）。
/// 槽位按「输入后输出」：`d_c` 是输入在 `buffer(0)`、`d_y` 是输出在 `buffer(1)`。
fn chunk_scan_correct(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    format!(
        "kernel void kernel_main(\n    device float* d_c [[buffer(0)]],\n    device float* d_{output} [[buffer(1)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    float c = d_c[g];\n    float pow = {decay}f;\n    for (int j = 0; j < len; ++j) {{\n        d_{output}[start + j] += c * pow;\n        pow *= {decay}f;\n    }}\n}}"
    )
}

/// 给内部内核函数体注入块头：`g`/`start`/`len` 三个局部量，供循环头改写与分块寻址引用。
fn inject_chunk_preamble(
    body: &str,
    kernel: &str,
    chunk: usize,
    scalar: &str,
) -> Result<String> {
    let marker = format!("void {kernel}(");
    let Some(signature_start) = body.find(&marker) else {
        return Err(Error::new(
            Code::Invalid,
            format!("找不到内核函数 `void {kernel}(`"),
        )
        .with("engine::codegen::metal::metal_chunk_scan"));
    };
    let Some(close) = body[signature_start..].find(") {") else {
        return Err(Error::new(
            Code::Invalid,
            format!("内核函数 `{kernel}` 签名不完整"),
        )
        .with("engine::codegen::metal::metal_chunk_scan"));
    };
    let brace = signature_start + close + 2;
    let preamble = format!(
        "    int g = (int)tid.x;\n    int start = g * {chunk};\n    int len = min({chunk}, {scalar} - start);\n"
    );
    let mut out = String::with_capacity(body.len() + preamble.len());
    out.push_str(&body[..=brace]);
    out.push_str(&preamble);
    out.push_str(&body[brace + 1..]);
    Ok(out)
}

/// 把以 `scalar` 为上界的串行循环头改写到块内边界：`for (int v = 0; v < {scalar}; ++v)` →
/// `for (int v = start; v < start + len; ++v)`。循环变量从模板读出，不对名字硬编码。
fn bind_chunk_loop(body: &str, scalar: &str) -> Result<String> {
    let bound = format!(" < {scalar}; ");
    let Some(bound_at) = body.find(&bound) else {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到以 `{scalar}` 为上界的串行循环"),
        )
        .with("engine::codegen::metal::metal_chunk_scan"));
    };
    let prefix = &body[..bound_at];
    let Some(open) = prefix.rfind("for (int ") else {
        return Err(Error::new(
            Code::Invalid,
            "循环上界不在 `for (int …)` 头里",
        )
        .with("engine::codegen::metal::metal_chunk_scan"));
    };
    let var = prefix[open + "for (int ".len()..]
        .split('=')
        .next()
        .unwrap_or("")
        .trim();
    let old_head = format!("for (int {var} = 0; {var} < {scalar}; ++{var})");
    let new_head = format!("for (int {var} = start; {var} < start + len; ++{var})");
    if !body.contains(&old_head) {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到串行循环 `{old_head}`"),
        )
        .with("engine::codegen::metal::metal_chunk_scan"));
    }
    Ok(body.replace(&old_head, &new_head))
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
    let partial = metal_chunk_reduce(kernel_source, signature, sample, chunk, "_cutile_sumsq_f32")?;
    let len = length_scalar(signature, sample)?;
    let map = format!(
        "{}\n\nkernel void kernel_main(\n    device float* d_rms [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_gamma [[buffer(2)]],\n    device float* d_y [[buffer(3)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    float rms = d_rms[0];\n    for (int j = 0; j < len; ++j) {{\n        d_y[start + j] = d_gamma[start + j] * d_x[start + j] / rms;\n    }}\n}}",
        msl_address_space(kernel_source),
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
    let max = metal_chunk_reduce(kernel_source, signature, sample, chunk, "_cutile_max_f32")?;
    let len = length_scalar(signature, sample)?;
    let preamble = msl_address_space(kernel_source);
    let expsum = format!(
        "{preamble}\n\nkernel void kernel_main(\n    device float* d_max [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_y [[buffer(2)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    d_y[g] = _cutile_exp_sum_f32(d_x + start, len, 1, d_max[0]);\n}}"
    );
    let map = format!(
        "{preamble}\n\nkernel void kernel_main(\n    device float* d_max [[buffer(0)]],\n    device float* d_sum [[buffer(1)]],\n    device float* d_x [[buffer(2)]],\n    device float* d_y [[buffer(3)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    _cutile_exp_norm_f32(d_y + start, d_x + start, len, 1, d_max[0], d_sum[0]);\n}}"
    );
    Ok((max, expsum, map))
}

fn bind_grid_loops(body: String, grid: &Grid) -> Result<String> {
    const AXES: [&str; 3] = ["x", "y", "z"];
    let mut out = body;
    for (dim, (thread, scalar)) in grid.dims.iter().enumerate() {
        let axis = AXES.get(dim).ok_or(Error::new(
            Code::Invalid,
            format!("网格维 {} 超过 Metal 上限 3", grid.dims.len()),
        ))?;
        out = bind_grid_loop(&out, thread, scalar, axis)?;
    }
    Ok(out)
}

/// 给内部内核函数追加 `uint3 tid` 形参：`void {kernel}(...) {` → `void {kernel}(..., uint3 tid) {`。
fn thread_param(body: &str, kernel: &str) -> Result<String> {
    let marker = format!("void {kernel}(");
    let Some(signature_start) = body.find(&marker) else {
        return Err(Error::new(
            Code::Invalid,
            format!("找不到内核函数 `void {kernel}(`"),
        )
        .with("engine::codegen::metal::metal_loop_kernel"));
    };
    let Some(close) = body[signature_start..].find(") {") else {
        return Err(Error::new(
            Code::Invalid,
            format!("内核函数 `{kernel}` 签名不完整"),
        )
        .with("engine::codegen::metal::metal_loop_kernel"));
    };
    let insert_at = signature_start + close;
    let mut out = String::with_capacity(body.len() + 12);
    out.push_str(&body[..insert_at]);
    out.push_str(", uint3 tid");
    out.push_str(&body[insert_at..]);
    Ok(out)
}

/// 把一个网格维的串行循环头绑定到线程下标：`for (int {thread} = 0; {thread} < {scalar}; ++{thread})`
/// → `for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})`。
fn bind_grid_loop(body: &str, thread: &str, scalar: &str, axis: &str) -> Result<String> {
    let head = format!("for (int {thread} = 0; {thread} < {scalar}; ++{thread})");
    let bound = format!(
        "for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})"
    );
    if !body.contains(&head) {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到网格循环 `{head}`"),
        )
        .with("engine::codegen::metal::metal_loop_kernel"));
    }
    Ok(body.replace(&head, &bound))
}

/// `cutile_launch` compute kernel：按签名顺序声明 `device` 缓冲并原样透传给内核；标量直接取样本值。
fn metal_launch(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .enumerate()
        .map(|(slot, param)| {
            format!(
                "device {} *d_{} [[buffer({slot})]]",
                param.ctype.c_name(),
                param.name
            )
        })
        .collect::<Vec<_>>()
        .join(",\n                          ");
    Ok(format!(
        "kernel void cutile_launch({buffers},\n                          uint tid [[thread_position_in_grid]]) {{\n    if (tid != 0) return;\n    {kernel}({call});\n}}"
    ))
}

/// host 驱动：把样本数据作为只读缓冲的初值，发射 `cutile_launch`，回读唯一可写输出的位模式。
fn metal_driver(signature: &Signature, sample: &Sample, output: &str) -> Result<String> {
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
    Ok(metal_driver_template(&data, &allocs, &sets, output, out_len))
}

/// 驱动正文模板：只拼字面与已算好的片段，不含循环与错误处理（那些留在 [`metal_driver`]）。
fn metal_driver_template(
    data: &str,
    allocs: &str,
    sets: &str,
    output: &str,
    out_len: usize,
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
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {{ fprintf(stderr, "本机没有 Metal 设备\n"); return 3; }}
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {{ fprintf(stderr, "metallib 装载失败：%s\n", error.localizedDescription.UTF8String); return 4; }}
        id<MTLFunction> function = [library newFunctionWithName:@"cutile_launch"];
        if (!function) {{ fprintf(stderr, "metallib 里没有 cutile_launch\n"); return 5; }}
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {{ fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String); return 6; }}
{allocs}        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
{sets}        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {{ fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String); return 7; }}
        const float *out = (const float *)b_{output}.contents;
        for (int i = 0; i < {out_len}; ++i) {{
            union {{ float f; unsigned int u; }} v;
            v.f = out[i];
            printf("%u\n", v.u);
        }}
    }}
    return 0;
}}
"#,
        data = data,
        allocs = allocs,
        sets = sets,
        output = output,
        out_len = out_len
    )
}
