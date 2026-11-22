//! Metal 分块族内核：归约、扫描、`rmsnorm`、`softmax` 的设备源拼装，以及计时基准驱动。
//!
//! 四条 lowering 共用一段「块内并行 + 部分结果落回 `y[g]`」的骨架，差异只在块内算子的组合；
//! 因此它们住一个文件，共享 `chunk_head` / `PARTIAL` / `EXPSUM` 这些从 [`crate::dialect`] 来的原语。

use super::*;

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
        "{head}kernel void {entry}(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n{chunk_lines}{body}\n}}"
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

/// 生产态 Metal 基准驱动：host 只负责输入上传与派发，样本计时取自 command buf 的 GPU 时间戳。
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
        .filter(|param| param.ctype.is_buf())
        .enumerate()
    {
        let buf = sample
            .buf(&param.name)
            .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
        let base = param.ctype.c_name();
        let name = param.name.as_ref();
        let len = buf.values.len();
        if param.mutable {
            allocs.push_str(&output_alloc(name, base, len));
        } else {
            data.push_str(&literal_decl(name, base, &buf.values));
            allocs.push_str(&input_alloc(name));
        }
        sets.push_str(&format!(
            "            [encoder setBuffer:b_{name} offset:0 atIndex:{slot}];\n",
        ));
    }
    Ok((data, allocs, sets))
}
