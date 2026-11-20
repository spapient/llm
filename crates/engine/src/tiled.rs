//! 设备侧分块 lowering：共享内存 GEMM 与（后续）分块注意力。
//!
//! **拥有**：中性 [`cutile::backend::Lower::TiledGemm`] 的 Metal 落地——把 GEMM 骨架的标量三重
//! 循环换成 threadgroup 共享内存分块。**不拥有**：骨架分派（`cutile::execution`）、逐点/循环/
//! 归约/扫描 lowering（`codegen`）、张量核 WMMA（`wmma`，CUDA）。
//!
//! 与 `codegen` 分开的理由：`codegen` 的方言表与 harness 已接近单文件上限，而分块内核是独立的
//! codegen 家族（共享内存 + 线程组屏障），把错误处理与索引算术收在一处更可读。

use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::{Param, Signature};

/// 生产态共享内存分块 f32 GEMM（Metal）：把标量三重循环换成 threadgroup 分块。
///
/// 语义与模板逐元素相同（`Y[i][j] = Σ_p A[i][p]·B[p][j]`），只换遍历：每个 `tile×tile`
/// threadgroup 把 A、B 的一个 `tile×tile` 块协同搬进 threadgroup 内存，规约轴在共享块上复用，
/// 每个输出不再对每次 `p` 重读设备全局内存。前置条件由调用方保证（`M/N/K` 均能被 `tile` 整除）。
///
/// 缓冲 A/B/Y 取签名前三条缓冲，标量 M/N/K 取签名三个整数形参的样本值；网格按 `[M, N]` 派发、
/// 线程组 `[tile, tile]`，`gid.x` 是输出行、`gid.y` 是输出列。
///
/// # Errors
///
/// 签名不足三条缓冲或三个整数标量、样本缺标量、或 `tile == 0` 时返回说明。
pub fn metal_tiled_gemm(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    tile: usize,
) -> Result<String> {
    if tile == 0 {
        return Err(
            Error::new(Code::Invalid, "分块大小必须大于 0").with("engine::tiled::metal_tiled_gemm")
        );
    }
    let (a, b, y) = gemm_buffers(signature)?;
    let (_, n, k) = gemm_dims(signature, sample)?;
    Ok(tiled_gemm_msl(a, b, y, n, k, tile))
}

/// 三条缓冲名（按签名序：A、B、Y）。
fn gemm_buffers(signature: &Signature) -> Result<(&str, &str, &str)> {
    let buffers: Vec<&Param> = signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .collect();
    match buffers.as_slice() {
        [a, b, y] => Ok((a.name.as_str(), b.name.as_str(), y.name.as_str())),
        _ => Err(
            Error::new(Code::Internal, "GEMM 签名需要 3 个缓冲（A/B/Y）")
                .with("engine::tiled::metal_tiled_gemm"),
        ),
    }
}

/// 三个整数标量 `(M, N, K)` 的样本值。
fn gemm_dims(signature: &Signature, sample: &Sample) -> Result<(usize, usize, usize)> {
    let scalars: Vec<&Param> = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
        .collect();
    if scalars.len() != 3 {
        return Err(
            Error::new(Code::Internal, "GEMM 签名需要 3 个整数标量（M/N/K）")
                .with("engine::tiled::metal_tiled_gemm"),
        );
    }
    Ok((
        dim(sample, scalars[0])?,
        dim(sample, scalars[1])?,
        dim(sample, scalars[2])?,
    ))
}

/// 一个整数标量的样本取值（必须存在且非负）。
fn dim(sample: &Sample, param: &Param) -> Result<usize> {
    let value = sample.scalar(&param.name);
    value
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("标量 `{}` 缺非负样本取值", param.name),
            )
        })
        .map_err(|error| error.with("engine::tiled::metal_tiled_gemm"))
}

/// 分块 GEMM 的 MSL 正文：只拼字面与已解出的维度，不含错误处理（那些留在 [`metal_tiled_gemm`]）。
fn tiled_gemm_msl(a: &str, b: &str, y: &str, n: usize, k: usize, tile: usize) -> String {
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device float* d_{a} [[buffer(0)]],\n\
         \x20    device float* d_{b} [[buffer(1)]],\n\
         \x20    device float* d_{y} [[buffer(2)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint N = {n}, K = {k}, T = {tile};\n\
         \x20    threadgroup float As[{tile}][{tile}];\n\
         \x20    threadgroup float Bs[{tile}][{tile}];\n\
         \x20    const uint i = gid.x, j = gid.y, li = lid.x, lj = lid.y;\n\
         \x20    float acc = 0.0f;\n\
         \x20    for (uint kk = 0; kk < K; kk += T) {{\n\
         \x20        As[li][lj] = d_{a}[i * K + kk + lj];\n\
         \x20        Bs[li][lj] = d_{b}[(kk + li) * N + j];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint p = 0; p < T; ++p) acc += As[li][p] * Bs[p][lj];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    d_{y}[i * N + j] = acc;\n\
         }}"
    )
}

/// 生产态 threadgroup 协作分块注意力（Metal）：一个线程组负责一个查询行。
///
/// `threads` 个线程分摊 S 个分数与 D 个输出维；行内 max 与指数和走线程组树归约，完整分数块留在
/// threadgroup 内存，避免每个查询行被单线程串行扫完整条序列。语义与
/// `O = softmax(QKᵀ/√D)·V` 逐元素相同。前置条件由调用方保证（线程数为 [32,1024] 内的 2 的幂、
/// `S·4` 字节的分数块不超过线程组内存上限、`D ≤ 64`）。
///
/// 派发几何：`grid = [S·threads]`、`block = [threads]`；每组的 `gid / threads` 是查询行。
///
/// # Errors
///
/// 签名不足 4 缓冲（q/k/v/y）或 2 整数标量（S/D）、样本缺标量、或线程数非法时返回说明。
pub fn metal_tiled_attention(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    threads: usize,
) -> Result<String> {
    if !threads.is_power_of_two() || threads < 2 {
        return Err(
            Error::new(Code::Invalid, "注意力线程数必须是 2 的幂且至少为 2")
                .with("engine::tiled::metal_tiled_attention"),
        );
    }
    let (q, k, v, y) = attention_buffers(signature)?;
    let (s, d) = attention_dims(signature, sample)?;
    Ok(tiled_attention_msl(q, k, v, y, s, d, threads))
}

/// 四条缓冲名（按签名序：Q、K、V、Y）。
fn attention_buffers(signature: &Signature) -> Result<(&str, &str, &str, &str)> {
    let buffers: Vec<&Param> = signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .collect();
    match buffers.as_slice() {
        [q, k, v, y] => Ok((
            q.name.as_str(),
            k.name.as_str(),
            v.name.as_str(),
            y.name.as_str(),
        )),
        _ => Err(
            Error::new(Code::Internal, "注意力签名需要 4 个缓冲（Q/K/V/Y）")
                .with("engine::tiled::metal_tiled_attention"),
        ),
    }
}

/// 两个整数标量 `(S, D)` 的样本值。
fn attention_dims(signature: &Signature, sample: &Sample) -> Result<(usize, usize)> {
    let scalars: Vec<&Param> = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
        .collect();
    if scalars.len() != 2 {
        return Err(
            Error::new(Code::Internal, "注意力签名需要 2 个整数标量（S/D）")
                .with("engine::tiled::metal_tiled_attention"),
        );
    }
    Ok((dim(sample, scalars[0])?, dim(sample, scalars[1])?))
}

/// 分块注意力的 MSL 正文：只拼字面与已解出的维度，不含错误处理。
fn tiled_attention_msl(
    q: &str,
    k: &str,
    v: &str,
    y: &str,
    s: usize,
    d: usize,
    threads: usize,
) -> String {
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device float* d_{q} [[buffer(0)]],\n\
         \x20    device float* d_{k} [[buffer(1)]],\n\
         \x20    device float* d_{v} [[buffer(2)]],\n\
         \x20    device float* d_{y} [[buffer(3)]],\n\
         \x20    uint gid [[thread_position_in_grid]],\n\
         \x20    uint lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint S = {s}, D = {d}, T = {threads};\n\
         \x20    threadgroup float scores[S];\n\
         \x20    threadgroup float red[T];\n\
         \x20    threadgroup float stats[2];\n\
         \x20    const uint i = gid / T, t = lid;\n\
         \x20    const float scale = sqrt((float)D);\n\
         \x20    for (uint j = t; j < S; j += T) {{\n\
         \x20        float dot = 0.0f;\n\
         \x20        for (uint p = 0; p < D; ++p) dot += d_{q}[i * D + p] * d_{k}[j * D + p];\n\
         \x20        scores[j] = dot / scale;\n\
         \x20    }}\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    float lmax = -3.402823466e38f;\n\
         \x20    for (uint j = t; j < S; j += T) lmax = max(lmax, scores[j]);\n\
         \x20    red[t] = lmax;\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    for (uint step = T / 2; step > 0; step >>= 1) {{\n\
         \x20        if (t < step) red[t] = max(red[t], red[t + step]);\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    if (t == 0) stats[0] = red[0];\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const float row_max = stats[0];\n\
         \x20    float lsum = 0.0f;\n\
         \x20    for (uint j = t; j < S; j += T) {{\n\
         \x20        float e = exp(scores[j] - row_max);\n\
         \x20        scores[j] = e;\n\
         \x20        lsum += e;\n\
         \x20    }}\n\
         \x20    red[t] = lsum;\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    for (uint step = T / 2; step > 0; step >>= 1) {{\n\
         \x20        if (t < step) red[t] += red[t + step];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    if (t == 0) stats[1] = red[0];\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const float prob_sum = stats[1];\n\
         \x20    for (uint h = t; h < D; h += T) {{\n\
         \x20        float acc = 0.0f;\n\
         \x20        for (uint j = 0; j < S; ++j) acc += scores[j] * d_{v}[j * D + h];\n\
         \x20        d_{y}[i * D + h] = acc / prob_sum;\n\
         \x20    }}\n\
         }}"
    )
}
