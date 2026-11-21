//! 转置与其它不归属上述族的设备端形态变换。

use super::*;

/// 张量核 GEMM 的 B 布局前置：把行主序 `rows × cols` 的 f32 转成 `cols × rows` 的 half。
///
/// 张量核要的 B 是 `N×K`（规约轴连续）且按 half 存，两者都不是模板里 `K×N` f32 的形状，所以
/// 权重上传后先跑一遍这个内核，之后每次 GEMM 只读转好的缓冲。32×32 分块走线程组内存，
/// 全局读、全局写都按行连续，与标量逐元素转置相比不牺牲合并访存。
///
/// 派发几何：线程组 `[⌈cols/32⌉, ⌈rows/32⌉]`，组内 `32×8` 个线程。
///
/// # Errors
///
/// `rows` 或 `cols` 为 0 时返回说明。
pub fn metal_transpose_half(rows: usize, cols: usize) -> Result<String> {
    if rows == 0 || cols == 0 {
        return Err(Error::new(Code::Invalid, "转置要求行列均非零")
            .with("engine::tiled::metal_transpose_half"));
    }
    Ok(format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         \n\
         kernel void kernel_main(\n\
         \x20    device const float* d_src [[buffer(0)]],\n\
         \x20    device half* d_dst [[buffer(1)]],\n\
         \x20    uint3 tgp [[threadgroup_position_in_grid]],\n\
         \x20    uint3 tid [[thread_position_in_threadgroup]]) {{\n\
         \x20    constexpr int R = {rows}, C = {cols};\n\
         \x20    threadgroup float tile[32][33];\n\
         \x20    const int sx = int(tgp.x) * 32 + int(tid.x);\n\
         \x20    const int sy = int(tgp.y) * 32 + int(tid.y);\n\
         \x20    for (int j = 0; j < 32; j += 8) {{\n\
         \x20        const int y = sy + j;\n\
         \x20        tile[int(tid.y) + j][int(tid.x)] =\n\
         \x20            (y < R && sx < C) ? d_src[(size_t)y * C + sx] : 0.0f;\n\
         \x20    }}\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const int dx = int(tgp.y) * 32 + int(tid.x);\n\
         \x20    const int dy = int(tgp.x) * 32 + int(tid.y);\n\
         \x20    for (int j = 0; j < 32; j += 8) {{\n\
         \x20        const int y = dy + j;\n\
         \x20        if (y < C && dx < R)\n\
         \x20            d_dst[(size_t)y * R + dx] = half(tile[int(tid.x)][int(tid.y) + j]);\n\
         \x20    }}\n\
         }}"
    ))
}

/// 生产态 threadgroup 协作分块注意力（Metal）：一个线程组负责一个查询行。
///
/// `threads` 个线程分摊 S 个分数与 D 个输出维；行内 max 与指数和走线程组树归约，完整分数块留在
/// threadgroup 内存，避免每个查询行被单线程串行扫完整条序列。语义与
/// `O = softmax(QKᵀ/√D)·V` 逐元素相同。前置条件由调用方保证（线程数为 `32..=1024` 内的 2 的幂、
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
         \x20    threadgroup float scores[{s}];\n\
         \x20    threadgroup float red[{threads}];\n\
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

use op::sig::CType;

/// 生产态 int8 分块 GEMM（Metal）：`A[M,K]·B[K,N]`，`i8` 输入、`i32` 累加、`f32` 输出。
///
/// 签名驱动入口：缓冲与维度的取法与 f32 分块 GEMM 相同（前三条缓冲、三个整数标量），额外的
/// 校验是元素类型必须是 `u8/u8/f32`——`Lower::Int8Gemm` 只该被 int8 GEMM 走到。
///
/// # Errors
///
/// 签名不是 `(u8 a, u8 b, float y, int M, int N, int K)`、样本缺标量，或形状不满足分块整除时返回说明。
pub fn metal_int8_gemm(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    tile: usize,
) -> Result<String> {
    check_int8_signature(signature)?;
    let (m, n, k) = gemm_dims(signature, sample)?;
    metal_int8_gemm_program(m, n, k, tile)
}

/// 签名是不是 `(u8 *a, u8 *b, float *y, int M, int N, int K)`；类型不符是分派错误，不是形状错误。
fn check_int8_signature(signature: &Signature) -> Result<()> {
    int8_signature(signature)
        .then_some(())
        .ok_or_else(int8_mismatch)
}

/// 签名不符时的诊断：与判定分开，各只回答一个问题。
fn int8_mismatch() -> Error {
    Error::new(
        Code::Internal,
        "int8 GEMM 签名需要 (u8 a, u8 b, float y, int M, int N, int K)",
    )
    .with(INT8_SITE)
}

/// 逐位核对 int8 GEMM 的形参类型（与 `cutile::execution` 的同一判据保持一致）。
fn int8_signature(signature: &Signature) -> bool {
    let types: Vec<CType> = signature.params().iter().map(|param| param.ctype).collect();
    types
        == [
            CType::U8,
            CType::U8,
            CType::F32,
            CType::Int,
            CType::Int,
            CType::Int,
        ]
}

/// int8 分块 GEMM 的 MSL：`A[M,K]·B[K,N]`，`i8` 输入、`i32` 累加、`f32` 输出。
///
/// 语义与 [`op::matmul_i8`] 的参考实现逐元素相同（整数累加，结果精确）。每个线程组把
/// `tile×tile` 的 A/B 块搬进线程组内存，收缩轴在共享块上复用；要求 `M/N/K` 都能被
/// `tile` 整除。派发几何：`grid = [M, N]`、`block = [tile, tile]`。
///
/// # Errors
///
/// `tile == 0` 或任一维度不能被 `tile` 整除时返回说明。
pub fn metal_int8_gemm_program(m: usize, n: usize, k: usize, tile: usize) -> Result<String> {
    check_int8_shape(m, n, k, tile)?;
    Ok(int8_gemm_msl(n, k, tile))
}

/// int8 GEMM 的形状前置条件：分块必须整除三个维度，否则尾块会读越界。
fn check_int8_shape(m: usize, n: usize, k: usize, tile: usize) -> Result<()> {
    if tile == 0 {
        return Err(Error::new(Code::Invalid, "分块大小必须大于 0").with(INT8_SITE));
    }
    if [m, n, k].iter().all(|&dim| dim.is_multiple_of(tile)) {
        return Ok(());
    }
    Err(Error::new(Code::Invalid, "int8 GEMM 要求 M/N/K 都能被分块整除").with(INT8_SITE))
}

/// int8 GEMM 的报错出处。
const INT8_SITE: &str = "engine::tiled::metal_int8_gemm_program";

/// int8 GEMM 的 MSL 正文：共享块以 `int` 存放，避免每次乘加重复做符号扩展。
fn int8_gemm_msl(n: usize, k: usize, tile: usize) -> String {
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device const char* d_a [[buffer(0)]],\n\
         \x20    device const char* d_b [[buffer(1)]],\n\
         \x20    device float* d_y [[buffer(2)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint N = {n}, K = {k}, T = {tile};\n\
         \x20    threadgroup int As[{tile}][{tile}];\n\
         \x20    threadgroup int Bs[{tile}][{tile}];\n\
         \x20    const uint i = gid.x, j = gid.y, li = lid.x, lj = lid.y;\n\
         \x20    int acc = 0;\n\
         \x20    for (uint kk = 0; kk < K; kk += T) {{\n\
         \x20        As[li][lj] = int(d_a[i * K + kk + lj]);\n\
         \x20        Bs[li][lj] = int(d_b[(kk + li) * N + j]);\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint p = 0; p < T; ++p) acc += As[li][p] * Bs[p][lj];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    d_y[i * N + j] = float(acc);\n\
         }}"
    )
}
