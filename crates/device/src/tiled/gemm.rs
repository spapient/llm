//! 分块 GEMM：共享内存分块、simdgroup 矩阵乘，以及它们的 MSL 骨架。

use super::*;

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
    per_thread: usize,
) -> Result<String> {
    if tile == 0 || per_thread == 0 {
        return Err(
            Error::new(Code::Invalid, "分块大小必须大于 0").with("engine::tiled::metal_tiled_gemm")
        );
    }
    let (a, b, y) = gemm_buffers(signature)?;
    let (_, n, k) = gemm_dims(signature, sample)?;
    Ok(tiled_gemm_msl(a, b, y, n, k, tile, per_thread))
}

/// 三条缓冲名（按签名序：A、B、Y）。

fn tiled_gemm_msl(
    a: &str,
    b: &str,
    y: &str,
    n: usize,
    k: usize,
    tile: usize,
    per_thread: usize,
) -> String {
    if per_thread == 1 {
        return simple_tiled_gemm_msl(a, b, y, n, k, tile);
    }
    register_blocked_gemm_msl(a, b, y, n, k, tile, per_thread)
}

/// 生产态 simdgroup 矩阵乘 GEMM（Metal）：每个 simdgroup 用 `8×8×8` 矩阵乘指令算
/// `row_frags × col_frags` 个输出片段。
///
/// 语义与模板逐元素相同（`Y[i][j] = Σ_p A[i][p]·B[p][j]`）。前置条件由调用方保证
/// （`K % 8 == 0`、`M` 被 `8·row_frags` 整除、`N` 被 `8·col_frags` 整除）。
///
/// 派发几何：`grid = [(M/(8·row_frags))·32, N/(8·col_frags)]`、`block = [32]`；每个
/// 线程组是一个 simdgroup，`gid.x / 32` 是输出行块、`gid.y` 是输出列块。
///
/// # Errors
///
/// 签名不足三条缓冲或三个整数标量、样本缺标量、片段数不在 `1..=4`、或 `K` 不是 8 的倍数时返回说明。
pub fn metal_simdgroup_gemm(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    row_frags: usize,
    col_frags: usize,
) -> Result<String> {
    check_fragments(row_frags, col_frags)?;
    let (a, b, y) = gemm_buffers(signature)?;
    let (_, n, k) = gemm_dims(signature, sample)?;
    check_k(k)?;
    Ok(simdgroup_gemm_msl(a, b, y, n, k, row_frags, col_frags))
}

/// simdgroup 路径的报错出处。
const SIMD_SITE: &str = "engine::tiled::metal_simdgroup_gemm";

/// 片段数闭集 `1..=4`：一片是 8×8，超过四片就把线程组预算吃光。
fn check_fragments(row_frags: usize, col_frags: usize) -> Result<()> {
    if (1..=4).contains(&row_frags) && (1..=4).contains(&col_frags) {
        return Ok(());
    }
    Err(Error::new(Code::Invalid, "simdgroup 片段数必须在 1..=4").with(SIMD_SITE))
}

/// `simdgroup_load(.., 8)` 隐含 8×8 片段：K 不是 8 的正整数倍时尾片段会读越界。
fn check_k(k: usize) -> Result<()> {
    if k != 0 && k.is_multiple_of(8) {
        return Ok(());
    }
    Err(Error::new(Code::Invalid, "simdgroup GEMM 要求 K 是 8 的正整数倍").with(SIMD_SITE))
}

/// 逐对展开 `row_frags × col_frags`：声明、乘加、写回三段共用同一遍历次序。
///
/// 收成一处是因为"哪些片段存在"只能有一个答案——三段各写一遍循环，改片段数就得改三处，
/// 而少改一处生成的内核要么漏写一个片段、要么读一个未声明的 `acc_r_c`。
fn each_pair(row_frags: usize, col_frags: usize, mut emit: impl FnMut(usize, usize)) {
    for r in 0..row_frags {
        for c in 0..col_frags {
            emit(r, c);
        }
    }
}

/// simdgroup GEMM 的 MSL 正文：片段数组按下标展开成具名寄存器，避免动态下标把矩阵落到线程栈。
fn simdgroup_gemm_msl(
    a: &str,
    b: &str,
    y: &str,
    n: usize,
    k: usize,
    row_frags: usize,
    col_frags: usize,
) -> String {
    let mut src = format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device const float* d_{a} [[buffer(0)]],\n\
         \x20    device const float* d_{b} [[buffer(1)]],\n\
         \x20    device float* d_{y} [[buffer(2)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint N = {n}, K = {k}, R = {row_frags}, C = {col_frags};\n\
         \x20    threadgroup float zeros[64];\n\
         \x20    for (uint i = lid.x; i < 64; i += 32) zeros[i] = 0.0f;\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const uint row0 = (gid.x / 32) * (8 * R), col0 = gid.y * (8 * C);\n"
    );
    each_pair(row_frags, col_frags, |r, c| {
        src.push_str(&format!(
            "    simdgroup_float8x8 acc_{r}_{c};\n\
             \x20   simdgroup_load(acc_{r}_{c}, zeros, 8);\n"
        ));
    });
    src.push_str("    for (uint kk = 0; kk < K; kk += 8) {\n");
    for r in 0..row_frags {
        src.push_str(&format!(
            "        simdgroup_float8x8 fa_{r};\n\
             \x20       simdgroup_load(fa_{r}, d_{a} + (row0 + {}) * K + kk, K);\n",
            r * 8,
        ));
    }
    for c in 0..col_frags {
        src.push_str(&format!(
            "        simdgroup_float8x8 fb_{c};\n\
             \x20       simdgroup_load(fb_{c}, d_{b} + kk * N + col0 + {}, N);\n",
            c * 8,
        ));
    }
    each_pair(row_frags, col_frags, |r, c| {
        src.push_str(&format!(
            "        simdgroup_multiply_accumulate(acc_{r}_{c}, fa_{r}, fb_{c}, acc_{r}_{c});\n"
        ));
    });
    src.push_str("    }\n");
    each_pair(row_frags, col_frags, |r, c| {
        src.push_str(&format!(
            "    simdgroup_store(acc_{r}_{c}, d_{y} + (row0 + {}) * N + col0 + {}, N);\n",
            r * 8,
            c * 8,
        ));
    });
    src.push('}');
    src
}

/// 分块 GEMM 各形态共用的 MSL 前导：三路设备缓冲接口 + 网格/线程组下标；常量行由调用方续写。
fn tiled_gemm_head(a: &str, b: &str, y: &str) -> String {
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device float* d_{a} [[buffer(0)]],\n\
         \x20    device float* d_{b} [[buffer(1)]],\n\
         \x20    device float* d_{y} [[buffer(2)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {{\n"
    )
}

/// 简单分块 GEMM（每线程一输出）的 MSL 正文。
fn simple_tiled_gemm_msl(a: &str, b: &str, y: &str, n: usize, k: usize, tile: usize) -> String {
    let mut src = tiled_gemm_head(a, b, y);
    src.push_str(&format!(
        "\x20    const uint N = {n}, K = {k}, T = {tile};\n\
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
    ));
    src
}

/// 4×4 寄存器分块 GEMM：每线程算 `per_thread × per_thread` 个输出，共享块按 `tile` 一次搬运。
///
/// 派发几何：`grid = [M/per_thread, N/per_thread]`、`block = [tile/per_thread, tile/per_thread]`；
/// 组内 `tid = lid.x·TT + lid.y` 协同搬运 `tile×tile` 的 A/B 块。要求 `M/N/K` 均被 `tile` 整除。
fn register_blocked_gemm_msl(
    a: &str,
    b: &str,
    y: &str,
    n: usize,
    k: usize,
    tile: usize,
    per_thread: usize,
) -> String {
    let threads = tile / per_thread;
    let mut src = tiled_gemm_head(a, b, y);
    src.push_str(&format!(
        "\x20    const uint N = {n}, K = {k}, T = {tile}, PT = {per_thread}, TT = {threads};\n\
         \x20    threadgroup float As[{tile}][{tile}];\n\
         \x20    threadgroup float Bs[{tile}][{tile}];\n\
         \x20    const uint gr = (gid.x - lid.x) * PT, gc = (gid.y - lid.y) * PT;\n\
         \x20    const uint r0 = gid.x * PT, c0 = gid.y * PT;\n\
         \x20    float acc[{per_thread}][{per_thread}];\n\
         \x20    for (uint ii = 0; ii < PT; ++ii)\n\
         \x20        for (uint jj = 0; jj < PT; ++jj) acc[ii][jj] = 0.0f;\n\
         \x20    const uint tid = lid.x * TT + lid.y;\n\
         \x20    for (uint kk = 0; kk < K; kk += T) {{\n\
         \x20        for (uint idx = tid; idx < T * T; idx += TT * TT) {{\n\
         \x20            uint rr = idx / T, cc = idx % T;\n\
         \x20            As[rr][cc] = d_{a}[(gr + rr) * K + kk + cc];\n\
         \x20            Bs[rr][cc] = d_{b}[(kk + rr) * N + gc + cc];\n\
         \x20        }}\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint p = 0; p < T; ++p) {{\n\
         \x20            for (uint ii = 0; ii < PT; ++ii) {{\n\
         \x20                float av = As[lid.x * PT + ii][p];\n\
         \x20                for (uint jj = 0; jj < PT; ++jj) acc[ii][jj] += av * Bs[p][lid.y * PT + jj];\n\
         \x20            }}\n\
         \x20        }}\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    for (uint ii = 0; ii < PT; ++ii)\n\
         \x20        for (uint jj = 0; jj < PT; ++jj) d_{y}[(r0 + ii) * N + c0 + jj] = acc[ii][jj];\n\
         }}"
    ));
    src
}
