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

/// 按每线程输出块大小选简单分块（`1`）或 4×4 寄存器分块（`4`）。
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

/// 生产态 flash attention（Metal）：一个线程组负责一个查询行，K/V 按 `block` 个键分块
/// 流式扫描，在线 softmax（running max/sum）把输出累加在每线程寄存器，不保留完整分数向量。
///
/// 语义与 `O = softmax(QKᵀ/√D)·V` 逐元素相同，只是把「先算完整行、再归一化」改成
/// 「逐块修正」：每块先求块内 max，与 running max 合并得到新的 max，指数与输出按
/// `exp(m_old - m_new)` 缩放后并入。内存从 `O(S)` 降到 `O(block)`，因而可流式处理长序列。
///
/// 派发几何：`grid = [S·threads]`、`block = [threads]`；每组的 `gid / threads` 是查询行。
///
/// # Errors
///
/// 签名不足 4 缓冲（q/k/v/y）或 2 整数标量（S/D）、样本缺标量、线程/块几何非法、
/// 或头宽 `D` 不在 `1..=64` 时返回说明。
pub fn metal_flash_attention(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    threads: usize,
    block: usize,
) -> Result<String> {
    if !flash_geometry_ok(threads, block) {
        return Err(Error::new(
            Code::Invalid,
            "flash attention 线程数须为 2 的幂、块为线程数倍数",
        )
        .with("engine::tiled::metal_flash_attention"));
    }
    let (q, k, v, y) = attention_buffers(signature)?;
    let (s, d) = attention_dims(signature, sample)?;
    if !(1..=64).contains(&d) {
        return Err(
            Error::new(Code::Invalid, "flash attention 要求头宽 D 在 1..=64")
                .with("engine::tiled::metal_flash_attention"),
        );
    }
    Ok(flash_attention_msl(
        q,
        k,
        v,
        y,
        Flash {
            seq: s,
            dim: d,
            threads,
            block,
        },
    ))
}

/// 线程/块几何：线程数为 2 的幂且至少 2，块非零且是线程数倍数。
fn flash_geometry_ok(threads: usize, block: usize) -> bool {
    threads.is_power_of_two() && threads >= 2 && block != 0 && block.is_multiple_of(threads)
}

/// 派发几何：序列长、头宽、线程数与键块。四个数在正文里只当常量用，收在一处以免每个
/// 生成函数都重算一遍 `div_ceil` 与“除不除得尽”。
#[derive(Clone, Copy)]
struct Flash {
    seq: usize,
    dim: usize,
    threads: usize,
    block: usize,
}

impl Flash {
    /// 每线程承担的寄存器输出维数。
    fn lanes(self) -> usize {
        self.dim.div_ceil(self.threads)
    }

    /// 键块是否除不尽序列：除不尽才需要尾块守卫。
    fn tail(self) -> bool {
        !self.seq.is_multiple_of(self.block)
    }
}

/// flash attention 的 MSL 正文：在线 softmax 的块内归约 + 寄存器输出累加。
fn flash_attention_msl(q: &str, k: &str, v: &str, y: &str, shape: Flash) -> String {
    let nd = shape.lanes();
    let tail = shape.tail();
    let (s, d) = (shape.seq.to_string(), shape.dim.to_string());
    let (threads, block) = (shape.threads.to_string(), shape.block.to_string());
    let nds = nd.to_string();
    let decls: String = (0..nd)
        .map(|n| format!("    float acc_{n} = 0.0f;\n"))
        .collect();
    let updates: String = (0..nd)
        .map(|n| flash_acc_update(v, n, shape, tail))
        .collect();
    let writes: String = (0..nd).map(|n| flash_output(y, n, shape)).collect();
    fill(
        FLASH_KERNEL,
        &[
            ("q", q),
            ("k", k),
            ("v", v),
            ("y", y),
            ("seq", &s),
            ("dim", &d),
            ("threads", &threads),
            ("block", &block),
            ("nd", &nds),
            ("score", &flash_score_body(q, k, tail)),
            ("weight", flash_weight(tail)),
            ("decls", &decls),
            ("updates", &updates),
            ("writes", &writes),
        ],
    )
}

/// 模板填空：`@name` 逐个替换成对应文本。
///
/// 缺项不会被静默放过——剩下的 `@name` 会原样进入内核，由 Metal 编译报错指向模板本身，
/// 而不是让"少填一项"变成一个能启动却算错的内核。
fn fill(template: &str, pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .fold(template.to_string(), |text, (name, value)| {
            text.replace(&format!("@{name}"), value)
        })
}

/// 一个寄存器输出维的回写：除以 `l` 后只写本线程负责的那一维。
fn flash_output(y: &str, n: usize, shape: Flash) -> String {
    if shape.dim.is_multiple_of(shape.threads) {
        return format!(
            "    {{ const uint h = t + {}; d_{y}[i * D + h] = acc_{n} / l; }}\n",
            n * shape.threads
        );
    }
    format!(
        "    {{ const uint h = t + {}; if (h < D) d_{y}[i * D + h] = acc_{n} / l; }}\n",
        n * shape.threads
    )
}

/// flash 内核模板：`@name` 占位由 [`fill`] 替换。
///
/// 正文与"传什么参数"分开，是因为这段内核没有分支——真正需要读的是占位符那几行，
/// 而把 60 余行正文和参数拼装塞进同一个函数只会一起越过函数长度上限。
const FLASH_KERNEL: &str = "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device const float* d_@q [[buffer(0)]],\n\
         \x20    device const float* d_@k [[buffer(1)]],\n\
         \x20    device const float* d_@v [[buffer(2)]],\n\
         \x20    device float* d_@y [[buffer(3)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {\n\
         \x20    const uint S = @seq, D = @dim, T = @threads, BK = @block, ND = @nd;\n\
         \x20    threadgroup float scores[BK];\n\
         \x20    threadgroup float red[T];\n\
         \x20    threadgroup float red_sum[T];\n\
         \x20    const uint i = gid.x / T, t = lid.x;\n\
         \x20    const float scale = sqrt((float)D);\n\
         \x20    float m = -3.402823466e38f, l = 0.0f;\n\
         @decls\
         \x20    for (uint kb = 0; kb < S; kb += BK) {\n\
         \x20        float tmax = -3.402823466e38f;\n\
         \x20        for (uint j = t; j < BK; j += T) {\n\
         \x20            float sc = -3.402823466e38f;\n\
         @score\
         \x20            scores[j] = sc;\n\
         \x20            tmax = max(tmax, sc);\n\
         \x20        }\n\
         \x20        red[t] = tmax;\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint st = T / 2; st > 0; st >>= 1) {\n\
         \x20            if (t < st) red[t] = max(red[t], red[t + st]);\n\
         \x20            threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        }\n\
         \x20        const float mn = max(m, red[0]);\n\
         \x20        const float alpha = exp(m - mn);\n\
         \x20        float tsum = 0.0f;\n\
         \x20        for (uint j = t; j < BK; j += T) {\n\
         \x20            float e = @weight;\n\
         \x20            scores[j] = e;\n\
         \x20            tsum += e;\n\
         \x20        }\n\
         \x20        red_sum[t] = tsum;\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint st = T / 2; st > 0; st >>= 1) {\n\
         \x20            if (t < st) red_sum[t] += red_sum[t + st];\n\
         \x20            threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        }\n\
         \x20        l = l * alpha + red_sum[0];\n\
         @updates\
         \x20        m = mn;\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }\n\
         @writes\
         }";

/// 分数块的单键计算体：`S` 不是块大小整数倍时带尾块守卫。
fn flash_score_body(q: &str, k: &str, tail: bool) -> String {
    let dot = format!(
        "float dot = 0.0f;\n\
         \x20               for (uint p = 0; p < D; ++p) dot += d_{q}[i * D + p] * d_{k}[(kb + j) * D + p];\n\
         \x20               sc = dot / scale;\n"
    );
    if tail {
        format!(
            "            if (kb + j < S) {{\n\
             \x20               {dot}\
             \x20           }}\n"
        )
    } else {
        format!("            {dot}")
    }
}

/// 指数权重：无尾块守卫时省掉逐键分支。
fn flash_weight(tail: bool) -> &'static str {
    if tail {
        "(kb + j < S) ? exp(scores[j] - mn) : 0.0f"
    } else {
        "exp(scores[j] - mn)"
    }
}

/// 一个寄存器输出维 `acc_n` 的在线更新：先按 `alpha` 缩放旧累加，再加当前键块的 `P·V`。
fn flash_acc_update(v: &str, n: usize, shape: Flash, tail: bool) -> String {
    let h = n * shape.threads;
    let add = format!("a += scores[j] * d_{v}[(kb + j) * D + h];");
    let value = if tail {
        format!("if (kb + j < S) {add}")
    } else {
        add
    };
    if shape.dim.is_multiple_of(shape.threads) {
        return format!(
            "        {{ const uint h = t + {h};\n\
             \x20           float a = acc_{n} * alpha;\n\
             \x20           for (uint j = 0; j < BK; ++j) {{ {value} }}\n\
             \x20           acc_{n} = a; }}\n"
        );
    }
    format!(
        "        {{ const uint h = t + {h};\n\
         \x20           if (h < D) {{\n\
         \x20               float a = acc_{n} * alpha;\n\
         \x20               for (uint j = 0; j < BK; ++j) {{ {value} }}\n\
         \x20               acc_{n} = a;\n\
         \x20           }} }}\n"
    )
}
