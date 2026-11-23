//! 标量 / simdgroup / 寄存器 / shared 分块 GEMM，以及 int8 口径的同一族实现。

use super::geometry::{gemm_bufs, gemm_dims};
use diagnostic::{Code, Error, Result};
use prof::model::Sample;
use prof::sig::{CType, Signature};

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
pub(crate) fn metal_tiled_gemm(
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
    let (a, b, y) = gemm_bufs(signature)?;
    let (_, n, k) = gemm_dims(signature, sample)?;
    Ok(tiled_gemm_msl(a, b, y, n, k, tile, per_thread))
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
        return shared_tile_gemm_msl(a, b, y, n, k, tile, TileMath::F32);
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
pub(crate) fn metal_simdgroup_gemm(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    row_frags: usize,
    col_frags: usize,
) -> Result<String> {
    check_fragments(row_frags, col_frags)?;
    let (a, b, y) = gemm_bufs(signature)?;
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
    let mut src = tiled_gemm_head(a, b, y, "const float");
    src.push_str(&format!(
        "\x20    const uint N = {n}, K = {k}, R = {row_frags}, C = {col_frags};\n\
         \x20    threadgroup float zeros[64];\n\
         \x20    for (uint i = lid.x; i < 64; i += 32) zeros[i] = 0.0f;\n\
         \x20    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    const uint row0 = (gid.x / 32) * (8 * R), col0 = gid.y * (8 * C);\n"
    ));
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

/// 分块 GEMM 各形态共用的 MSL 前导：三路设备缓冲接口（`elem` 是 A/B 的元素类型）+ 网格/线程组
/// 下标；常量行由调用方续写。
fn tiled_gemm_head(a: &str, b: &str, y: &str, elem: &str) -> String {
    format!(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\
         kernel void kernel_main(\n\
         \x20    device {elem}* d_{a} [[buffer(0)]],\n\
         \x20    device {elem}* d_{b} [[buffer(1)]],\n\
         \x20    device float* d_{y} [[buffer(2)]],\n\
         \x20    uint3 gid [[thread_position_in_grid]],\n\
         \x20    uint3 lid [[thread_position_in_threadgroup]]) {{\n"
    )
}

/// 共享块 GEMM 的数值口径：`f32` 直接乘加；`int8` 在 `int` 共享块里累加、写回时收窄成 `f32`。
/// 两种口径的搬运、屏障与收缩循环逐字相同，只有「缓冲/共享块类型、读入提升、写回收窄」不同。
#[derive(Clone, Copy)]
enum TileMath {
    F32,
    Int8,
}

impl TileMath {
    /// A/B 缓冲的 C 类型与共享块（等价于累加器）的 C 类型。
    fn types(self) -> (&'static str, &'static str) {
        match self {
            Self::F32 => ("float", "float"),
            Self::Int8 => ("const char", "int"),
        }
    }

    /// 读入一项：`int8` 先提升到 `int`，避免每次乘加重复做符号扩展。
    fn load(self, expr: &str) -> String {
        match self {
            Self::F32 => expr.to_owned(),
            Self::Int8 => format!("int({expr})"),
        }
    }

    /// 累加器声明行。
    fn accumulator(self) -> &'static str {
        match self {
            Self::F32 => "    float acc = 0.0f;\n",
            Self::Int8 => "    int acc = 0;\n",
        }
    }

    /// 写回表达式的叶子。
    fn result(self) -> &'static str {
        match self {
            Self::F32 => "acc",
            Self::Int8 => "float(acc)",
        }
    }
}

/// 简单分块 GEMM（每线程一输出）的 MSL 正文：`tile×tile` 的 A/B 块协同搬进线程组内存，收缩轴在
/// 共享块上复用。f32 与 int8 共用这一份骨架，差异全在 [`TileMath`]。
fn shared_tile_gemm_msl(
    a: &str,
    b: &str,
    y: &str,
    n: usize,
    k: usize,
    tile: usize,
    math: TileMath,
) -> String {
    let (elem, block) = math.types();
    let mut src = tiled_gemm_head(a, b, y, elem);
    src.push_str(&format!(
        "\x20    const uint N = {n}, K = {k}, T = {tile};\n\
         \x20    threadgroup {block} As[{tile}][{tile}];\n\
         \x20    threadgroup {block} Bs[{tile}][{tile}];\n\
         \x20    const uint i = gid.x, j = gid.y, li = lid.x, lj = lid.y;\n\
         {acc}\
         \x20    for (uint kk = 0; kk < K; kk += T) {{\n\
         \x20        As[li][lj] = {a_load};\n\
         \x20        Bs[li][lj] = {b_load};\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint p = 0; p < T; ++p) acc += As[li][p] * Bs[p][lj];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    d_{y}[i * N + j] = {result};\n\
         }}",
        acc = math.accumulator(),
        a_load = math.load(&format!("d_{a}[i * K + kk + lj]")),
        b_load = math.load(&format!("d_{b}[(kk + li) * N + j]")),
        result = math.result(),
    ));
    src
}

/// 4×4 寄存器分块 GEMM：每线程算 `per_thread × per_thread` 个输出，共享块按 `tile` 一次搬运。
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
    let mut src = tiled_gemm_head(a, b, y, "float");
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

/// 生产态 int8 分块 GEMM（Metal）：`A[M,K]·B[K,N]`，`i8` 输入、`i32` 累加、`f32` 输出。
///
/// 签名驱动入口：缓冲与维度的取法与 f32 分块 GEMM 相同（前三条缓冲、三个整数标量），额外的
/// 校验是元素类型必须是 `u8/u8/f32`——`Lower::Int8Gemm` 只该被 int8 GEMM 走到。
///
/// # Errors
///
/// 签名不是 `(u8 a, u8 b, float y, int M, int N, int K)`、样本缺标量，或形状不满足分块整除时返回说明。
pub(crate) fn metal_int8_gemm(
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

/// 逐位核对 int8 GEMM 的形参类型（与 `llm_optim::execution` 的同一判据保持一致）。
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
/// 语义与 [`nn::matmul_i8`] 的参考实现逐元素相同（整数累加，结果精确）。每个线程组把
/// `tile×tile` 的 A/B 块搬进线程组内存，收缩轴在共享块上复用；要求 `M/N/K` 都能被
/// `tile` 整除。派发几何：`grid = [M, N]`、`block = [tile, tile]`。
///
/// # Errors
///
/// `tile == 0` 或任一维度不能被 `tile` 整除时返回说明。
pub(crate) fn metal_int8_gemm_program(m: usize, n: usize, k: usize, tile: usize) -> Result<String> {
    check_int8_shape(m, n, k, tile)?;
    Ok(shared_tile_gemm_msl(
        "a",
        "b",
        "y",
        n,
        k,
        tile,
        TileMath::Int8,
    ))
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
