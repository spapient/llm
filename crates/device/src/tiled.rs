//! 设备侧分块 lowering：共享内存 GEMM 与（后续）分块注意力。

use diagnostic::{Code, Error, Result};
use nn::model::Sample;
use nn::sig::{CType, Param, Signature};

// ---------------------------------------------------------------------------
// 签名 → 缓冲名与几何：各族共用
// ---------------------------------------------------------------------------

/// 签名里全部缓冲参数（按序）。
fn buf_params(signature: &Signature) -> Vec<&Param> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .collect()
}

/// 签名里全部标量参数（按序）。
fn scalar_params(signature: &Signature) -> Vec<&Param> {
    signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buf())
        .collect()
}

/// 三条缓冲名（按签名序：A、B、Y）。
fn gemm_bufs(signature: &Signature) -> Result<(&str, &str, &str)> {
    match buf_params(signature).as_slice() {
        [a, b, y] => Ok((a.name.as_str(), b.name.as_str(), y.name.as_str())),
        _ => Err(
            Error::new(Code::Internal, "GEMM 签名需要 3 个缓冲（A/B/Y）")
                .with("engine::tiled::metal_tiled_gemm"),
        ),
    }
}

/// 三个整数标量 `(M, N, K)` 的样本值。
fn gemm_dims(signature: &Signature, sample: &Sample) -> Result<(usize, usize, usize)> {
    match scalar_params(signature).as_slice() {
        [m, n, k] => Ok((dim(sample, m)?, dim(sample, n)?, dim(sample, k)?)),
        _ => Err(
            Error::new(Code::Internal, "GEMM 签名需要 3 个整数标量（M/N/K）")
                .with("engine::tiled::metal_tiled_gemm"),
        ),
    }
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

/// 四条缓冲名（按签名序：Q、K、V、Y）。
fn attention_bufs(signature: &Signature) -> Result<(&str, &str, &str, &str)> {
    match buf_params(signature).as_slice() {
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
    match scalar_params(signature).as_slice() {
        [s, d] => Ok((dim(sample, s)?, dim(sample, d)?)),
        _ => Err(
            Error::new(Code::Internal, "注意力签名需要 2 个整数标量（S/D）")
                .with("engine::tiled::metal_tiled_attention"),
        ),
    }
}

#[cfg(test)]
#[path = "../tests/tiled.rs"]
mod tests;

use llm_optim::backend::TensorGeometry;

/// 张量核 GEMM 的几何选型：按输出规模在两组实测配置间取优，并让规约块尽量整除 `K`。
#[must_use]
pub fn tensor_geometry(m: usize, n: usize, k: usize) -> TensorGeometry {
    let small = m * n <= 4 * 1024 * 1024;
    // 行跨度必须整除 `M`（`matmul2d` 整块读 A），小 `M` 时逐档降到能整除的最大跨度；
    // 一个都不整除说明这个形状走不了张量核，留给调用方回落，这里仍给 128 让校验报错。
    let tile_m = [128usize, 64, 32]
        .into_iter()
        .find(|t| m.is_multiple_of(*t))
        .unwrap_or(128);
    let (tile_n, simdgroups) = if small { (64, 4) } else { (128, 8) };
    let fits = |t: usize| tile_n * (t + 4) * 2 <= 32 * 1024;
    let k_tile = if small {
        [128usize, 96, 64]
            .into_iter()
            .find(|t| fits(*t) && k.is_multiple_of(*t))
    } else {
        None
    }
    .or_else(|| [96usize, 64].into_iter().find(|t| fits(*t)))
    .unwrap_or(64);
    TensorGeometry {
        tile_m,
        tile_n,
        k_tile,
        simdgroups,
    }
}

/// 生产态张量核心 GEMM（Metal 4 的 `mpp::tensor_ops::matmul2d`）：一个 `tile_m × tile_n`
/// 输出块由 `32·simdgroups` 个线程协同算完，规约轴按 `k_tile` 切片。
///
/// 语义与模板逐元素相同（`Y[i][j] = Σ_p A[i][p]·B[p][j]`），但乘加走 GPU 的矩阵乘单元而不是
/// 标量 FMA。前置条件由调用方保证：`M % tile_m == 0`；B 必须是**布局前置过的** `N×K` half
/// （见 [`metal_transpose_half`]），A 仍是 `M×K` 的 f32 行主序。
///
/// 输出块取矩形而非方阵：线程组内存只被 N 方向跨度占用（B 片是 `tile_n × k_tile`），把 M 方向
/// 压小就换来同尺寸共享内存下两倍的线程组数，尾部波次更满、延迟隐藏更好——这也是线程组资源
/// 不变还能提高占用率的关键。
///
/// 派发几何：线程组按 `[N/tile_n, M/tile_m]` 铺开，组内 `32·SG` 个线程；`tgp.x` 是输出列块、
/// `tgp.y` 是输出行块。
///
/// # Errors
///
/// 签名不足三条缓冲或三个整数标量、样本缺标量、或几何参数越界时返回说明。
pub fn metal_tensor_gemm(
    _kernel: &str,
    signature: &Signature,
    sample: &Sample,
    geometry: TensorGeometry,
) -> Result<String> {
    let (a, b, y) = gemm_bufs(signature)?;
    let (m, n, k) = gemm_dims(signature, sample)?;
    check_tensor_geometry(m, n, k, geometry)?;
    Ok(tensor_gemm_msl(a, b, y, m, n, k, geometry))
}

/// 张量核路径的报错出处。
const TENSOR_SITE: &str = "engine::tiled::metal_tensor_gemm";

/// 张量核 GEMM 的几何前提：输出块的行/列跨度、规约块尺寸、simdgroup 数，以及 M 维必须整块对齐。
fn check_tensor_geometry(m: usize, n: usize, k: usize, geometry: TensorGeometry) -> Result<()> {
    check_geometry(geometry)?;
    check_tensor_shape(m, n, k, geometry)?;
    Ok(())
}

/// 张量核几何自身的三档前置：跨度档位、规约块/simdgroup、线程组内存占用。
fn check_geometry(geometry: TensorGeometry) -> Result<()> {
    let bad = |why: &str| Err(Error::new(Code::Invalid, why).with(TENSOR_SITE));
    if !geometry.spans_ok() {
        return bad("张量核输出块的行/列跨度都必须是 32、64 或 128");
    }
    if !geometry.block_ok() {
        return bad("张量核规约块必须是 ≥32 的 16 的倍数，simdgroup 数必须是 4 或 8");
    }
    if !geometry.fits_threadgroup_memory() {
        return bad("张量核 B 片超出 32 KiB 线程组内存");
    }
    Ok(())
}

/// 张量核形状前提：M 是输出块行跨度的正整数倍，且 N/K 非零。
fn check_tensor_shape(m: usize, n: usize, k: usize, geometry: TensorGeometry) -> Result<()> {
    if m == 0 || n == 0 || k == 0 || !m.is_multiple_of(geometry.tile_m) {
        return Err(Error::new(
            Code::Invalid,
            "张量核要求 M 是输出块行跨度的正整数倍，且 N/K 非零",
        )
        .with(TENSOR_SITE));
    }
    Ok(())
}

/// 张量核 GEMM 的 MSL 正文：`M/N/K` 与几何全部烘成编译期常量，故无标量缓冲绑定。
///
/// 当两片 B 片（双缓冲）也装得进 32 KiB 线程组内存时走软件流水：预热第 0 片后每轮
/// 预取下一片与当前片乘加重叠，barrier 由每轮两次减为一次，B 装载延迟不再裸露在
/// 关键路径上；装不下则退回单片主循环。
fn tensor_gemm_msl(
    a: &str,
    b: &str,
    y: &str,
    m: usize,
    n: usize,
    k: usize,
    geometry: TensorGeometry,
) -> String {
    let TensorGeometry {
        tile_m,
        tile_n,
        k_tile,
        simdgroups,
    } = geometry;
    // 规约轴被 `k_tile` 整除时没有尾块可走：直接生成单描述符、无分支的主循环，
    // 省掉逐轮的边界判断与动态 extent 版本（小形状上这段开销不可忽略）。
    let exact = geometry.exact(k);
    let tail_decl = tensor_tail_decl(exact);
    // 双缓冲要两片 B 都塞进线程组内存，塞不下就退回单片。
    let double = 2 * tile_n * (k_tile + 4) * 2 <= 32 * 1024;
    let (sb, tb, body) = if double {
        (
            "    threadgroup half sb[2][NRN * KP];\n",
            "sb[0]",
            tensor_double_loop(a, b, exact),
        )
    } else {
        (
            "    threadgroup half sb[NRN * KP];\n",
            "sb",
            tensor_single_loop(a, b, exact),
        )
    };
    format!(
        "#include <metal_stdlib>\n\
         #include <metal_tensor>\n\
         #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n\
         using namespace metal;\n\
         \n\
         kernel void kernel_main(\n\
         \x20    device const float* d_{a} [[buffer(0)]],\n\
         \x20    device const half* d_{b} [[buffer(1)]],\n\
         \x20    device float* d_{y} [[buffer(2)]],\n\
         \x20    uint3 tgp [[threadgroup_position_in_grid]],\n\
         \x20    uint tiitg [[thread_index_in_threadgroup]]) {{\n\
         \x20    constexpr int M = {m}, N = {n}, K = {k};\n\
         \x20    constexpr int TSM = {tile_m}, TSN = {tile_n}, TK = {k_tile}, SG = {simdgroups};\n\
         \x20    constexpr int NRN = TSN, KP = TK + 4, NT = 32 * SG;\n\
         \x20    constexpr int IPR = TK / 16, WI = NRN * IPR;\n\
         \x20    const int ra = int(tgp.y) * TSM, rb = int(tgp.x) * NRN;\n\
         {sb}\
         \x20    auto tA = tensor((device float*)d_{a} + (size_t)ra * K,\n\
         \x20                     dextents<int32_t, 2>(K, TSM), array<int, 2>{{1, K}});\n\
         \x20    auto tB = tensor({tb}, dextents<int32_t, 2>(TK, NRN), array<int, 2>{{1, KP}});\n\
         \x20    using D = mpp::tensor_ops::matmul2d_descriptor;\n\
         \x20    constexpr auto MA = D::mode::multiply_accumulate;\n\
         \x20    mpp::tensor_ops::matmul2d<D(TSM, NRN, TK, false, true, true, MA),\n\
         \x20        execution_simdgroups<SG>> mm_full;\n\
         {tail_decl}\
         \x20    auto cT = mm_full.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n\
         \x20    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;\n\
         {body}\
         \x20    auto tD = tensor(d_{y} + (size_t)ra * N + rb,\n\
         \x20                     dextents<int32_t, 2>(min(NRN, N - rb), TSM), array<int, 2>{{1, N}});\n\
         \x20    cT.store(tD);\n\
         }}"
    )
}

/// B 片协同装载语句：`dst` 是线程组缓冲基址表达式（单片 `sb`、双片 `sb[buf]` 或 `sb[1 - buf]`），
/// `koff` 是规约轴起点表达式（`0`、`k0` 或 `k0 + TK`），`base` 是首行缩进空格数。
/// half4 快速路径要求整 16 元素在界内且目标 8 字节对齐，否则逐元素补齐（越界补 0）。
fn tensor_load(b: &str, dst: &str, koff: &str, base: usize) -> String {
    let i0 = " ".repeat(base);
    let i1 = " ".repeat(base + 4);
    let i2 = " ".repeat(base + 8);
    let i3 = " ".repeat(base + 12);
    [
        format!("{i0}for (int w = int(tiitg); w < WI; w += NT) {{"),
        format!("{i1}const int row = w / IPR, col = (w % IPR) * 16, kp = {koff} + col;"),
        format!("{i1}const size_t base = (size_t)(rb + row) * K + kp;"),
        format!("{i1}threadgroup half* dst = {dst} + row * KP + col;"),
        format!("{i1}if (rb + row < N && kp + 15 < K && (base & 3) == 0) {{"),
        format!("{i2}const device half4* src = (const device half4*)(d_{b} + base);"),
        format!("{i2}threadgroup half4* out = (threadgroup half4*)dst;"),
        format!("{i2}out[0] = src[0]; out[1] = src[1]; out[2] = src[2]; out[3] = src[3];"),
        format!("{i1}}} else {{"),
        format!("{i2}for (int i = 0; i < 16; ++i)"),
        format!("{i3}dst[i] = (rb + row < N && kp + i < K) ? d_{b}[base + (size_t)i] : (half)0;"),
        format!("{i1}}}"),
        format!("{i0}}}"),
    ]
    .join("\n")
}

/// 单缓冲主循环：每轮 装载→barrier→乘加→barrier，B 装载延迟完全暴露在关键路径上。
fn tensor_single_loop(a: &str, b: &str, exact: bool) -> String {
    [
        "    for (int k0 = 0; k0 < K; k0 += TK) {".to_string(),
        tensor_load(b, "sb", "k0", 8),
        "        threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
        tensor_mma(a, exact, "sb"),
        "        threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
        "    }".to_string(),
    ]
    .join("\n")
        + "\n"
}

/// 双缓冲主循环：预热第 0 片后每轮预取下一片与当前片乘加重叠，barrier 由两次减为一次。
fn tensor_double_loop(a: &str, b: &str, exact: bool) -> String {
    [
        "    // 软件流水：预热第 0 片，此后预取下一片与当前片乘加重叠，双片轮流缓冲。".to_string(),
        tensor_load(b, "sb[0]", "0", 4),
        "    threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
        "    for (int k0 = 0, buf = 0; k0 < K; k0 += TK, buf ^= 1) {".to_string(),
        "        if (k0 + TK < K) {".to_string(),
        tensor_load(b, "sb[1 - buf]", "k0 + TK", 12),
        "        }".to_string(),
        tensor_mma(a, exact, "sb[buf]"),
        "        threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
        "    }".to_string(),
    ]
    .join("\n")
        + "\n"
}

/// 尾块声明：规约轴不被 `k_tile` 整除时，额外声明一个 `dynamic_extent` 的 matmul2d 描述符。
fn tensor_tail_decl(exact: bool) -> String {
    if exact {
        String::new()
    } else {
        "    mpp::tensor_ops::matmul2d<D(TSM, NRN, static_cast<int>(dynamic_extent), false, true, true, MA),\n        execution_simdgroups<SG>> mm_tail;\n".to_string()
    }
}

/// 主循环体里的矩阵乘语句：整除时走满块描述符，否则按 `k0 + TK <= K` 分流尾块。
/// `bbase` 是 B 片线程组缓冲基址表达式（单片 `sb`、双片 `sb[buf]`）。
fn tensor_mma(a: &str, exact: bool, bbase: &str) -> String {
    if exact {
        [
            format!("        auto av = tensor((device float*)d_{a} + (size_t)ra * K + k0, dextents<int32_t, 2>(TK, TSM), array<int, 2>{{1, K}});"),
            format!("        auto bv = tensor({bbase}, dextents<int32_t, 2>(TK, NRN), array<int, 2>{{1, KP}});"),
            "        mm_full.run(av, bv, cT);".to_string(),
        ]
        .join("\n")
    } else {
        [
            "        if (k0 + TK <= K) {".to_string(),
            format!("            auto av = tensor((device float*)d_{a} + (size_t)ra * K + k0, dextents<int32_t, 2>(TK, TSM), array<int, 2>{{1, K}});"),
            format!("            auto bv = tensor({bbase}, dextents<int32_t, 2>(TK, NRN), array<int, 2>{{1, KP}});"),
            "            mm_full.run(av, bv, cT);".to_string(),
            "        } else {".to_string(),
            "            const int ke = K - k0;".to_string(),
            format!("            auto av = tensor((device float*)d_{a} + (size_t)ra * K + k0, dextents<int32_t, 2>(ke, TSM), array<int, 2>{{1, K}});"),
            format!("            auto bv = tensor({bbase}, dextents<int32_t, 2>(ke, NRN), array<int, 2>{{1, KP}});"),
            "            mm_tail.run(av, bv, cT);".to_string(),
            "        }".to_string(),
        ]
        .join("\n")
    }
}


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
    let (q, k, v, y) = attention_bufs(signature)?;
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
    let (q, k, v, y) = attention_bufs(signature)?;
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
