//! 设备侧分块 lowering：共享内存 GEMM 与（后续）分块注意力。

use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::{Param, Signature};

mod flash;
mod gemm;
mod misc;

pub use flash::metal_flash_attention;
pub use gemm::{metal_simdgroup_gemm, metal_tiled_gemm};
pub use misc::{
    metal_int8_gemm, metal_int8_gemm_program, metal_tiled_attention, metal_transpose_half,
};

// ---------------------------------------------------------------------------
// 签名 → 缓冲名与几何：各族共用
// ---------------------------------------------------------------------------

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
    let (a, b, y) = gemm_buffers(signature)?;
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
    let mma = tensor_mma(a, exact);
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
         \x20    threadgroup half sb[NRN * KP];\n\
         \x20    auto tA = tensor((device float*)d_{a} + (size_t)ra * K,\n\
         \x20                     dextents<int32_t, 2>(K, TSM), array<int, 2>{{1, K}});\n\
         \x20    auto tB = tensor(sb, dextents<int32_t, 2>(TK, NRN), array<int, 2>{{1, KP}});\n\
         \x20    using D = mpp::tensor_ops::matmul2d_descriptor;\n\
         \x20    constexpr auto MA = D::mode::multiply_accumulate;\n\
         \x20    mpp::tensor_ops::matmul2d<D(TSM, NRN, TK, false, true, true, MA),\n\
         \x20        execution_simdgroups<SG>> mm_full;\n\
         {tail_decl}\
         \x20    auto cT = mm_full.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();\n\
         \x20    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.0f;\n\
         \x20    for (int k0 = 0; k0 < K; k0 += TK) {{\n\
         \x20        for (int w = int(tiitg); w < WI; w += NT) {{\n\
         \x20            const int row = w / IPR, col = (w % IPR) * 16, kp = k0 + col;\n\
         \x20            const size_t base = (size_t)(rb + row) * K + kp;\n\
         \x20            threadgroup half* dst = sb + row * KP + col;\n\
         \x20            if (rb + row < N && kp + 15 < K && (base & 3) == 0) {{\n\
         \x20                const device half4* src = (const device half4*)(d_{b} + base);\n\
         \x20                threadgroup half4* out = (threadgroup half4*)dst;\n\
         \x20                out[0] = src[0]; out[1] = src[1]; out[2] = src[2]; out[3] = src[3];\n\
         \x20            }} else {{\n\
         \x20                for (int i = 0; i < 16; ++i)\n\
         \x20                    dst[i] = (rb + row < N && kp + i < K) ? d_{b}[base + (size_t)i] : (half)0;\n\
         \x20            }}\n\
         \x20        }}\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         {mma}\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    auto tD = tensor(d_{y} + (size_t)ra * N + rb,\n\
         \x20                     dextents<int32_t, 2>(min(NRN, N - rb), TSM), array<int, 2>{{1, N}});\n\
         \x20    cT.store(tD);\n\
         }}"
    )
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
fn tensor_mma(a: &str, exact: bool) -> String {
    if exact {
        format!(
            "        auto av = tensor((device float*)d_{a} + (size_t)ra * K + k0,\n\
             \x20                          dextents<int32_t, 2>(TK, TSM), array<int, 2>{{1, K}});\n\
             \x20        auto bv = tensor(sb, dextents<int32_t, 2>(TK, NRN), array<int, 2>{{1, KP}});\n\
             \x20        mm_full.run(av, bv, cT);\n"
        )
    } else {
        format!(
            "        if (k0 + TK <= K) {{\n\
             \x20            auto av = tensor((device float*)d_{a} + (size_t)ra * K + k0,\n\
             \x20                              dextents<int32_t, 2>(TK, TSM), array<int, 2>{{1, K}});\n\
             \x20            auto bv = tensor(sb, dextents<int32_t, 2>(TK, NRN), array<int, 2>{{1, KP}});\n\
             \x20            mm_full.run(av, bv, cT);\n\
             \x20        }} else {{\n\
             \x20            const int ke = K - k0;\n\
             \x20            auto av = tensor((device float*)d_{a} + (size_t)ra * K + k0,\n\
             \x20                              dextents<int32_t, 2>(ke, TSM), array<int, 2>{{1, K}});\n\
             \x20            auto bv = tensor(sb, dextents<int32_t, 2>(ke, NRN), array<int, 2>{{1, KP}});\n\
             \x20            mm_tail.run(av, bv, cT);\n\
             \x20        }}\n"
        )
    }
}
