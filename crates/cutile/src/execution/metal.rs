//! 职责：Metal 程序生成——把算子流水线降成各骨架的 MSL 内核源（网格 / 分块 GEMM / 注意力 / 归约 / 扫描 / softmax）。

use crate::backend::{Lower, TensorGeometry};
use crate::builtin::Catalogue;
use crate::decl::Skeleton;
use crate::dirs::Dirs;
use crate::pipeline::Run;
use crate::prof::Emit;
use op::sig::CType;

use super::{chunk_size, lower, lower_sources, pair, resolve, scan_chunked, single, triple};

/// 生产态 Metal 内核源：把一条算子走完整条流水线（见证建立证据 → 协商 → prepare → 骨架 lowering），
/// 返回可交给 `engine::metal_executor::MetalExecutor::register` 进程内编译的 MSL。
///
/// 与 [`execute`] 的分工：`execute` 在 C 宿主车道编译执行并当场对拍；本函数只产出生产态的 Metal
/// 内核源，执行与对拍发生在调用方（L7 组合根把这段 MSL 注入设备后端）。因此本函数不做对拍，
/// 只保证「证据存在 → 协商成立 → 拼接出合法 MSL」。
///
/// 骨架分派是「串行兜底 + 网格快路径」：归约、带归约/多标量逐元素、未声明 `[grid]` 的 Gemm、
/// 注意力与扫描统一走单线程串行（正确性兜底）；纯逐点单标量走逐元素网格、声明 `[grid]` 的 Gemm
/// 走行网格（并行快路径，属 M3-5 逐步铺开的性能面）。因此本函数对**全部骨架**都有定义。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_grid_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    lower(&run, &prepared, m, n)
}

/// 生产态 Metal 分块 GEMM 内核源：把 Gemm 骨架按 `tile×tile` 共享内存分块降成 MSL。
///
/// 只对 `M/N/K` 均能被 `tile` 整除的形状走分块（部分线程组会破坏 `threadgroup_barrier` 的
/// 统一控制流）；非 Gemm 骨架、`tile == 0` 或形状不整除时回落到 [`metal_grid_program`] 的
/// 网格/串行路径，因此本入口对全部骨架都有定义。语义与模板逐元素相同（`Y = A·B`），只换遍历。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_tiled_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
    tile: usize,
    per_thread: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    if !tile_ready(&run, m, n, tile, per_thread) {
        return lower(&run, &prepared, m, n);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::TiledGemm { tile, per_thread },
            &run,
            &prepared,
            m,
            n,
        )?,
        "分块 gemm",
    )
}

/// 分块 GEMM 的前置条件：Gemm 骨架、块几何合法、`M/N/K` 都能被 `tile` 整除。
fn tile_ready(run: &Run<'_>, m: usize, n: usize, tile: usize, per_thread: usize) -> bool {
    if !is_gemm(run, tile) {
        return false;
    }
    if !valid_block(tile, per_thread) {
        return false;
    }
    let k = (run.model.sample)(m, n).scalar("K").unwrap_or(0);
    divisible_shape(m, n, k, tile)
}

/// 算子是否是可分块 GEMM，且块边长合法（`tile == 0` 恒假，避免除零）。
fn is_gemm(run: &Run<'_>, tile: usize) -> bool {
    tile != 0 && run.op.skeleton == Skeleton::Gemm
}

/// 块几何：每线程输出 `1`（简单分块）或 `4`（4×4 寄存器分块），且线程组每维在 `[2, 32]`。
fn valid_block(tile: usize, per_thread: usize) -> bool {
    if per_thread != 1 && per_thread != 4 {
        return false;
    }
    let threads = tile / per_thread;
    (2..=32).contains(&threads) && tile.is_multiple_of(per_thread)
}

/// 形状能否整除分块：`M/N/K` 都是 `tile` 的正整数倍。
fn divisible_shape(m: usize, n: usize, k: i64, tile: usize) -> bool {
    k > 0 && divides(m, tile) && divides(n, tile) && divides(k as usize, tile)
}

/// `value` 能否被 `tile` 整除（`tile == 0` 时恒假，避免除零）。
fn divides(value: usize, tile: usize) -> bool {
    tile != 0 && value.is_multiple_of(tile)
}

/// 生产态 Metal 分块注意力内核源：一个 threadgroup 负责一个查询行，`threads` 个线程分摊 S 个
/// 分数与 D 个输出维，行内 max/sum 走线程组树归约，完整分数留在 threadgroup 内存。
///
/// 只对「elementwise 骨架 + 4 缓冲 q/k/v/y + 2 标量 S/D」的注意力形走分块；线程数不是 2 的幂、
/// 分数块超出线程组内存上限、或签名不符时回落到 [`metal_grid_program`] 的网格/串行路径。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_tiled_attention_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    s: usize,
    d: usize,
    threads: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", s, d)?;
    if !attention_ready(&run, s, d, threads) {
        return lower(&run, &prepared, s, d);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::TiledAttention { threads },
            &run,
            &prepared,
            s,
            d,
        )?,
        "分块 attention",
    )
}

/// 生产态 Metal simdgroup GEMM 内核源：每个 simdgroup 用 `8×8×8` 矩阵乘指令算
/// `row_frags × col_frags` 个 `8×8` 输出片段，K 轴按 8 前进。
///
/// 只对「Gemm 骨架 + `K % 8 == 0` + `M` 被 `8·row_frags` 整除 + `N` 被 `8·col_frags`
/// 整除」的形状走 simdgroup；片段数不在 `1..=4` 或形状不满足时回落到 [`metal_grid_program`]。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_simdgroup_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
    row_frags: usize,
    col_frags: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    if !simdgroup_ready(&run, m, n, row_frags, col_frags) {
        return lower(&run, &prepared, m, n);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::SimdGroupGemm {
                row_frags,
                col_frags,
            },
            &run,
            &prepared,
            m,
            n,
        )?,
        "simdgroup gemm",
    )
}

/// simdgroup GEMM 的前置条件：片段数合法且 `M/N/K` 满足 8 的整除约束。
fn simdgroup_ready(run: &Run<'_>, m: usize, n: usize, row_frags: usize, col_frags: usize) -> bool {
    fragment_span(row_frags, col_frags) && fragment_shape(run, m, n, row_frags, col_frags)
}

/// 每个 simdgroup 沿 M/N 的输出片段数限定在 `1..=4`（`8×8` 片段，最多 `32×32` 输出）。
fn fragment_span(row_frags: usize, col_frags: usize) -> bool {
    (1..=4).contains(&row_frags) && (1..=4).contains(&col_frags)
}

/// Gemm 骨架 + K 轴能被 8 整除 + M/N 分别能被 `8·片段数` 整除。
fn fragment_shape(run: &Run<'_>, m: usize, n: usize, row_frags: usize, col_frags: usize) -> bool {
    if run.op.skeleton != Skeleton::Gemm {
        return false;
    }
    let k = (run.model.sample)(m, n).scalar("K").unwrap_or(0);
    k > 0
        && (k as usize).is_multiple_of(8)
        && divides(m, 8 * row_frags)
        && divides(n, 8 * col_frags)
}

/// 生产态 Metal 张量核 GEMM 内核源：一个 `tile_m × tile_n` 输出块由 `32·simdgroups` 个线程
/// 协同算完，规约轴按 `k_tile` 切片，乘加走硬件的矩阵乘单元。
///
/// **布局前置**：张量核要的 B 是「规则约轴连续、按 half 存」的 `N×K` 权重（见
/// `engine::tiled::metal_transpose_half`），与模板里 `K×N` 的 f32 B 不同形。调用方必须先跑
/// 一遍布局前置，再拿本入口生成的主内核反复派发；本入口只负责描述乘加本身。
///
/// 只对「Gemm 骨架 + `M` 被 `tile_m` 整除 + 几何参数合法」的形状走张量核，其余回落到
/// [`metal_grid_program`]，因此本入口对全部骨架都有定义。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_tensor_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
    geometry: TensorGeometry,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    if !tensor_ready(&run, m, n, geometry) {
        return lower(&run, &prepared, m, n);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::TensorGemm { geometry },
            &run,
            &prepared,
            m,
            n,
        )?,
        "张量核 gemm",
    )
}

/// 张量核 GEMM 的前置条件：Gemm 骨架、几何参数成闭集、`M` 被 `tile_m` 整除。
///
/// `M` 的整除性是硬约束——`matmul2d` 按整块读 A，尾块会读越界；N 维不必对齐（装载零填、
/// 写回裁边），K 维不必对齐（末块走 `dynamic_extent`）。线程组内存只随 `tile_n` 增长。
fn tensor_ready(run: &Run<'_>, m: usize, n: usize, geometry: TensorGeometry) -> bool {
    run.op.skeleton == Skeleton::Gemm
        && n > 0
        && divides(m, geometry.tile_m)
        && geometry_ready(geometry)
}

/// 张量核几何的三档前置都满足：跨度档位、规约块/simdgroup、线程组内存占用。
fn geometry_ready(geometry: TensorGeometry) -> bool {
    geometry.spans_ok() && geometry.block_ok() && geometry.fits_threadgroup_memory()
}

/// 生产态 Metal int8 分块 GEMM 内核源：`A[M,K]·B[K,N]`，`i8` 输入、`i32` 精确累加、`f32` 输出。
///
/// 只对「Gemm 骨架 + 两个 `u8` 输入、一个 `f32` 输出、三个整数标量 + `M/N/K` 均能被 `tile`
/// 整除」的签名走 int8 共享内存分块；`tile` 越界、骨架或签名不符、形状不整除时回落到
/// [`metal_grid_program`] 的网格/串行路径，因此本入口对全部骨架都有定义。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_int8_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
    tile: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    if !int8_ready(&run, m, n, tile) {
        return lower(&run, &prepared, m, n);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::Int8Gemm { tile },
            &run,
            &prepared,
            m,
            n,
        )?,
        "int8 分块 gemm",
    )
}

/// int8 GEMM 的前置条件：块边长合法、Gemm 骨架、int8 签名、`M/N/K` 都能被 `tile` 整除。
fn int8_ready(run: &Run<'_>, m: usize, n: usize, tile: usize) -> bool {
    if !(2..=32).contains(&tile) || run.op.skeleton != Skeleton::Gemm {
        return false;
    }
    if !int8_signature(run) {
        return false;
    }
    let k = (run.model.sample)(m, n).scalar("K").unwrap_or(0);
    divisible_shape(m, n, k, tile)
}

/// 签名是不是 int8 GEMM：`(u8 *a, u8 *b, float *y, int M, int N, int K)`。
///
/// 按位置核对类型而不是按参数名：ABI 的顺序就是绑定顺序，名字只用于样本取值。
fn int8_signature(run: &Run<'_>) -> bool {
    let params = run.op.signature.params();
    matches!(params, [a, b, y, m, n, k]
        if a.ctype == CType::U8
            && b.ctype == CType::U8
            && y.ctype == CType::F32
            && [m, n, k].iter().all(|param| param.ctype == CType::Int))
}

/// 生产态 Metal flash attention 内核源：一个 threadgroup 负责一个查询行，K/V 按 `block`
/// 个键分块流式扫描，在线 softmax 的 running max/sum 让输出累加在每线程寄存器里，
/// 不保留完整 `S` 长度的分数向量。
///
/// 只对「elementwise 骨架 + 4 缓冲 q/k/v/y + 2 标量 S/D + 线程数为 2 的幂 + block 为线程数
/// 倍数 + `D ≤ 64`」的形状走 flash；不满足时回落到 [`metal_grid_program`]。
///
/// # Errors
///
/// 算子没有声明、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_flash_attention_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    s: usize,
    d: usize,
    threads: usize,
    block: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", s, d)?;
    if !flash_ready(&run, s, d, threads, block) {
        return lower(&run, &prepared, s, d);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::FlashAttention { threads, block },
            &run,
            &prepared,
            s,
            d,
        )?,
        "flash attention",
    )
}

/// flash attention 的前置条件：线程/块几何合法、elementwise 骨架、注意力签名与形状。
fn flash_ready(run: &Run<'_>, s: usize, d: usize, threads: usize, block: usize) -> bool {
    tileable_threads(threads)
        && batching_block(threads, block)
        && run.op.skeleton == Skeleton::Elementwise
        && attention_shape(run, s, d)
}

/// 键块大小：`[32, 1024]` 内且是线程数的整数倍，保证每轮线程负载均匀。
fn batching_block(threads: usize, block: usize) -> bool {
    (32..=1024).contains(&block) && block.is_multiple_of(threads)
}

/// 分块 attention 的前置条件：线程数合法、elementwise 骨架、4 缓冲 + 2 标量的注意力签名。
fn attention_ready(run: &Run<'_>, s: usize, d: usize, threads: usize) -> bool {
    if !tileable_threads(threads) {
        return false;
    }
    if run.op.skeleton != Skeleton::Elementwise {
        return false;
    }
    attention_shape(run, s, d)
}

/// 线程数必须是 [32, 1024] 内的 2 的幂（树归约与线程组上限）。
fn tileable_threads(threads: usize) -> bool {
    threads.is_power_of_two() && (32..=1024).contains(&threads)
}

/// 注意力签名与形状：4 缓冲（q/k/v/y）+ 2 标量（S/D），且形状落在共享内存/头宽上限内。
fn attention_shape(run: &Run<'_>, s: usize, d: usize) -> bool {
    let params = run.op.signature.params();
    let buffers = params.iter().filter(|p| p.ctype.is_buffer()).count();
    let scalars = params.iter().filter(|p| !p.ctype.is_buffer()).count();
    buffers == 4 && scalars == 2 && in_bounds(s, d)
}

/// `S·4` 字节的分数块不超过 32 KiB 线程组内存，头宽不超过模板的 64。
fn in_bounds(s: usize, d: usize) -> bool {
    s > 0 && d > 0 && s <= 8192 && d <= 64
}

/// 生产态分块扫描内核源：走完整条流水线后把时间轴递推切成块，返回两段 MSL `(局部扫描, 校正)`，
/// 主机在两次派发间做块级携带前缀。扫描是线性递推，分块天然需要两段内核，故不并入单内核的
/// [`metal_grid_program`]；串行（未声明 `[chunk]`）仍走 [`metal_grid_program`]。
///
/// # Errors
///
/// 算子没有声明 `[chunk]`、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_scan_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    scan_chunked(&run, &prepared, m, n)
}

/// 生产态分块 RMSNorm 内核源：走完整条流水线后把「平方和归约 + 归一化」切成两段，返回
/// `(部分平方和, 归一化)`，主机在两次派发间折叠部分和算出 `rms`。串行（未声明 `[chunk]`）仍走
/// [`metal_grid_program`]。
///
/// # Errors
///
/// 算子没有声明 `[chunk]`、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_rmsnorm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    let chunk = chunk_size(run.op, "cutile::execution::metal_rmsnorm_program")?;
    pair(
        lower_sources(
            Emit::Metal,
            &Lower::ChunkRmsnorm { chunk },
            &run,
            &prepared,
            m,
            n,
        )?,
        "rmsnorm",
    )
}

/// 生产态分块 Softmax 内核源：走完整条流水线后把「最大归约 + 指数和归约 + 归一化」切成三段，返回
/// `(部分最大, 部分指数和, 归一化)`，主机在派发间折叠出 `max` 与 `sum`。串行（未声明 `[chunk]`）
/// 仍走 [`metal_grid_program`]。
///
/// # Errors
///
/// 算子没有声明 `[chunk]`、`metal` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn metal_softmax_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    let chunk = chunk_size(run.op, "cutile::execution::metal_softmax_program")?;
    triple(
        lower_sources(
            Emit::Metal,
            &Lower::ChunkSoftmax { chunk },
            &run,
            &prepared,
            m,
            n,
        )?,
        "softmax",
    )
}
