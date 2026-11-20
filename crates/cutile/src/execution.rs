//! 真机执行：用一份给定样本走完整条流水线并执行一次。
//!
//! 与 `verify` 的分工是**样本从哪来**：`verify` 用算子声明的确定性样本，回答"这台机器能算这条
//! 算子"；本模块用调用方给的真实数据，回答"这份数据算出来的数对不对"。两者共用同一条
//! `pipeline`，因此不存在第二套执行路径。

use crate::atom::{AttrKey, Prim};
use crate::backend::{Launch, Lower};
use crate::build;
use crate::builtin::Catalogue;
use crate::cap::{Path, Proof};
use crate::decl::Skeleton;
use crate::dirs::Dirs;
use crate::emit::Fragment;
use crate::pipeline::{Prepared, Run};
use crate::prof::{Emit, Profile};
use crate::witness;
use diagnostic::{Code, Error};
use op::decl::{Grid, OpDecl};
use op::model::Sample;
use op::sig::CType;

/// 一次真机执行：调用方给的样本 → 编译 → 跑一次 → 与标量参考对拍。
#[derive(Debug, Clone, PartialEq)]
pub struct Execution {
    /// 设备名（档案里的 `id`）。
    pub device: String,
    /// 厂商。
    pub vendor: String,
    /// 协商选中的变体。
    pub variant: String,
    /// 选中的 tile。
    pub tile: String,
    /// 缓存键（源文本的身份）。
    pub key: String,
    /// 生成的片段行数。
    pub fragment: Fragment,
    /// 内核输出。
    pub output: Vec<f32>,
    /// 标量参考输出。
    pub expected: Vec<f32>,
    /// 最大相对误差。
    pub max_rel_err: f64,
    /// 对拍阈值（由算子声明的元素类型推出）。
    pub tolerance: f64,
}

/// 用一份**给定的**样本走完整条流水线并执行一次。
///
/// 与 [`crate::verify::verify`] 的区别都是刻意的：样本由调用方给（真实权重），因此不做形状矩阵、
/// 只跑这一份；执行前仍要求设备有见证证据——没有证据的后端算出来的数不是结果，只是现象；
/// 对拍不通过是错误（[`Code::Invalid`]），不是"环境问题"。
///
/// 回答什么：这份数据在这台机器上算出来的数对不对。不回答什么：这台机器一般能不能算这条算子
/// （那是形状矩阵的事），也不改档案里的等级。
///
/// # Errors
///
/// 算子没有声明或没有语义表、设备不可用、设备没有执行证据、协商失败、形状与算子粒度不符、
/// 签名与样本不一致、编译/执行失败，或对拍超出容差时返回说明。
pub fn execute(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    device: &str,
    sample: &Sample,
) -> diagnostic::Result<Execution> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let profile = catalogue.resolve(device)?;
    let run = established(op, profile, dirs)?;
    one(&run, profile, dirs, sample)
}

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

/// 生产态 Metal 张量核 GEMM 内核源：一个 `tile × tile` 输出块由 `32·simdgroups` 个线程协同
/// 算完，规约轴按 `k_tile` 切片，乘加走硬件的矩阵乘单元。
///
/// **布局前置**：张量核要的 B 是「规则约轴连续、按 half 存」的 `N×K` 权重（见
/// `engine::tiled::metal_transpose_half`），与模板里 `K×N` 的 f32 B 不同形。调用方必须先跑
/// 一遍布局前置，再拿本入口生成的主内核反复派发；本入口只负责描述乘加本身。
///
/// 只对「Gemm 骨架 + `M` 被 `tile` 整除 + 几何参数合法」的形状走张量核，其余回落到
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
    tile: usize,
    k_tile: usize,
    simdgroups: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    if !tensor_ready(&run, m, n, tile, k_tile, simdgroups) {
        return lower(&run, &prepared, m, n);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::TensorGemm {
                tile,
                k_tile,
                simdgroups,
            },
            &run,
            &prepared,
            m,
            n,
        )?,
        "张量核 gemm",
    )
}

/// 张量核 GEMM 的前置条件：Gemm 骨架、几何参数成闭集、`M` 被 `tile` 整除。
///
/// `M` 的整除性是硬约束——`matmul2d` 按整块读 A，尾块会读越界；N 维不必对齐（装载零填、
/// 写回裁边），K 维不必对齐（末块走 `dynamic_extent`）。
fn tensor_ready(
    run: &Run<'_>,
    m: usize,
    n: usize,
    tile: usize,
    k_tile: usize,
    simdgroups: usize,
) -> bool {
    run.op.skeleton == Skeleton::Gemm
        && n > 0
        && divides(m, tile)
        && matches!(tile, 64 | 128)
        && k_tile >= 32
        && k_tile.is_multiple_of(16)
        && matches!(simdgroups, 4 | 8)
        && tile * (k_tile + 4) * 2 <= 32 * 1024
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

/// 生产态 CUDA 逐点网格内核源：走完整条流水线后把逐点内核包成按网格派发的 `kernel_main`。
/// 本入口只覆盖「纯逐点」签名（[`OpDecl::pointwise`]）；gemm/attention 的循环注入与串行兜底的
/// CUDA 镜像后续单独落地，不在此入口静默生成错的内核。
///
/// # Errors
///
/// `nvidia` 档案缺失、设备缺执行证据、协商失败、形状非法或算子非逐点形时返回说明。
pub fn cuda_grid_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let op = run.op;
    if !op.pointwise() {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "`{}` 不是纯逐点形，CUDA 逐点网格未覆盖（循环注入/串行兜底后续落地）",
                op.name
            ),
        )
        .with("cutile::execution::cuda_grid_program"));
    }
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Cuda)?.lower(
            &prepared.source.text,
            &op.signature,
            &sample,
            &Lower::Pointwise,
        )?,
        "cuda 逐点",
    )
}

/// 生产态 CUDA 分块归约内核源：走完整条流水线后把长轴切成块，每线程归约一块写部分结果 `y[g]`，
/// 主机折叠部分结果得最终标量。与 [`metal_grid_program`] 的归约骨架同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_reduce_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    single(
        lower_sources(Emit::Cuda, &reduce_kind(run.op)?, &run, &prepared, m, n)?,
        "cuda 归约",
    )
}

/// 生产态 CUDA 分块 RMSNorm 内核源：两段（部分平方和 + 归一化），主机在两次派发间折叠出
/// `rms` 再跑第二段。与 [`metal_rmsnorm_program`] 同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_rmsnorm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let chunk = chunk_size(run.op, "cutile::execution::cuda_rmsnorm_program")?;
    pair(
        lower_sources(
            Emit::Cuda,
            &Lower::ChunkRmsnorm { chunk },
            &run,
            &prepared,
            m,
            n,
        )?,
        "cuda rmsnorm",
    )
}

/// 生产态 CUDA 分块 Softmax 内核源：三段（部分最大 + 部分指数和 + 归一化），主机在派发间折叠
/// `max` 与 `sum`。与 [`metal_softmax_program`] 同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_softmax_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String, String)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let chunk = chunk_size(run.op, "cutile::execution::cuda_softmax_program")?;
    triple(
        lower_sources(
            Emit::Cuda,
            &Lower::ChunkSoftmax { chunk },
            &run,
            &prepared,
            m,
            n,
        )?,
        "cuda softmax",
    )
}

/// 生产态 CUDA 循环注入网格内核（`Gemm` 骨架：gemv 一维 / gemm 二维共用一条 lowering）。
///
/// 返回 `(设备源, 各维网格大小)`：网格大小按 `[grid]` 声明的标量从样本取值（`dims[i]` 对应
/// `kernel_main<<<dim3(dims[0], dims[1], dims[2]), 1>>>` 的 x/y/z）；host 驱动由调用方用
/// `engine::cuda::cuda_loop_driver` 拼——设备段与驱动分属两层，机制层不生成 host C 文本。
///
/// # Errors
///
/// 算子没声明 `[grid]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, Vec<usize>)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let grid = grid_of(run.op, op_name)?;
    let dims = grid_dims(grid, &(run.model.sample)(m, n))?;
    Ok((cuda_loop_source(grid, &run, &prepared, m, n)?, dims))
}

/// 算子的 `[grid]` 声明：没有声明就没有网格，循环注入无从谈起。
fn grid_of<'a>(op: &'a OpDecl, op_name: &str) -> diagnostic::Result<&'a Grid> {
    op.grid.as_ref().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("`{op_name}` 没有声明 `[grid]`，CUDA 循环注入不适用"),
        )
        .with("cutile::execution::cuda_gemm_program")
    })
}

/// CUDA 循环注入的设备源：`Loop` 降级后只取那一段源码（单文件即单翻译单元）。
fn cuda_loop_source(
    grid: &Grid,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    single(
        lower_sources(Emit::Cuda, &Lower::Loop(grid.clone()), run, prepared, m, n)?,
        "cuda gemm",
    )
}

/// 网格每一维的大小：把 `[grid]` 里的标量名按样本求值成 `usize`。
///
/// 抽出来的理由是可测性：`cuda_gemm_program` 本来就是编排（解析 → 求维 → 降级 → 取单源），
/// 让它在循环里再摊三层错误分支，读的人得同时跟两条线。
fn grid_dims(grid: &Grid, sample: &Sample) -> diagnostic::Result<Vec<usize>> {
    grid.dims
        .iter()
        .map(|(_, scalar)| dim_size(scalar, sample))
        .collect()
}

/// 一个网格标量的大小：样本没有取值、或取值不是非负 `usize`，都算声明与样本对不上。
fn dim_size(scalar: &str, sample: &Sample) -> diagnostic::Result<usize> {
    let size = sample.scalar(scalar).ok_or_else(|| {
        Error::new(Code::Invalid, format!("网格标量 `{scalar}` 没有样本取值"))
            .with("cutile::execution::cuda_gemm_program")
    })?;
    usize::try_from(size).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("网格标量 `{scalar}` = {size} 不是非负 usize"),
        )
        .with("cutile::execution::cuda_gemm_program")
    })
}

/// 生产态 CUDA 张量核（WMMA）GEMM 内核源：`64×64` 输出块、`BK=8`、4 个 `32×32×8` 片段。
///
/// 形状在生成期写死（每形状一份产物）：`M`/`N` 整除 `64` 时无守卫，否则载入/写出带行列守卫。
/// host 驱动由调用方用 `engine::wmma::gemm_driver` 生成；`K % 8 != 0` 时回落
/// [`cuda_gemm_program`] 的循环注入路径。
///
/// # Errors
///
/// 算子不是 `Gemm` 骨架、`nvidia` 档案缺失、设备缺执行证据、协商失败、形状非法或 `K % 8 != 0`。
pub fn cuda_wmma_gemm_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    single(
        lower_sources(Emit::Cuda, &Lower::WmmaGemm, &run, &prepared, m, n)?,
        "cuda wmma gemm",
    )
}

/// 生产态 CUDA 分块扫描内核源：两段（局部扫描 + 校正），主机在两次派发间折叠携带前缀
/// `C_{g+1} = a^len·C_g + 尾态`。与 [`metal_scan_program`] 同语义，仅后端方言差异。
///
/// # Errors
///
/// 算子没声明 `[chunk]`、`nvidia` 档案缺失、设备缺执行证据、协商失败或形状非法时返回说明。
pub fn cuda_scan_program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String, f32)> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "nvidia", m, n)?;
    let decay = scan_decay(run.op)?;
    let (local, correct) = scan_pair(Emit::Cuda, &run, &prepared, m, n, "cuda scan")?;
    Ok((local, correct, decay))
}

/// 扫描的两段源码：块大小与衰减 → `Lower` → 降源 → `(局部扫描, 校正)`。
///
/// 衰减读两遍是有意的：它既要进 `Lower`（分块几何），又要单独回给主机折叠携带前缀。
fn scan_pair(
    emit: Emit,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
    role: &str,
) -> diagnostic::Result<(String, String)> {
    let kind = scan_kind(run.op)?;
    pair(lower_sources(emit, &kind, run, prepared, m, n)?, role)
}

/// 解析算子/档案/证据并编译出可执行产物：`(流水线, 产物)`。
fn resolve<'a>(
    catalogue: &'a Catalogue,
    dirs: &Dirs,
    op_name: &str,
    vendor: &str,
    m: usize,
    n: usize,
) -> diagnostic::Result<(Run<'a>, Prepared)> {
    let (op, profile, proof) = lookup(catalogue, dirs, op_name, vendor)?;
    prepare(op, profile, proof, dirs, m, n)
}

/// 找算子与 `vendor` 档案，验证据后给出证明。
fn lookup<'a>(
    catalogue: &'a Catalogue,
    dirs: &Dirs,
    op_name: &str,
    vendor: &str,
) -> diagnostic::Result<(&'a OpDecl, &'a Profile, Proof)> {
    let Some(op) = catalogue.op(op_name) else {
        return Err(Error::new(Code::Missing, format!("没有算子 `{op_name}`")));
    };
    let Some(profile) = catalogue.profile(vendor) else {
        return Err(Error::new(Code::Missing, format!("没有 `{vendor}` 档案")));
    };
    let evidence = witness::establish(profile, dirs);
    let proof = evidence.proof(profile);
    if let Some(error) = evidence.failure {
        return Err(error.with(format!(
            "`{vendor}` 设备缺执行证据，不在未验证的后端上生成生产内核"
        )));
    }
    Ok((op, profile, proof))
}

/// 建流水线、协商、编译出源文本。
fn prepare<'a>(
    op: &'a OpDecl,
    profile: &'a Profile,
    proof: Proof,
    dirs: &Dirs,
    m: usize,
    n: usize,
) -> diagnostic::Result<(Run<'a>, Prepared)> {
    let run = Run::new(op, profile, proof)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let prepared = run.prepare(&path, dirs, m, n)?;
    Ok((run, prepared))
}

/// 逐点网格的块大小。派发几何归 cutile 定（几何由声明与样本推出），`engine` 的驱动只按它发射。
pub const POINT_THREADS: usize = 256;

/// 声明 → 并行形态：`kind` 说怎么降、`grid` 说派多少次。**唯一分派点**。
///
/// verify / bench 的设备通道与各生产态入口都从这里取，`engine` 只照 [`Launch`] 装配、不第二次判断
/// 该并行还是串行，也不按算子名挑驱动（按名字挑是第二份分派，必然与 `[chunk]` / `[grid]` 漂移）。
///
/// # Errors
///
/// 网格维或长度标量在样本里取不到合法值，或分块族缺 `[chunk]` / 扫描衰减。
pub fn launch(op: &OpDecl, sample: &Sample) -> diagnostic::Result<Launch> {
    let kind = kind(op)?;
    let items = items_of(&kind, op, sample)?;
    // 逐点才成组（一元素一线程、块内 `POINT_THREADS` 线程）；分块与循环注入是「一块/一维一线程」，
    // 组内再塞线程只会让同一块被多算几遍。
    let group = match kind {
        Lower::Pointwise => POINT_THREADS,
        _ => 1,
    };
    Ok(Launch { kind, items, group })
}

/// 骨架分派：串行兜底 + 网格快路径 + 分块族。不回答派发几何（那是 [`grid_of_kind`]）。
fn kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    match op.skeleton {
        Skeleton::Gemm => Ok(grid_kind(op)),
        Skeleton::Elementwise => Ok(chunked_elementwise(op)),
        _ => chunked_kind(op),
    }
}

/// 非逐元素、非 gemm 的两族：声明了 `[chunk]` 就分块降级，否则串行。
///
/// 扫描沿时间轴串行递推，正文就是一条跨元素循环；网格按元素拆只会让每线程各算一整遍。
/// 归约未声明 `[chunk]` 时同理：串行（`grid = [1,1,1]`）。
fn chunked_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    match (op.skeleton, op.chunk.is_some()) {
        (Skeleton::Reduce, true) => reduce_kind(op),
        (Skeleton::Scan, true) => scan_kind(op),
        _ => Ok(serial_kind()),
    }
}

/// 串行兜底：零维网格，正文原样跑一遍。
fn serial_kind() -> Lower {
    Lower::Loop(Grid { dims: Vec::new() })
}

/// 声明了 `[grid]` 的骨架：按 `[grid]` 各维拆线程（gemv 一维、gemm 二维共用一条）。
fn grid_kind(op: &OpDecl) -> Lower {
    op.grid
        .as_ref()
        .map_or_else(serial_kind, |grid| Lower::Loop(grid.clone()))
}

/// 逐元素族：纯逐点直接开线程，其余按 `[chunk]` 分块降级（[`chunked_elementwise`]）。
///
/// 分块逐元素：`[chunk]` 遇上 `[needs] reduce` 的 `max` 是三趟 softmax（最大 → 指数和 → 归一化），
/// 遇上 `[needs] dot` 是两趟 rmsnorm（平方和 → 归一化）；其余正文已沿轴走完或下标依赖行 gather，
/// 并行按元素拆会算错，退串行。
fn chunked_elementwise(op: &OpDecl) -> Lower {
    if op.pointwise() {
        return Lower::Pointwise;
    }
    let Some(chunk) = op.chunk.as_ref() else {
        return grid_kind(op);
    };
    if reduce_op(op) == Some("max") {
        return Lower::ChunkSoftmax { chunk: chunk.size };
    }
    if op.needs.iter().any(|need| need.prim == Prim::Dot) {
        return Lower::ChunkRmsnorm { chunk: chunk.size };
    }
    serial_kind()
}

/// 派发几何（工作项数）：逐点是元素数、循环注入按 `[grid]` 各维取值、分块族按块数。
fn items_of(kind: &Lower, op: &OpDecl, sample: &Sample) -> diagnostic::Result<[usize; 3]> {
    if let Lower::Loop(grid) = kind {
        return Ok(pad(&grid_dims(grid, sample)?));
    }
    match slab(kind) {
        Some(chunk) => Ok(along(length_of(op, sample)?.div_ceil(chunk))),
        None => Ok([1, 1, 1]),
    }
}

/// 沿长度轴展开的块大小：逐点一族格即元素、分块族按 `[chunk]` 声明。
///
/// `None` 是「这一形态不沿长度轴」——gather 一类的下标依赖行不能按元素拆，几何固定为一格。
fn slab(kind: &Lower) -> Option<usize> {
    match kind {
        Lower::Pointwise => Some(1),
        Lower::ChunkReduce { chunk, .. }
        | Lower::ChunkRmsnorm { chunk }
        | Lower::ChunkSoftmax { chunk }
        | Lower::ChunkScan { chunk, .. } => Some(*chunk),
        _ => None,
    }
}

/// 沿长度轴的一维几何：空长仍给一格，否则网格为零既不执行也不报错。
fn along(len: usize) -> [usize; 3] {
    [len.max(1), 1, 1]
}

/// 三维补一：`[d0, d1]` → `[d0, d1, 1]`；空网格是 `[1, 1, 1]`。`[grid]` 声明层已限三维修。
fn pad(dims: &[usize]) -> [usize; 3] {
    let mut grid = [1usize; 3];
    for (slot, dim) in dims.iter().take(3).enumerate() {
        grid[slot] = *dim;
    }
    grid
}

/// 长度标量：签名里唯一的标量（`N` / `n` / `len`），网格按它算。
fn length_of(op: &OpDecl, sample: &Sample) -> diagnostic::Result<usize> {
    let name = op
        .signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .map(|param| param.name.clone())
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("`{}` 没有长度标量，网格无从派生", op.name),
            )
            .with("cutile::execution::launch")
        })?;
    dim_size(&name, sample)
}

/// 归约需求的 `op` 取值（`sum` / `max` / …）：从声明的 `[needs] reduce` 读，不看算子名。
fn reduce_op(op: &OpDecl) -> Option<&str> {
    op.needs
        .iter()
        .find(|need| need.prim == Prim::Reduce)?
        .attrs
        .iter()
        .find(|attr| attr.key() == AttrKey::Op)?
        .values()
        .first()
        .map(String::as_str)
}

/// 归约原语名：从声明的 `[needs] reduce`（`op` + `dt`）推出 preamble 里的软件归约函数
/// （`_cutile_sum_f32` / `_cutile_max_f32`），与模板 `@reduce(..., func: _cutile_*)` 同一约定，
/// 不对模板文本做字符串猜测。
fn reduce_func(op: &OpDecl) -> diagnostic::Result<String> {
    let need = op
        .needs
        .iter()
        .find(|need| need.prim == Prim::Reduce)
        .ok_or_else(|| Error::new(Code::Invalid, format!("`{}` 没有归约需求", op.name)))?;
    let op_name = reduce_op(op)
        .ok_or_else(|| Error::new(Code::Invalid, format!("`{}` 的归约需求缺 `op`", op.name)))?;
    let dt = need.dt.names().first().copied().unwrap_or("f32");
    Ok(format!("_cutile_{op_name}_{dt}"))
}

/// 单份 MSL 的形态：分块归一化 / 分块扫描是多段源，单源入口没有主机折叠的位置，退串行（多段由
/// `metal_rmsnorm_program` / `metal_softmax_program` / `metal_scan_program` 各自主张）。
fn single_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    match kind(op)? {
        Lower::ChunkRmsnorm { .. } | Lower::ChunkSoftmax { .. } | Lower::ChunkScan { .. } => {
            Ok(serial_kind())
        }
        kind => Ok(kind),
    }
}

/// 生产态 MSL：声明 → 单份形态 → 降源。
fn lower(run: &Run<'_>, prepared: &Prepared, m: usize, n: usize) -> diagnostic::Result<String> {
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Metal)?.lower(
            &prepared.source.text,
            &run.op.signature,
            &sample,
            &single_kind(run.op)?,
        )?,
        "降源",
    )
}

/// 扫描分块：声明了 `[chunk]` 的走「局部扫描 + 校正」两段内核，返回 `(局部扫描, 校正)`；未声明
/// 时报错（串行由 [`metal_grid_program`] 的骨架兜底覆盖，此处只做分块）。
fn scan_chunked(
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<(String, String)> {
    pair(
        lower_sources(Emit::Metal, &scan_kind(run.op)?, run, prepared, m, n)?,
        "扫描",
    )
}

/// 扫描衰减 `a`：从 `[consts]` 的 `a` 槽读出（扫描的离散对角半群步），解析成 f32 供校正内核拼
/// 几何级数。找不到或非数值时报错，不静默用默认值。
fn scan_decay(op: &OpDecl) -> diagnostic::Result<f32> {
    let raw = op
        .consts
        .iter()
        .find(|(key, _)| key == "a")
        .map(|(_, value)| value.as_str())
        .ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("`{}` 的 `[consts]` 缺扫描衰减 `a`", op.name),
            )
        })?;
    raw.parse::<f32>().map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("`{}` 的扫描衰减 `a` 不是数值：`{raw}`", op.name),
        )
    })
}

/// 本机只编译 C，且证据先于执行：没有执行证据的后端算出来的数不是结果，只是现象。
fn established<'a>(
    op: &'a OpDecl,
    profile: &'a Profile,
    dirs: &Dirs,
) -> diagnostic::Result<Run<'a>> {
    if profile.meta.emit != Emit::C {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "设备 `{}` 的输出语言是 `{}`，本机只能编译执行 C",
                profile.meta.id, profile.meta.emit
            ),
        ));
    }
    let evidence = witness::establish(profile, dirs);
    let proof = evidence.proof(profile);
    if let Some(error) = evidence.failure {
        return Err(error.with("该设备缺少执行证据，真实数据不在未验证的后端上运行"));
    }
    Run::new(op, profile, proof)
}

/// 一个样本：协商、形状校验、编译执行一次、对拍、组装报告。
fn one(
    run: &Run<'_>,
    profile: &Profile,
    dirs: &Dirs,
    sample: &Sample,
) -> diagnostic::Result<Execution> {
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    check_sample(u64::from(run.model.group.max(1)), sample)?;
    let prepared = run.prepare(&path, dirs, sample.m, sample.n)?;
    run_once(run, profile, dirs, sample, path, &prepared)
}

/// 编译执行一次、与标量参考对拍，并组装报告（字段全部来自这次执行，不读档案自述）。
fn run_once(
    run: &Run<'_>,
    profile: &Profile,
    dirs: &Dirs,
    sample: &Sample,
    path: Path,
    prepared: &Prepared,
) -> diagnostic::Result<Execution> {
    let harness = prepared.harness(run.op, sample, 1)?;
    let tool = crate::toolchain::detect();
    let (outcome, _) = build::execute(profile, tool.as_ref(), &prepared.key, &harness, dirs)?;
    let expected = (run.model.reference)(sample)?;
    let tolerance = crate::reference::tolerance(run.model.dtype);
    let max_rel_err = check_parity(&outcome.bits, &expected, tolerance)?;
    Ok(Execution {
        device: profile.meta.id.to_string(),
        vendor: profile.meta.vendor.to_string(),
        variant: path.variant,
        tile: prepared.plan.tile.to_string(),
        key: prepared.key.hex(),
        fragment: Fragment::new(prepared.source.lines()),
        output: outcome.bits,
        expected,
        max_rel_err,
        tolerance,
    })
}

/// 样本形状校验：m/n 非零，且 m 是形状粒度的整数倍。
fn check_sample(group: u64, sample: &Sample) -> diagnostic::Result<()> {
    if sample.m == 0 || sample.n == 0 || !(sample.m as u64).is_multiple_of(group) {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "样本形状 {}x{} 不自洽：m 必须是形状粒度 {group} 的正整数倍",
                sample.m, sample.n
            ),
        ));
    }
    Ok(())
}

/// 对拍：长度一致且最大相对误差不超阈值，返回实测误差。
fn check_parity(bits: &[f32], expected: &[f32], tolerance: f64) -> diagnostic::Result<f64> {
    if bits.len() != expected.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("内核输出 {} 个数，样本声明 {}", bits.len(), expected.len()),
        ));
    }
    let max_rel_err = crate::reference::max_rel_err(bits, expected);
    if max_rel_err > tolerance {
        return Err(Error::new(
            Code::Invalid,
            format!("对拍失败：max_rel_err {max_rel_err:.3e} > {tolerance:.0e}"),
        ));
    }
    Ok(max_rel_err)
}

/// 经 [`crate::backend::Codegen::lower`] 缝取 vendor 后端的生产态设备源。机制层不认厂商，只认
/// `emit` 与 [`Lower`] 描述；cutile 内不出现 Metal / CUDA 字面量（L3 契约）。
fn backend(emit: Emit) -> diagnostic::Result<&'static dyn crate::backend::Backend> {
    crate::device::spec(emit).ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("本机没有 `{emit}` 语言的生产态后端"),
        )
    })
}

/// 样本 + 后端查表 + 生产态 lowering 收在一处：调用方只写一个 `?`，不逐处重复「取样本、查后端、
/// 降源」三段样板，也压低各入口的 Ccn。
fn lower_sources(
    emit: Emit,
    kind: &Lower,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<Vec<String>> {
    let sample = (run.model.sample)(m, n);
    backend(emit)?.lower(&prepared.source.text, &run.op.signature, &sample, kind)
}

/// 声明里必带的 `[chunk]` 块大小；缺声明时报 [`Code::Unsupported`]（串行兜底另有路径，这里只做分块）。
fn chunk_size(op: &OpDecl, location: &str) -> diagnostic::Result<usize> {
    let chunk = op.chunk.as_ref().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("`{}` 没有声明 `[chunk]`", op.name),
        )
        .with(location)
    })?;
    Ok(chunk.size)
}

/// 分块归约的 [`Lower`] 描述：块大小来自 `[chunk]`、软件归约原语名来自 `[needs] reduce`。
fn reduce_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    let chunk = chunk_size(op, "cutile::execution::cuda_reduce_program")?;
    let func = reduce_func(op)?;
    Ok(Lower::ChunkReduce { chunk, func })
}

/// 分块扫描的 [`Lower`] 描述：块大小来自 `[chunk]`、衰减 `a` 来自 `[consts]`。
fn scan_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    let chunk = chunk_size(op, "cutile::execution::metal_scan_program")?;
    let decay = scan_decay(op)?;
    Ok(Lower::ChunkScan { chunk, decay })
}

/// 从 lowering 结果取一段设备源；缺段是后端违约，机制层不猜。
fn pop_one(sources: &mut Vec<String>, role: &str) -> diagnostic::Result<String> {
    sources
        .pop()
        .ok_or_else(|| Error::new(Code::Internal, format!("{role} lowering 少一段设备源")))
}

/// lowering 结果应恰好一段设备源。
fn single(mut sources: Vec<String>, role: &str) -> diagnostic::Result<String> {
    pop_one(&mut sources, role)
}

/// lowering 结果应恰好两段，按弹出顺序还原 `(第一段, 第二段)`。
fn pair(mut sources: Vec<String>, role: &str) -> diagnostic::Result<(String, String)> {
    let b = pop_one(&mut sources, role)?;
    let a = pop_one(&mut sources, role)?;
    Ok((a, b))
}

/// lowering 结果应恰好三段，按弹出顺序还原。
fn triple(mut sources: Vec<String>, role: &str) -> diagnostic::Result<(String, String, String)> {
    let c = pop_one(&mut sources, role)?;
    let b = pop_one(&mut sources, role)?;
    let a = pop_one(&mut sources, role)?;
    Ok((a, b, c))
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
