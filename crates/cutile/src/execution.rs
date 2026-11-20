//! 真机执行：用一份给定样本走完整条流水线并执行一次。
//!
//! 与 `verify` 的分工是**样本从哪来**：`verify` 用算子声明的确定性样本，回答"这台机器能算这条
//! 算子"；本模块用调用方给的真实数据，回答"这份数据算出来的数对不对"。两者共用同一条
//! `pipeline`，因此不存在第二套执行路径。

use crate::atom::{AttrKey, Prim};
use crate::backend::Lower;
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
/// 返回可交给 `engine::metal::MetalExecutor::register` 进程内编译的 MSL。
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
) -> diagnostic::Result<String> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, "metal", m, n)?;
    if !tile_ready(&run, m, n, tile) {
        return lower(&run, &prepared, m, n);
    }
    single(
        lower_sources(
            Emit::Metal,
            &Lower::TiledGemm { tile },
            &run,
            &prepared,
            m,
            n,
        )?,
        "分块 gemm",
    )
}

/// 分块 GEMM 的前置条件：Gemm 骨架、`tile > 0`、`M/N/K` 都能被 `tile` 整除。
fn tile_ready(run: &Run<'_>, m: usize, n: usize, tile: usize) -> bool {
    if !is_gemm(run, tile) {
        return false;
    }
    let k = (run.model.sample)(m, n).scalar("K").unwrap_or(0);
    divisible_shape(m, n, k, tile)
}

/// 算子是否是可分块 GEMM，且块边长合法（`tile == 0` 恒假，避免除零）。
fn is_gemm(run: &Run<'_>, tile: usize) -> bool {
    tile != 0 && run.op.skeleton == Skeleton::Gemm
}

/// 形状能否整除分块：`M/N/K` 都是 `tile` 的正整数倍。
fn divisible_shape(m: usize, n: usize, k: i64, tile: usize) -> bool {
    k > 0 && divides(m, tile) && divides(n, tile) && divides(k as usize, tile)
}

/// `value` 能否被 `tile` 整除（`tile == 0` 时恒假，避免除零）。
fn divides(value: usize, tile: usize) -> bool {
    tile != 0 && value % tile == 0
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
/// `engine::codegen::cuda_loop_driver` 拼——设备段与驱动分属两层，机制层不生成 host C 文本。
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

/// 骨架分派：串行兜底 + 网格快路径。
fn lower(run: &Run<'_>, prepared: &Prepared, m: usize, n: usize) -> diagnostic::Result<String> {
    let op = run.op;
    match op.skeleton {
        Skeleton::Reduce => reduce(op, run, prepared, m, n),
        Skeleton::Elementwise => elementwise(op, run, prepared, m, n),
        Skeleton::Gemm => gemm(op, run, prepared, m, n),
        // 扫描沿时间轴串行递推，正文就是一条跨元素循环；网格按元素拆只会让每线程各算一整遍。
        // 收敛到单线程串行（grid = [1,1,1]），与 Reduce 共用同一条 lowering。
        Skeleton::Scan => serial(op, run, prepared, m, n),
    }
}

/// 归约骨架：声明了 `[chunk]` 的走分块归约（每线程归约一块、主机折叠部分和），否则串行兜底。
fn reduce(
    op: &OpDecl,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let Some(chunk) = op.chunk.as_ref() else {
        return serial(op, run, prepared, m, n);
    };
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Metal)?.lower(
            &prepared.source.text,
            &op.signature,
            &sample,
            &Lower::ChunkReduce {
                chunk: chunk.size,
                func: reduce_func(op)?,
            },
        )?,
        "归约",
    )
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
    let op_name = need
        .attrs
        .iter()
        .find(|attr| attr.key() == AttrKey::Op)
        .and_then(|attr| attr.values().first())
        .ok_or_else(|| Error::new(Code::Invalid, format!("`{}` 的归约需求缺 `op`", op.name)))?;
    let dt = need.dt.names().first().copied().unwrap_or("f32");
    Ok(format!("_cutile_{op_name}_{dt}"))
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

/// 串行兜底：单线程走完整条正文，缓冲原样、标量取样本值。
fn serial(
    op: &OpDecl,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Metal)?.lower(
            &prepared.source.text,
            &op.signature,
            &sample,
            &Lower::Loop(Grid { dims: Vec::new() }),
        )?,
        "串行",
    )
}

/// 循环注入网格快路径（带归约逐元素、声明了 `[grid]` 的 Gemm）。
fn looped(
    op: &OpDecl,
    run: &Run<'_>,
    prepared: &Prepared,
    grid: &Grid,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Metal)?.lower(
            &prepared.source.text,
            &op.signature,
            &sample,
            &Lower::Loop(grid.clone()),
        )?,
        "循环",
    )
}

/// 逐元素骨架：纯逐点单标量走网格；带归约/两趟（rmsnorm/softmax/attention）或非纯逐点签名
/// （embed/rope 多标量、dequant）走串行兜底——正文里的归约已沿轴遍历、或下标依赖行 gather/成对步长，
/// 网格按元素拆只会算错。声明了 `[grid]` 的走循环注入——attention 按查询行拆线程、行内软归一化留在
/// 每线程串行。
fn elementwise(
    op: &OpDecl,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    if !op.pointwise() {
        if let Some(grid) = op.grid.as_ref() {
            return looped(op, run, prepared, grid, m, n);
        }
        return serial(op, run, prepared, m, n);
    }
    let sample = (run.model.sample)(m, n);
    single(
        backend(Emit::Metal)?.lower(
            &prepared.source.text,
            &op.signature,
            &sample,
            &Lower::Pointwise,
        )?,
        "逐点",
    )
}

/// Gemm 骨架：声明了 `[grid]` 的走循环注入网格快路径（gemv 一维、gemm 二维共用一条 lowering）；
/// 未声明的走串行兜底。
fn gemm(
    op: &OpDecl,
    run: &Run<'_>,
    prepared: &Prepared,
    m: usize,
    n: usize,
) -> diagnostic::Result<String> {
    if let Some(grid) = op.grid.as_ref() {
        return looped(op, run, prepared, grid, m, n);
    }
    serial(op, run, prepared, m, n)
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
mod tests {
    use super::*;

    fn sample(m: usize, n: usize) -> Sample {
        // `check_sample` 只读 m/n：空缓冲与空标量足以驱动形状判据，不引入任何硬件或夹具。
        Sample {
            m,
            n,
            scalars: Vec::new(),
            buffers: Vec::new(),
        }
    }

    #[test]
    fn check_sample_accepts_a_shape_on_the_group_grid() {
        check_sample(4, &sample(8, 5)).expect("8 是 group 4 的正整数倍");
        check_sample(1, &sample(1, 1)).expect("group=1 时任意正 m 都在网格上");
    }

    #[test]
    fn check_sample_rejects_zero_or_off_grid_shapes() {
        for bad in [sample(0, 5), sample(5, 0), sample(7, 5)] {
            assert_eq!(
                check_sample(4, &bad).unwrap_err().code(),
                Code::Invalid,
                "形状 {}x{} 必须被拒",
                bad.m,
                bad.n
            );
        }
    }

    #[test]
    fn check_parity_rejects_length_mismatch_and_drift() {
        assert_eq!(
            check_parity(&[1.0], &[1.0, 2.0], 0.1).unwrap_err().code(),
            Code::Invalid,
            "长度不一致先于误差判定"
        );
        assert_eq!(
            check_parity(&[1.0, 2.0], &[1.0, 3.0], 0.1)
                .unwrap_err()
                .code(),
            Code::Invalid,
            "相对误差超容差"
        );
    }

    #[test]
    fn check_parity_reports_the_error_within_tolerance() {
        assert_eq!(check_parity(&[1.0], &[1.0], 0.1).expect("相等"), 0.0);
        // 样本取 1/16 网格上的数：`1.0625f32` 与 `1.0f32` 的差在 f32 里精确等于 0.0625，
        // 于是"返回的是实测误差而不是布尔"可以用 `==` 判。写成 `1.05` 会让 `|1.05f32 - 1.0|`
        // 与十进制 0.05 差 4.8e-8，任何 1e-9 级断言都会假失败——那是样本选错，不是实现有漂移。
        let err = check_parity(&[1.0625], &[1.0], 0.1).expect("6.25% 误差在 0.1 容差内");
        assert_eq!(err, 0.0625, "返回实测误差而不是只给布尔");
    }
}
