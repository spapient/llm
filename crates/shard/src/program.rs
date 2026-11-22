//! 内核程序的装配面：后端 / 形态分派 / 网格与生命周期参数的落位。
//!
//! 这一段原本与执行面同住 `execution.rs`（那次合并把 `program.rs` 折了进去），但两者是
//! 两件事：执行面回答"跑一次要多大代价、与参考对得上吗"（`Execution` / `execute`），
//! 装配面回答"这一次该生成什么源、用什么形态、哪些前置条件必须先成立"（[`Mode`] /
//! [`program`] / [`Stages`]）。装配面不认识执行结果，执行面只调用它的产物；分开之后两边
//! 都回到单文件 600 行以内，`execution::*` 的历史路径由执行面的重导出保持。

use crate::backend::{Lower, TensorGeometry};
use crate::execution::{grid_dims, kind, lower_sources, resolve, single_kind};
use crate::pipeline::{Prepared, Run};
use cutile::builtin::Catalogue;
use cutile::dirs::Dirs;
use cutile::decl::Skeleton;
use cutile::prof::Emit;
use diagnostic::{Code, Error};
use prof::decl::{Grid, OpDecl};
use nn::model::Sample;
use nn::sig::CType;

/// 错误定位：所有形态共用这一个站点（换形态不改定位）。
const SITE: &str = "cutile::execution::program";

/// 后端：厂商档案键 + 输出方言。机制层只认这两个取值，厂商实现归 `device`（L6）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Metal（`metal` 档案，MSL）。
    Metal,
    /// CUDA（`nvidia` 档案，CUDA C）。
    Cuda,
}

impl Backend {
    /// 厂商档案键：`catalogue.profile` 的查找名。
    const fn tag(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "nvidia",
        }
    }

    /// 设备源方言。
    const fn emit(self) -> Emit {
        match self {
            Self::Metal => Emit::Metal,
            Self::Cuda => Emit::Cuda,
        }
    }
}

/// 一次形态产出的设备源 + 派发附加数据。
#[derive(Debug, Clone, PartialEq)]
pub struct Stages {
    sources: Vec<String>,
    decay: Option<f32>,
    dims: Option<Vec<usize>>,
}

impl Stages {
    /// 单段设备源。
    ///
    /// # Errors
    ///
    /// 后端少给了设备源（违约）。
    pub fn one(self, role: &str) -> diagnostic::Result<String> {
        self.take::<1>(role).map(|[one]| one)
    }

    /// 两段设备源，按声明顺序还原。
    ///
    /// # Errors
    ///
    /// 后端少给了设备源（违约）。
    pub fn two(self, role: &str) -> diagnostic::Result<(String, String)> {
        self.take::<2>(role).map(|[first, second]| (first, second))
    }

    /// 三段设备源，按声明顺序还原。
    ///
    /// # Errors
    ///
    /// 后端少给了设备源（违约）。
    pub fn three(self, role: &str) -> diagnostic::Result<(String, String, String)> {
        self.take::<3>(role).map(|[first, second, third]| (first, second, third))
    }

    /// 按声明顺序还原 `N` 段设备源：`pop_one` 出栈是逆序，从数组尾往回填。
    fn take<const N: usize>(mut self, role: &str) -> diagnostic::Result<[String; N]> {
        let mut out = std::array::from_fn(|_| String::new());
        for slot in out.iter_mut().rev() {
            *slot = pop_one(&mut self.sources, role)?;
        }
        Ok(out)
    }

    /// 扫描衰减 `a`：只有分块扫描的 lowering 带它，供主机在两次派发间折叠携带前缀。
    pub const fn decay(&self) -> Option<f32> {
        self.decay
    }

    /// 循环注入的网格各维：`dims[i]` 是 `dim3` 的 x/y/z；其它形态为 `None`。
    pub fn dims(&self) -> Option<&[usize]> {
        self.dims.as_deref()
    }
}

/// 取一段设备源；缺段是后端违约，机制层不猜。
fn pop_one(sources: &mut Vec<String>, role: &str) -> diagnostic::Result<String> {
    sources
        .pop()
        .ok_or_else(|| Error::new(Code::Internal, format!("{role} lowering 少一段设备源")))
}

/// 内核形态：目标 lowering 的来源 + 是否严格要求该形态。
#[derive(Debug, Clone)]
pub struct Mode {
    target: Target,
    strict: bool,
}

/// 目标 lowering 的来源。
#[derive(Debug, Clone)]
enum Target {
    /// 按声明派生：`multi` 取多段族（分块归约/归一化/扫描），否则单源（分块族退串行）。
    Declared { multi: bool },
    /// 显式形态：几何由调用方给。
    Exact(Lower),
    /// 循环注入：`[grid]` 必须声明，网格各维随 [`Stages::dims`] 回给主机。
    Grid,
}

impl Mode {
    /// 单源、按声明：分块族退串行（一源一内核）。
    pub fn declared() -> Self {
        Self::declared_with(false)
    }

    /// 多段、按声明：分块归约/归一化/扫描按 `[chunk]` 与 `[needs]` 切段。
    pub fn segmented() -> Self {
        Self::declared_with(true)
    }

    /// 逐点网格。非纯逐点形直接报错，不回落（逐点网格覆盖不了别的正文）。
    pub fn pointwise() -> Self {
        Self::exact(Lower::Pointwise, true)
    }

    /// 循环注入网格：`[grid]` 必须声明，网格各维随 [`Stages::dims`] 回给主机。
    pub fn loop_injected() -> Self {
        Self {
            target: Target::Grid,
            strict: false,
        }
    }

    /// WMMA 张量核。
    pub fn wmma() -> Self {
        Self::exact(Lower::WmmaGemm, false)
    }

    /// 分块 GEMM：形状不整除时回落 [`Self::declared`]。
    pub fn tiled(tile: usize, per_thread: usize) -> Self {
        Self::exact(Lower::TiledGemm { tile, per_thread }, false)
    }

    /// 分块 attention：一个 threadgroup 负责一个查询行。
    pub fn tiled_attention(threads: usize) -> Self {
        Self::exact(Lower::TiledAttention { threads }, false)
    }

    /// simdgroup 分片 GEMM。
    pub fn simdgroup(row_frags: usize, col_frags: usize) -> Self {
        Self::exact(
            Lower::SimdGroupGemm {
                row_frags,
                col_frags,
            },
            false,
        )
    }

    /// 张量核 GEMM。
    pub fn tensor(geometry: TensorGeometry) -> Self {
        Self::exact(Lower::TensorGemm { geometry }, false)
    }

    /// int8 分块 GEMM（`i8` 输入、`i32` 累加、`f32` 输出）。
    pub fn int8(tile: usize) -> Self {
        Self::exact(Lower::Int8Gemm { tile }, false)
    }

    /// flash attention：K/V 按块流式扫描，行内不保留完整分数向量。
    pub fn flash(threads: usize, block: usize) -> Self {
        Self::exact(Lower::FlashAttention { threads, block }, false)
    }

    fn declared_with(multi: bool) -> Self {
        Self {
            target: Target::Declared { multi },
            strict: false,
        }
    }

    fn exact(lower: Lower, strict: bool) -> Self {
        Self {
            target: Target::Exact(lower),
            strict,
        }
    }

    /// 形态 → 目标 lowering。声明形态现场派生，显式形态原样给出。
    fn lower(&self, op: &OpDecl, op_name: &str) -> diagnostic::Result<Lower> {
        match &self.target {
            Target::Declared { multi: false } => single_kind(op),
            Target::Declared { multi: true } => kind(op),
            Target::Exact(lower) => Ok(lower.clone()),
            Target::Grid => Ok(Lower::Loop(grid_of(op, op_name)?.clone())),
        }
    }

    /// 前置条件不成立时回落声明形态；`strict` 形态直接报错。
    fn pick(&self, lower: Lower, run: &Run<'_>, sample: &Sample) -> diagnostic::Result<Lower> {
        if ready(&lower, run, sample) {
            return Ok(lower);
        }
        if self.strict {
            return Err(rejected(&lower, run.op));
        }
        single_kind(run.op)
    }

    /// 循环注入的网格各维：其它形态不给（派发几何由 [`super::launch`] 现算）。
    fn grid_dims(&self, lower: &Lower, sample: &Sample) -> diagnostic::Result<Option<Vec<usize>>> {
        let Target::Grid = self.target else {
            return Ok(None);
        };
        let Lower::Loop(grid) = lower else {
            return Ok(None);
        };
        Ok(Some(grid_dims(grid, sample)?))
    }
}

/// 生产态设备源：解析算子/档案/证据 → 定形态 → 降源，返回设备源与派发附加数据。
///
/// # Errors
///
/// 算子或档案缺失、设备缺执行证据、协商失败、形态的前置条件不成立且不回落、或后端降源失败。
pub fn program(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    backend: Backend,
    mode: Mode,
    m: usize,
    n: usize,
) -> diagnostic::Result<Stages> {
    let (run, prepared) = resolve(catalogue, dirs, op_name, backend.tag(), m, n)?;
    let sample = run.sample(m, n);
    assemble(&mode, &run, &prepared, op_name, backend.emit(), &sample)
}

/// 生产态设备源，形状由调用方给定：`sample` 里的 `m` / `n` 与标量（如 GEMM 的收缩轴 `K`）
/// 一律按它，不再由模型的采样函数推。
///
/// 与 [`program`] 的差别只在形状来源：模型的采样函数只吃 `(m, n)`，收缩轴这类维度定不下来，
/// 形状前提（尾块、不整除）就无从验证。这里让调用方连同标量一起给出完整样本。
///
/// # Errors
///
/// 与 [`program`] 相同。
pub fn program_sample(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    backend: Backend,
    mode: Mode,
    sample: &Sample,
) -> diagnostic::Result<Stages> {
    let (mut run, prepared) = resolve(catalogue, dirs, op_name, backend.tag(), sample.m, sample.n)?;
    run.shape = Some(sample.clone());
    assemble(&mode, &run, &prepared, op_name, backend.emit(), sample)
}

/// 定形态 → 判前置 → 降源。
fn assemble(
    mode: &Mode,
    run: &Run<'_>,
    prepared: &Prepared,
    op_name: &str,
    emit: Emit,
    sample: &Sample,
) -> diagnostic::Result<Stages> {
    let lower = mode.lower(run.op, op_name)?;
    let chosen = mode.pick(lower, run, sample)?;
    stage(mode, &chosen, run, prepared, emit, sample)
}

/// 目标 lowering + 声明 → 设备源与附加数据。
fn stage(
    mode: &Mode,
    lower: &Lower,
    run: &Run<'_>,
    prepared: &Prepared,
    emit: Emit,
    sample: &Sample,
) -> diagnostic::Result<Stages> {
    Ok(Stages {
        sources: lower_sources(emit, lower, run, prepared, sample)?,
        decay: decay_of(lower),
        dims: mode.grid_dims(lower, sample)?,
    })
}

/// 分块扫描的衰减 `a`：`ChunkScan` 的 lowering 自带，不必再读一遍 `[consts]`。
fn decay_of(lower: &Lower) -> Option<f32> {
    match lower {
        Lower::ChunkScan { decay, .. } => Some(*decay),
        _ => None,
    }
}

/// 不回落形态的前置条件不成立：报出算子与目标 lowering。
fn rejected(lower: &Lower, op: &OpDecl) -> Error {
    let message = match lower {
        Lower::Pointwise => format!(
            "`{}` 不是纯逐点形，逐点网格未覆盖（循环注入/串行兜底是别的形态）",
            op.name
        ),
        _ => format!("`{}` 不满足 `{lower:?}` 形态的前置条件", op.name),
    };
    Error::new(Code::Unsupported, message).with(SITE)
}

/// 声明了 `[grid]` 的循环注入：没声明就没有网格可发射。
fn grid_of<'a>(op: &'a OpDecl, op_name: &str) -> diagnostic::Result<&'a Grid> {
    op.grid.as_ref().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("`{op_name}` 没有声明 `[grid]`，循环注入不适用"),
        )
        .with(SITE)
    })
}

// ── 形态前置条件 ────────────────────────────────────────────────────────
// 判据逐条搬自原各入口；契约本身写在 `Lower` 各变体的文档里，这里只说成立与否。

/// 一个 lowering 对给定流水线/样本是否成立。无前置的形态（声明派生、循环注入、WMMA 等）恒成立。
/// 形态前置条件：先按"哪一类内核"分流，再落到这一形态自己那条规则。
///
/// 一条规则只写一次、只写在一个函数里（[`tiled_gemm_ready`] 等），于是这张分派表本身
/// 没有一处 `&&`：加一种形态只需要加一行 arm 和一个规则函数。
fn ready(lower: &Lower, run: &Run<'_>, sample: &Sample) -> bool {
    match lower {
        Lower::Pointwise => run.op.pointwise(),
        gemm @ (Lower::TiledGemm { .. }
        | Lower::SimdGroupGemm { .. }
        | Lower::TensorGemm { .. }
        | Lower::Int8Gemm { .. }) => gemm_ready(gemm, run, sample),
        attention @ (Lower::TiledAttention { .. } | Lower::FlashAttention { .. }) => {
            attention_ready(attention, run, sample)
        }
        _ => true,
    }
}

/// GEMM 家族：四张形态各有自己的 tile 约束，判别只在这里分一次。
fn gemm_ready(lower: &Lower, run: &Run<'_>, sample: &Sample) -> bool {
    match lower {
        Lower::TiledGemm { tile, per_thread } => tiled_gemm_ready(*tile, *per_thread, run, sample),
        Lower::SimdGroupGemm {
            row_frags,
            col_frags,
        } => simd_gemm_ready(*row_frags, *col_frags, run, sample),
        Lower::TensorGemm { geometry } => tensor_gemm_ready(*geometry, run, sample),
        Lower::Int8Gemm { tile } => int8_gemm_ready(*tile, run, sample),
        _ => false,
    }
}

/// 注意力家族：分块注意力与 flash 共用形状判据，只差"要不要块内批处理"。
fn attention_ready(lower: &Lower, run: &Run<'_>, sample: &Sample) -> bool {
    match lower {
        Lower::TiledAttention { threads } => tiled_attention_ready(*threads, run, sample),
        Lower::FlashAttention { threads, block } => {
            flash_attention_ready(*threads, *block, run, sample)
        }
        _ => false,
    }
}

/// 分块 GEMM：骨架是 GEMM、tile 组合合法、形状能整块切开。
fn tiled_gemm_ready(tile: usize, per_thread: usize, run: &Run<'_>, sample: &Sample) -> bool {
    is_gemm(run, tile) && valid_block(tile, per_thread) && shapes(sample, tile)
}

/// SIMT 组 GEMM：片段跨度不超组宽，且片段形状与样本对得上。
fn simd_gemm_ready(row_frags: usize, col_frags: usize, run: &Run<'_>, sample: &Sample) -> bool {
    fragment_span(row_frags, col_frags) && fragment_shape(run, sample, row_frags, col_frags)
}

/// 张量核 GEMM：骨架、非空归约维、两维都能按 tile 整除、几何自洽。
fn tensor_gemm_ready(geometry: TensorGeometry, run: &Run<'_>, sample: &Sample) -> bool {
    run.op.skeleton == Skeleton::Gemm
        && sample.n > 0
        && divides(sample.m, geometry.tile_m)
        && geometry_ready(geometry)
}

/// int8 GEMM：tile 在 2..=32、骨架是 GEMM、签名是 int8 对、形状能整块切开。
fn int8_gemm_ready(tile: usize, run: &Run<'_>, sample: &Sample) -> bool {
    (2..=32).contains(&tile) && run.op.skeleton == Skeleton::Gemm && int8_signature(run)
        && shapes(sample, tile)
}

/// 分块注意力：线程可整块、骨架是逐点、形状满足 `d ≤ s` 的注意力几何。
fn tiled_attention_ready(threads: usize, run: &Run<'_>, sample: &Sample) -> bool {
    tileable_threads(threads) && run.op.skeleton == Skeleton::Elementwise && attention_shape(run, sample)
}

/// flash 注意力：在分块注意力之上再要求块内批处理可行。
fn flash_attention_ready(
    threads: usize,
    block: usize,
    run: &Run<'_>,
    sample: &Sample,
) -> bool {
    tiled_attention_ready(threads, run, sample) && batching_block(threads, block)
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
fn shapes(sample: &Sample, tile: usize) -> bool {
    let k = sample.scalar("K").unwrap_or(0);
    k > 0 && divides(sample.m, tile) && divides(sample.n, tile) && divides(k as usize, tile)
}

/// `value` 能否被 `tile` 整除（`tile == 0` 时恒假，避免除零）。
fn divides(value: usize, tile: usize) -> bool {
    tile != 0 && value.is_multiple_of(tile)
}

/// 每个 simdgroup 沿 M/N 的输出片段数限定在 `1..=4`（`8×8` 片段，最多 `32×32` 输出）。
fn fragment_span(row_frags: usize, col_frags: usize) -> bool {
    (1..=4).contains(&row_frags) && (1..=4).contains(&col_frags)
}

/// Gemm 骨架 + K 轴能被 8 整除 + M/N 分别能被 `8·片段数` 整除。
fn fragment_shape(run: &Run<'_>, sample: &Sample, row_frags: usize, col_frags: usize) -> bool {
    if run.op.skeleton != Skeleton::Gemm {
        return false;
    }
    let k = sample.scalar("K").unwrap_or(0);
    k > 0
        && (k as usize).is_multiple_of(8)
        && divides(sample.m, 8 * row_frags)
        && divides(sample.n, 8 * col_frags)
}

/// 张量核几何的三档前置都满足：跨度档位、规约块/simdgroup、线程组内存占用。
fn geometry_ready(geometry: TensorGeometry) -> bool {
    geometry.spans_ok() && geometry.block_ok() && geometry.fits_threadgroup_memory()
}

/// 签名是不是 int8 GEMM：`(u8 *a, u8 *b, float *y, int M, int N, int K)`。
fn int8_signature(run: &Run<'_>) -> bool {
    let params = run.op.signature.params();
    matches!(params, [a, b, y, m, n, k]
        if a.ctype == CType::U8
            && b.ctype == CType::U8
            && y.ctype == CType::F32
            && [m, n, k].iter().all(|param| param.ctype == CType::Int))
}

/// 键块大小：`[32, 1024]` 内且是线程数的整数倍，保证每轮线程负载均匀。
fn batching_block(threads: usize, block: usize) -> bool {
    (32..=1024).contains(&block) && block.is_multiple_of(threads)
}

/// 线程数必须是 [32, 1024] 内的 2 的幂（树归约与线程组上限）。
fn tileable_threads(threads: usize) -> bool {
    threads.is_power_of_two() && (32..=1024).contains(&threads)
}

/// 注意力签名与形状：4 缓冲（q/k/v/y）+ 2 标量（S/D），且形状落在共享内存/头宽上限内。
fn attention_shape(run: &Run<'_>, sample: &Sample) -> bool {
    let params = run.op.signature.params();
    let bufs = params.iter().filter(|p| p.ctype.is_buf()).count();
    let scalars = params.iter().filter(|p| !p.ctype.is_buf()).count();
    bufs == 4 && scalars == 2 && in_bounds(sample.m, sample.n)
}

/// `S·4` 字节的分数块不超过 32 KiB 线程组内存，头宽不超过模板的 64。
fn in_bounds(s: usize, d: usize) -> bool {
    s > 0 && d > 0 && s <= 8192 && d <= 64
}

