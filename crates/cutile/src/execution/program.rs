//! 生产态设备内核源：一条算子 → 一份后端可编译的设备源（外加派发要用的附加数据）。
//!
//! 与 [`super::execute`] 的分工：那里把设备源编译执行并当场对拍，回答"这条算子在这台机器上算得
//! 对不对"；这里只到"源生成"为止，回答"这条算子该生成哪种内核"，因此本模块不做对拍。
//!
//! 入口只有 [`program`]：`(后端, 形态, 形状)` 决定降哪一种内核。17 个按骨架命名的
//! `metal_*_program` / `cuda_*_program` 曾是同一段 `resolve → 前置判断 → lower → 取段` 的复制，
//! 厂商名也因此漏进机制层的公开面（`testkit::boundary` 的厂商名基线）。现在厂商只是 [`Backend`]
//! 的两个取值，语义差异全落在 [`Mode`] 的数据与它自带的前置条件上。
//!
//! 前置条件（块几何、签名、形状整除……）的契约写在 [`Lower`] 各变体自己的文档里，这里只判断
//! "成立与否"：不成立就回落 [`Mode::declared`]，不重复第二遍契约。

use super::{grid_dims, kind, lower_sources, resolve, single_kind};
use crate::backend::{Lower, TensorGeometry};
use crate::builtin::Catalogue;
use crate::decl::Skeleton;
use crate::dirs::Dirs;
use crate::pipeline::{Prepared, Run};
use crate::prof::Emit;
use diagnostic::{Code, Error};
use op::decl::{Grid, OpDecl};
use op::model::Sample;
use op::sig::CType;

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
///
/// 段数随形态变（归一化两段、softmax 三段、循环注入附带网格各维），因此不按位置返回元组：
/// 调用方按角色取名，多段与附加数据不再靠"第几个返回值"约定。
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
    pub fn one(mut self, role: &str) -> diagnostic::Result<String> {
        pop_one(&mut self.sources, role)
    }

    /// 两段设备源，按声明顺序还原。
    ///
    /// # Errors
    ///
    /// 后端少给了设备源（违约）。
    pub fn two(mut self, role: &str) -> diagnostic::Result<(String, String)> {
        let second = pop_one(&mut self.sources, role)?;
        let first = pop_one(&mut self.sources, role)?;
        Ok((first, second))
    }

    /// 三段设备源，按声明顺序还原。
    ///
    /// # Errors
    ///
    /// 后端少给了设备源（违约）。
    pub fn three(mut self, role: &str) -> diagnostic::Result<(String, String, String)> {
        let third = pop_one(&mut self.sources, role)?;
        let second = pop_one(&mut self.sources, role)?;
        let first = pop_one(&mut self.sources, role)?;
        Ok((first, second, third))
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

/// 内核形态：目标 lowering 的来源 + 可选的形状前置条件。
///
/// 构造器名说的是**形态**（单源 / 多段 / 逐点 / 分块 / 张量核……），不说厂商：同一形态在两个
/// 后端上只有方言不同，是同一份语义词。
#[derive(Debug, Clone)]
pub struct Mode {
    plan: Plan,
    gate: Option<Gate>,
    strict: bool,
}

/// 目标 lowering 的来源。
#[derive(Debug, Clone)]
enum Plan {
    /// 按声明派生：`multi` 取多段族（分块归约/归一化/扫描），否则单源（分块族退串行）。
    Declared {
        /// 是否保留多段形态。
        multi: bool,
    },
    /// 显式形态：几何由调用方给。
    Exact(Lower),
    /// 循环注入：`[grid]` 必须声明，网格各维随 [`Stages::dims`] 回给主机。
    Grid,
}

/// 形状前置条件：`(流水线, m, n, 目标 lowering) -> 是否成立`。
type Gate = fn(&Run<'_>, usize, usize, &Lower) -> bool;

impl Mode {
    /// 单源、按声明：分块族退串行（一源一内核）。
    pub fn declared() -> Self {
        Self::plain(Plan::Declared { multi: false })
    }

    /// 多段、按声明：分块归约/归一化/扫描按 `[chunk]` 与 `[needs]` 切段。
    pub fn segmented() -> Self {
        Self::plain(Plan::Declared { multi: true })
    }

    /// 逐点网格。非纯逐点形直接报错，不回落（逐点网格覆盖不了别的正文）。
    pub fn pointwise() -> Self {
        Self::gated(Lower::Pointwise, pointwise_ready, true)
    }

    /// 循环注入网格：`[grid]` 必须声明，网格各维随 [`Stages::dims`] 回给主机。
    pub fn loop_injected() -> Self {
        Self {
            plan: Plan::Grid,
            gate: None,
            strict: false,
        }
    }

    /// WMMA 张量核。
    pub fn wmma() -> Self {
        Self::plain(Plan::Exact(Lower::WmmaGemm))
    }

    /// 分块 GEMM：形状不整除时回落 [`Self::declared`]。
    pub fn tiled(tile: usize, per_thread: usize) -> Self {
        Self::gated(Lower::TiledGemm { tile, per_thread }, tile_ready, false)
    }

    /// 分块 attention：一个 threadgroup 负责一个查询行。
    pub fn tiled_attention(threads: usize) -> Self {
        Self::gated(Lower::TiledAttention { threads }, attention_ready, false)
    }

    /// simdgroup 分片 GEMM。
    pub fn simdgroup(row_frags: usize, col_frags: usize) -> Self {
        Self::gated(
            Lower::SimdGroupGemm {
                row_frags,
                col_frags,
            },
            simdgroup_ready,
            false,
        )
    }

    /// 张量核 GEMM。
    pub fn tensor(geometry: TensorGeometry) -> Self {
        Self::gated(Lower::TensorGemm { geometry }, tensor_ready, false)
    }

    /// int8 分块 GEMM（`i8` 输入、`i32` 累加、`f32` 输出）。
    pub fn int8(tile: usize) -> Self {
        Self::gated(Lower::Int8Gemm { tile }, int8_ready, false)
    }

    /// flash attention：K/V 按块流式扫描，行内不保留完整分数向量。
    pub fn flash(threads: usize, block: usize) -> Self {
        Self::gated(Lower::FlashAttention { threads, block }, flash_ready, false)
    }

    /// 无前置条件的形态。
    fn plain(plan: Plan) -> Self {
        Self {
            plan,
            gate: None,
            strict: false,
        }
    }

    /// 带前置条件的显式形态。
    fn gated(lower: Lower, gate: Gate, strict: bool) -> Self {
        Self {
            plan: Plan::Exact(lower),
            gate: Some(gate),
            strict,
        }
    }

    /// 形态 → 目标 lowering。声明形态现场派生，显式形态原样给出。
    fn lower(&self, op: &OpDecl, op_name: &str) -> diagnostic::Result<Lower> {
        match &self.plan {
            Plan::Declared { multi: false } => single_kind(op),
            Plan::Declared { multi: true } => kind(op),
            Plan::Exact(lower) => Ok(lower.clone()),
            Plan::Grid => Ok(Lower::Loop(grid_of(op, op_name)?.clone())),
        }
    }

    /// 前置条件不成立时回落声明形态；`strict` 形态直接报错。
    fn pick(&self, lower: Lower, run: &Run<'_>, m: usize, n: usize) -> diagnostic::Result<Lower> {
        let Some(gate) = self.gate else {
            return Ok(lower);
        };
        if gate(run, m, n, &lower) {
            return Ok(lower);
        }
        if self.strict {
            return Err(rejected(&lower, run.op));
        }
        single_kind(run.op)
    }

    /// 循环注入的网格各维：其它形态不给（派发几何由 [`super::launch`] 现算）。
    fn grid_dims(&self, lower: &Lower, sample: &Sample) -> diagnostic::Result<Option<Vec<usize>>> {
        let Plan::Grid = self.plan else {
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
    assemble(&mode, &run, &prepared, op_name, backend.emit(), m, n)
}

/// 定形态 → 判前置 → 降源。
fn assemble(
    mode: &Mode,
    run: &Run<'_>,
    prepared: &Prepared,
    op_name: &str,
    emit: Emit,
    m: usize,
    n: usize,
) -> diagnostic::Result<Stages> {
    let lower = mode.lower(run.op, op_name)?;
    let chosen = mode.pick(lower, run, m, n)?;
    stage(mode, &chosen, run, prepared, emit, m, n)
}

/// 目标 lowering + 声明 → 设备源与附加数据。
fn stage(
    mode: &Mode,
    lower: &Lower,
    run: &Run<'_>,
    prepared: &Prepared,
    emit: Emit,
    m: usize,
    n: usize,
) -> diagnostic::Result<Stages> {
    Ok(Stages {
        sources: lower_sources(emit, lower, run, prepared, m, n)?,
        decay: decay_of(lower),
        dims: mode.grid_dims(lower, &(run.model.sample)(m, n))?,
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

/// 纯逐点：正文不吃跨元素依赖，一元素一线程成立。
fn pointwise_ready(run: &Run<'_>, _m: usize, _n: usize, _lower: &Lower) -> bool {
    run.op.pointwise()
}

/// 分块 GEMM：可分块、块几何合法、`M/N/K` 都能被 `tile` 整除。
fn tile_ready(run: &Run<'_>, m: usize, n: usize, lower: &Lower) -> bool {
    let &Lower::TiledGemm { tile, per_thread } = lower else {
        return false;
    };
    is_gemm(run, tile) && valid_block(tile, per_thread) && shapes(run, m, n, tile)
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
fn shapes(run: &Run<'_>, m: usize, n: usize, tile: usize) -> bool {
    let k = (run.model.sample)(m, n).scalar("K").unwrap_or(0);
    k > 0 && divides(m, tile) && divides(n, tile) && divides(k as usize, tile)
}

/// `value` 能否被 `tile` 整除（`tile == 0` 时恒假，避免除零）。
fn divides(value: usize, tile: usize) -> bool {
    tile != 0 && value.is_multiple_of(tile)
}

/// simdgroup GEMM：片段数在 `1..=4`、`K % 8 == 0`、`M/N` 分别被 `8·片段数` 整除。
fn simdgroup_ready(run: &Run<'_>, m: usize, n: usize, lower: &Lower) -> bool {
    let &Lower::SimdGroupGemm {
        row_frags,
        col_frags,
    } = lower
    else {
        return false;
    };
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

/// 张量核 GEMM：Gemm 骨架、几何参数成闭集、`M` 被 `tile_m` 整除。
///
/// `M` 的整除性是硬约束——`matmul2d` 按整块读 A，尾块会读越界；N 维不必对齐（装载零填、
/// 写回裁边），K 维不必对齐（末块走 `dynamic_extent`）。线程组内存只随 `tile_n` 增长。
fn tensor_ready(run: &Run<'_>, m: usize, n: usize, lower: &Lower) -> bool {
    let &Lower::TensorGemm { geometry } = lower else {
        return false;
    };
    run.op.skeleton == Skeleton::Gemm
        && n > 0
        && divides(m, geometry.tile_m)
        && geometry_ready(geometry)
}

/// 张量核几何的三档前置都满足：跨度档位、规约块/simdgroup、线程组内存占用。
fn geometry_ready(geometry: TensorGeometry) -> bool {
    geometry.spans_ok() && geometry.block_ok() && geometry.fits_threadgroup_memory()
}

/// int8 GEMM：块边长合法、Gemm 骨架、int8 签名、`M/N/K` 都能被 `tile` 整除。
fn int8_ready(run: &Run<'_>, m: usize, n: usize, lower: &Lower) -> bool {
    let &Lower::Int8Gemm { tile } = lower else {
        return false;
    };
    (2..=32).contains(&tile)
        && run.op.skeleton == Skeleton::Gemm
        && int8_signature(run)
        && shapes(run, m, n, tile)
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

/// flash attention：线程/块几何合法、elementwise 骨架、注意力签名与形状。
fn flash_ready(run: &Run<'_>, s: usize, d: usize, lower: &Lower) -> bool {
    let &Lower::FlashAttention { threads, block } = lower else {
        return false;
    };
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
fn attention_ready(run: &Run<'_>, s: usize, d: usize, lower: &Lower) -> bool {
    let &Lower::TiledAttention { threads } = lower else {
        return false;
    };
    tileable_threads(threads)
        && run.op.skeleton == Skeleton::Elementwise
        && attention_shape(run, s, d)
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
