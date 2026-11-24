//! 内核程序的装配面：后端 / 形态分派 / 网格与生命周期参数的落位。
//!
//! 这一段原本与执行面同住 `execution.rs`（那次合并把 `program.rs` 折了进去），但两者是
//! 两件事：执行面回答"跑一次要多大代价、与参考对得上吗"（`Execution` / `execute`），
//! 装配面回答"这一次该生成什么源、用什么形态、哪些前置条件必须先成立"（[`Mode`] /
//! [`program`] / [`Stages`]）。装配面不认识执行结果，执行面只调用它的产物；分开之后两边
//! 都回到单文件 600 行以内，`execution::*` 的历史路径由执行面的重导出保持。

use crate::backend::Lower;
use crate::execution::{grid_dims, kind, lower_sources, resolve, single_kind};
use crate::pipeline::{Prepared, Run};
use cutile::builtin::Catalogue;
use cutile::dirs::Dirs;
use cutile::prof::Emit;
use diagnostic::{Code, Error};
use prof::decl::{Grid, OpDecl};
use prof::model::Sample;

/// 错误定位：所有形态共用这一个站点（换形态不改定位）。
const SITE: &str = "shard::execution::program";

/// 后端：厂商档案键 + 输出方言。机制层只认这两个取值，厂商实现归 `shard` 后端岛。
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
        self.take::<3>(role)
            .map(|[first, second, third]| (first, second, third))
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

/// 一个 lowering 对给定流水线/样本是否成立。无前置的形态（声明派生、循环注入）恒成立。
fn ready(lower: &Lower, run: &Run<'_>, _sample: &Sample) -> bool {
    match lower {
        Lower::Pointwise => run.op.pointwise(),
        _ => true,
    }
}
