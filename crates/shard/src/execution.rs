//! 真机执行：用一份给定样本走完整条流水线并执行一次。

use crate::backend::{Launch, Lower, TensorGeometry};
use crate::build;
use crate::pipeline::{Prepared, Run};
use crate::witness;
use cutile::atom::{AttrKey, Prim};
use cutile::builtin::Catalogue;
use cutile::cap::{Path, Proof};
use cutile::decl::Skeleton;
use cutile::dirs::Dirs;
use cutile::emit::Fragment;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error};
use prof::decl::{Grid, OpDecl};
use nn::model::Sample;
use nn::sig::CType;

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
        .find(|param| !param.ctype.is_buf())
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

/// 单源形态：分块归一化 / 分块扫描是多段源，单源入口没有主机折叠的位置，退串行（多段形态走
/// [`program::Mode::segmented`]）。
fn single_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    match kind(op)? {
        Lower::ChunkRmsnorm { .. } | Lower::ChunkSoftmax { .. } | Lower::ChunkScan { .. } => {
            Ok(serial_kind())
        }
        kind => Ok(kind),
    }
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
    let tolerance = cutile::reference::tolerance(run.model.dtype);
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
    let max_rel_err = cutile::reference::max_rel_err(bits, expected);
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
    sample: &Sample,
) -> diagnostic::Result<Vec<String>> {
    backend(emit)?.lower(&prepared.source.text, &run.op.signature, sample, kind)
}

/// 声明里必带的 `[chunk]` 块大小；缺声明时报 [`Code::Unsupported`]（串行兜底另有路径，这里只做分块）。
fn chunk_size(op: &OpDecl) -> diagnostic::Result<usize> {
    let chunk = op.chunk.as_ref().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("`{}` 没有声明 `[chunk]`", op.name),
        )
        .with("cutile::execution::program")
    })?;
    Ok(chunk.size)
}

/// 分块归约的 [`Lower`] 描述：块大小来自 `[chunk]`、软件归约原语名来自 `[needs] reduce`。
fn reduce_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    let chunk = chunk_size(op)?;
    let func = reduce_func(op)?;
    Ok(Lower::ChunkReduce { chunk, func })
}

/// 分块扫描的 [`Lower`] 描述：块大小来自 `[chunk]`、衰减 `a` 来自 `[consts]`。
fn scan_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
    let chunk = chunk_size(op)?;
    let decay = scan_decay(op)?;
    Ok(Lower::ChunkScan { chunk, decay })
}

/// 网格每一维的大小：把 `[grid]` 里的标量名按样本求值成 `usize`。
pub(super) fn grid_dims(grid: &Grid, sample: &Sample) -> diagnostic::Result<Vec<usize>> {
    grid.dims
        .iter()
        .map(|(_, scalar)| dim_size(scalar, sample))
        .collect()
}

/// 一个网格标量的大小：样本没有取值、或取值不是非负 `usize`，都算声明与样本对不上。
pub(super) fn dim_size(scalar: &str, sample: &Sample) -> diagnostic::Result<usize> {
    let size = sample.scalar(scalar).ok_or_else(|| {
        Error::new(Code::Invalid, format!("网格标量 `{scalar}` 没有样本取值"))
            .with("cutile::execution::program")
    })?;
    usize::try_from(size).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("网格标量 `{scalar}` = {size} 不是非负 usize"),
        )
        .with("cutile::execution::program")
    })
}

#[cfg(test)]
#[path = "../tests/execution_tests.rs"]
mod tests;
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
fn ready(lower: &Lower, run: &Run<'_>, sample: &Sample) -> bool {
    match lower {
        Lower::Pointwise => run.op.pointwise(),
        Lower::TiledGemm { tile, per_thread } => {
            is_gemm(run, *tile) && valid_block(*tile, *per_thread) && shapes(sample, *tile)
        }
        Lower::TiledAttention { threads } => {
            tileable_threads(*threads)
                && run.op.skeleton == Skeleton::Elementwise
                && attention_shape(run, sample)
        }
        Lower::SimdGroupGemm {
            row_frags,
            col_frags,
        } => {
            fragment_span(*row_frags, *col_frags)
                && fragment_shape(run, sample, *row_frags, *col_frags)
        }
        Lower::TensorGemm { geometry } => {
            run.op.skeleton == Skeleton::Gemm
                && sample.n > 0
                && divides(sample.m, geometry.tile_m)
                && geometry_ready(*geometry)
        }
        Lower::FlashAttention { threads, block } => {
            tileable_threads(*threads)
                && batching_block(*threads, *block)
                && run.op.skeleton == Skeleton::Elementwise
                && attention_shape(run, sample)
        }
        Lower::Int8Gemm { tile } => {
            (2..=32).contains(tile)
                && run.op.skeleton == Skeleton::Gemm
                && int8_signature(run)
                && shapes(sample, *tile)
        }
        _ => true,
    }
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
