//! 真机执行：用一份给定样本走完整条流水线并执行一次。

use crate::backend::{Launch, Lower};
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
use prof::model::Sample;

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
pub(crate) fn resolve<'a>(
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
pub(crate) fn kind(op: &OpDecl) -> diagnostic::Result<Lower> {
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
            .with("shard::execution::launch")
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
pub(crate) fn single_kind(op: &OpDecl) -> diagnostic::Result<Lower> {
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
        device: profile.meta.id.clone(),
        vendor: profile.meta.vendor.clone(),
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
    crate::device::require(emit)
}

/// 样本 + 后端查表 + 生产态 lowering 收在一处：调用方只写一个 `?`，不逐处重复「取样本、查后端、
/// 降源」三段样板，也压低各入口的 Ccn。
pub(crate) fn lower_sources(
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
        .with("shard::execution::program")
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
pub(crate) fn grid_dims(grid: &Grid, sample: &Sample) -> diagnostic::Result<Vec<usize>> {
    grid.dims
        .iter()
        .map(|(_, scalar)| dim_size(scalar, sample))
        .collect()
}

/// 一个网格标量的大小：样本没有取值、或取值不是非负 `usize`，都算声明与样本对不上。
pub(super) fn dim_size(scalar: &str, sample: &Sample) -> diagnostic::Result<usize> {
    let size = sample.scalar(scalar).ok_or_else(|| {
        Error::new(Code::Invalid, format!("网格标量 `{scalar}` 没有样本取值"))
            .with("shard::execution::program")
    })?;
    usize::try_from(size).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("网格标量 `{scalar}` = {size} 不是非负 usize"),
        )
        .with("shard::execution::program")
    })
}

#[cfg(test)]
#[path = "../tests/execution_tests.rs"]
mod tests;

// 装配面的公开路径在这里转出：`execution::{Backend, Mode, program, …}` 是既有签名。
pub use crate::program::{Backend, Mode, Stages, program, program_sample};
