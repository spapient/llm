//! 真机执行：用一份给定样本走完整条流水线并执行一次。
//!
//! 与 `verify` 的分工是**样本从哪来**：`verify` 用算子声明的确定性样本，回答"这台机器能算这条
//! 算子"；本模块用调用方给的真实数据，回答"这份数据算出来的数对不对"。两者共用同一条
//! `pipeline`，因此不存在第二套执行路径。

use crate::atom::Prim;
use crate::build;
use crate::builtin::Catalogue;
use crate::cap::Path;
use crate::decl::Skeleton;
use crate::dirs::Dirs;
use crate::emit::Fragment;
use crate::pipeline::{Prepared, Run};
use crate::prof::{Emit, Profile};
use crate::witness;
use diagnostic::{Code, Error};
use op::decl::OpDecl;
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
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let profile = catalogue
        .profile("metal")
        .ok_or_else(|| Error::new(Code::Missing, "没有 `metal` 档案"))?;
    let evidence = witness::establish(profile, dirs);
    let proof = evidence.proof(profile);
    if let Some(error) = evidence.failure {
        return Err(error.with("Metal 设备缺执行证据，不在未验证的后端上生成生产内核"));
    }
    let run = Run::new(op, profile, proof)?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let prepared = run.prepare(&path, dirs, m, n)?;
    match op.skeleton {
        Skeleton::Reduce => {
            let sample = (run.model.sample)(m, n);
            crate::harness::metal_serial_kernel(&prepared.source.text, &op.signature, &sample)
        }
        Skeleton::Elementwise => {
            // 带归约/两趟（rmsnorm/softmax/attention）或非纯逐点签名（embed/rope 多标量、dequant）
            // 走串行兜底：正文里的归约已沿轴遍历、或下标依赖行 gather/成对步长，网格按元素拆只会算错。
            // 声明了 `[grid]` 的则走循环注入——attention 按查询行拆线程、行内软归一化留在每线程串行。
            let reduces = op
                .needs
                .iter()
                .any(|need| matches!(need.prim, Prim::Reduce | Prim::Dot));
            let scalars = op
                .signature
                .params()
                .iter()
                .filter(|param| !param.ctype.is_buffer())
                .count();
            if reduces || scalars != 1 {
                if let Some(grid) = op.grid.as_ref() {
                    let sample = (run.model.sample)(m, n);
                    return crate::harness::metal_loop_kernel(
                        &prepared.source.text,
                        &op.signature,
                        &sample,
                        grid,
                    );
                }
                let sample = (run.model.sample)(m, n);
                return crate::harness::metal_serial_kernel(
                    &prepared.source.text,
                    &op.signature,
                    &sample,
                );
            }
            crate::harness::metal_grid_kernel(&prepared.source.text, &op.signature)
        }
        Skeleton::Gemm => match op.grid.as_ref() {
            // 声明了 `[grid]` 的走循环注入网格快路径（gemv 一维、gemm 二维共用一条 lowering）；未声明的
            // 走串行兜底。
            Some(grid) => {
                let sample = (run.model.sample)(m, n);
                crate::harness::metal_loop_kernel(
                    &prepared.source.text,
                    &op.signature,
                    &sample,
                    grid,
                )
            }
            None => {
                let sample = (run.model.sample)(m, n);
                crate::harness::metal_serial_kernel(&prepared.source.text, &op.signature, &sample)
            }
        },
        Skeleton::Attention => {
            // 串行是通用正确性兜底：单线程走完整条正文，缓冲原样、标量取样本值。并行化属 M3-5。
            let sample = (run.model.sample)(m, n);
            crate::harness::metal_serial_kernel(&prepared.source.text, &op.signature, &sample)
        }
        Skeleton::Scan => {
            // 扫描沿时间轴串行递推，正文就是一条跨元素循环；网格按元素拆只会让每线程各算一整遍。
            // 收敛到单线程串行（grid = [1,1,1]），与 Reduce 共用同一条 lowering。
            let sample = (run.model.sample)(m, n);
            crate::harness::metal_serial_kernel(&prepared.source.text, &op.signature, &sample)
        }
    }
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

#[cfg(test)]
#[path = "../tests/execution_tests.rs"]
mod tests;
