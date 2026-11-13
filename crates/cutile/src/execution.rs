//! 真机执行：用一份给定样本走完整条流水线并执行一次。

use crate::build;
use crate::builtin::Catalogue;
use crate::emit::Fragment;
use crate::pipeline::Run;
use crate::witness;
use diagnostic::{Code, Error};
use llm_config::Settings;
use op::model::Sample;

/// 一次真机执行：调用方给的样本 → 编译 → 跑一次 → 与标量参考对拍。
///
/// `verify` 用的是算子自己的确定性样本，它证明的是"这台机器能算这条算子"；本结构证明的是
/// "**这份数据**在这台机器上算出来的数是对的"。两者共用同一条流水线，差别只在样本从哪来——
/// 因此不需要第二套执行路径。
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
/// 与 [`crate::verify::verify`] 的三点区别，都是刻意的：
///
/// - 样本由调用方给（真实权重），因此不做形状矩阵，只跑这一份；
/// - 执行前仍要求设备有见证证据：没有证据的后端算出来的数不是结果，只是现象；
/// - 对拍不通过是错误（[`Code::Invalid`]），不是"环境问题"——它说明产物与它自己声明的语义不符。
///
/// # Errors
///
/// 算子没有声明或没有语义表、设备不可用、设备没有执行证据、协商失败、形状与算子粒度不符、
/// 签名与样本不一致、编译/执行失败，或对拍超出容差时返回说明。
pub fn execute(
    catalogue: &Catalogue,
    settings: &Settings,
    op_name: &str,
    device: &str,
    sample: &Sample,
) -> diagnostic::Result<Execution> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let profile = catalogue.resolve(device)?;
    if profile.meta.emit != crate::prof::Emit::C {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "设备 `{}` 的输出语言是 `{}`，本机只能编译执行 C",
                profile.meta.id, profile.meta.emit
            ),
        ));
    }
    // 证据先于执行：一条没有被执行证据的能力，跑出来的数不能当结果用。
    let evidence = witness::establish(profile, settings);
    if let Some(error) = evidence.failure {
        return Err(error.with("该设备缺少执行证据，真实数据不在未验证的后端上运行"));
    }
    let run = Run::new(op, profile, evidence.proof(profile))?;
    let path = run
        .negotiate()
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let group = u64::from(run.model.group.max(1));
    check_sample(group, sample)?;

    // 形状、代价、计划与源文本：与 doctor / verify / bench 同一条流水线。
    let prepared = run.prepare(&path, settings, sample.m, sample.n)?;
    let harness = prepared.harness(op, sample, 1)?;
    let tool = crate::backend::detect();
    let (outcome, _) = build::execute(profile, tool.as_ref(), &prepared.key, &harness, settings)?;

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
