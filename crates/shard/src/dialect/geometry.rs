//! 签名 / 样本几何助手：从 [`Signature`] 与 [`Sample`] 取长度、缓冲名、实参与输出形状，
//! 归约 / 扫描 / 归一化 / 逐点各族 lowering 共用这一条查表。

use cutile::harness::output_name;
use diagnostic::{Code, Error, Result};
use prof::model::{Sample, Values};
use prof::sig::{Param, Signature};

/// 长度标量：签名里唯一非缓冲参数（扫描/归约/归一化的 `N`），取样本值。
pub(crate) fn length_scalar(signature: &Signature, sample: &Sample) -> Result<i64> {
    let scalar = scalar_name(signature)?;
    sample.scalar(scalar).ok_or(Error::new(
        Code::Invalid,
        format!("长度标量 `{scalar}` 没有样本取值"),
    ))
}

/// 唯一长度标量名（非缓冲参数）：归约/扫描/循环注入共用。
pub(crate) fn scalar_name(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buf())
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有长度标量").with("shard::dialect::scalar_name"))
}

/// 只读输入缓冲名（唯一非可写缓冲）：归约/扫描/循环注入三族 lowering 共用这一条查表。
pub(crate) fn read_input(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .find(|param| !param.mutable)
        .map(|param| param.name.as_str())
        .ok_or(
            Error::new(Code::Internal, "内核没有只读输入缓冲").with("shard::dialect::read_input"),
        )
}

/// 长度标量的 `usize` 视图；负值或超长返回 `Invalid`。归约 / 扫描 / 归一化三族的几何判定共用。
pub(crate) fn length_usize(signature: &Signature, sample: &Sample) -> Result<usize> {
    let len = length_scalar(signature, sample)?;
    usize::try_from(len).map_err(|_| {
        Error::new(
            Code::Invalid,
            format!(
                "长度标量 `{}` = {len} 不是非负 usize",
                scalar_name(signature).unwrap_or("?")
            ),
        )
        .with("shard::dialect::length_usize")
    })
}

/// 只读缓冲名，按签名顺序。CUDA / Metal 的归约、扫描、归一化 lowering 共用这一条查表。
pub(crate) fn readonly_bufs(signature: &Signature) -> Vec<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf() && !param.mutable)
        .map(|param| param.name.as_str())
        .collect()
}

/// launch 实参：缓冲传设备指针 `d_<name>`，标量取样本值。
pub(crate) fn launch_args(sample: &Sample, params: &[Param]) -> Result<String> {
    let mut args = Vec::with_capacity(params.len());
    for param in params {
        if param.ctype.is_buf() {
            args.push(format!("d_{}", param.name));
        } else {
            let value = sample.scalar(&param.name).ok_or_else(|| {
                Error::new(
                    Code::Internal,
                    format!("标量实参 `{}` 没有取值", param.name),
                )
            })?;
            args.push(value.to_string());
        }
    }
    Ok(args.join(", "))
}

/// 输出缓冲的（名字，元素数）：只接受非空 f32，回读路径按 float 解释字节。三端 benchmark/回读共用。
pub(crate) fn f32_output_shape(signature: &Signature, sample: &Sample) -> Result<(String, usize)> {
    let output = output_name(signature)?;
    let values = sample
        .buf(output)
        .ok_or(Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))?;
    if !matches!(&values.values, Values::F32(_)) || values.values.is_empty() {
        return Err(Error::new(
            Code::Unsupported,
            "设备回读当前要求非空 f32 输出",
        ));
    }
    Ok((output.to_owned(), values.values.len()))
}

/// 生产态实参：缓冲取 `d_{name} + n`（线程处理的元素）、标量取 `1`（每线程一个元素）。
pub(crate) fn grid_args(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            if param.ctype.is_buf() {
                format!("d_{} + n", param.name)
            } else {
                "1".to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}
