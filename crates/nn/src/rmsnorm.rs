//! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
use crate::model::{Buf, Cost, Sample, ShapeCtx, cost_n, x_buf};
use diagnostic::{Code, Error};
use tensor::View;

/// 数值稳定项：均方根里加的小常数。
pub const EPS: f32 = 1e-6;

// 单轴定义与形状投影复用 unary 骨架：本文件只声明 rmsnorm 特有的代价、样本与遍历。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：两趟读 `x`、一趟读 `gamma`、一趟写 `y`，运算按每元素约 4 次（平方、加法、
/// 除法、乘法）估算。这里给出可审计的显式口径，不伪装成一条 einsum。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
    cost_n(shape, "rmsnorm", 4.0, 3)
}

/// 确定性样本：`x`、`gamma`、`y` 三个等长缓冲。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        bufs: vec![
            x_buf(len),
            Buf::f32("gamma", crate::model::linear(len, 0.01, 0.5)),
            Buf::f32("y", vec![0.0; len]),
        ],
    }
}

/// 公共前置：输入与权重等长且非空；输出长度由对应入口单独核对。
fn check(x: &[f32], gamma: &[f32]) -> diagnostic::Result<()> {
    if x.len() != gamma.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("x 与 gamma 长度不一致：{} 与 {}", x.len(), gamma.len()),
        ));
    }
    if x.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            "rmsnorm 输入不能为空：均方根无定义",
        ));
    }
    Ok(())
}

/// 归一化第一趟：平方和。累加是串行依赖，保留逐元素循环；参考与内核共用同一份口径。
fn sum_sq(x: &[f32]) -> f32 {
    let mut sum = 0.0;
    for value in x {
        sum += value * value;
    }
    sum
}

/// 校验数值稳定项；有限且非负才允许进入参考或内核。
fn validate_epsilon(epsilon: f32) -> diagnostic::Result<()> {
    if epsilon.is_finite() && epsilon >= 0.0 {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        "rmsnorm epsilon 必须是有限非负数",
    ))
}

/// 一次平方根和倒数，供整行乘法复用，避免每元素一次除法。
fn inverse_rms(x: &[f32], epsilon: f32) -> f32 {
    (sum_sq(x) / (x.len() as f32) + epsilon).sqrt().recip()
}

/// 标量参考：两趟，与内核同一公式。
///
/// # Errors
///
/// 输入为空或权重长度不符时返回说明。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let gamma = sample.f32("gamma")?;
    check(x, gamma)?;
    let scale = inverse_rms(x, EPS);
    Ok(x.iter()
        .zip(gamma.iter())
        .map(|(value, weight)| weight * value * scale)
        .collect())
}

/// 单行数值核：调用方已完成校验；批路径复用它以避免逐行重复检查。
fn normalize(x: &[f32], gamma: &[f32], out: &mut [f32], epsilon: f32) {
    let scale = inverse_rms(x, epsilon);
    for ((out, x), gamma) in out.iter_mut().zip(x).zip(gamma) {
        *out = gamma * x * scale;
    }
}

/// 使用默认 [`EPS`] 的内核入口。
///
/// # Errors
///
/// 三个缓冲不等长或输入为空时返回说明。
pub fn forward(
    x: View<'_, f32>,
    gamma: View<'_, f32>,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    forward_with_epsilon(x, gamma, out, EPS)
}

/// 可配置 epsilon 的内核入口；全部校验完成后才开始写入。
///
/// # Errors
///
/// 三个缓冲不等长、输入为空或 epsilon 非法时返回说明。
pub fn forward_with_epsilon(
    x: View<'_, f32>,
    gamma: View<'_, f32>,
    out: &mut [f32],
    epsilon: f32,
) -> diagnostic::Result<()> {
    validate_epsilon(epsilon)?;
    let x = x.as_slice();
    let gamma = gamma.as_slice();
    check(x, gamma)?;
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
        ));
    }
    normalize(x, gamma, out, epsilon);
    Ok(())
}

/// 批前置：只在本入口校验一次，行宽与非空约束覆盖所有后续分段。
fn check_batch(x: &[f32], gamma: &[f32], out: &[f32], row_dim: usize) -> Result<(), Error> {
    check_row_config(gamma, row_dim)?;
    check_batch_bufs(x, out, row_dim)
}

/// 行配置只需校验一次，与整批输入输出缓冲判据分离。
fn check_row_config(gamma: &[f32], row_dim: usize) -> Result<(), Error> {
    if row_dim == 0 {
        return Err(Error::new(Code::Invalid, "rmsnorm 行宽不能为 0"));
    }
    if gamma.len() != row_dim {
        return Err(Error::new(
            Code::Invalid,
            format!("gamma 长度 {} 不等于行宽 {row_dim}", gamma.len()),
        ));
    }
    Ok(())
}

/// 批缓冲必须等长、非空且能按行宽完整分段。
fn check_batch_bufs(x: &[f32], out: &[f32], row_dim: usize) -> Result<(), Error> {
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
        ));
    }
    if x.is_empty() || !x.len().is_multiple_of(row_dim) {
        return Err(Error::new(Code::Invalid, "rmsnorm 批缓冲不是行宽整数倍"));
    }
    Ok(())
}

/// 使用默认 [`EPS`] 的批量入口。
///
/// # Errors
///
/// 行宽、缓冲或默认配置不合法时返回说明。
pub fn forward_batch(
    x: View<'_, f32>,
    gamma: View<'_, f32>,
    out: &mut [f32],
    row_dim: usize,
) -> diagnostic::Result<()> {
    forward_batch_with_epsilon(x, gamma, out, row_dim, EPS)
}

/// 可配置 epsilon 的批量入口；每段独立归一化并复用 gamma。
///
/// # Errors
///
/// 行宽为 0、gamma 与行宽不等长、缓冲不完整或 epsilon 非法时返回说明。
pub fn forward_batch_with_epsilon(
    x: View<'_, f32>,
    gamma: View<'_, f32>,
    out: &mut [f32],
    row_dim: usize,
    epsilon: f32,
) -> diagnostic::Result<()> {
    validate_epsilon(epsilon)?;
    let x = x.as_slice();
    let gamma = gamma.as_slice();
    check_batch(x, gamma, out, row_dim)?;
    for (x, out) in x.chunks_exact(row_dim).zip(out.chunks_exact_mut(row_dim)) {
        normalize(x, gamma, out, epsilon);
    }
    Ok(())
}

/// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
pub type RmsNormF32 = RmsNorm;

crate::operator_here!(RmsNorm, "rmsnorm_f32", 2, (4096, 4096), true, reference);

/// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
pub struct RmsNorm;
