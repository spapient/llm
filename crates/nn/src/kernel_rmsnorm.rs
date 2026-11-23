//! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
use crate::model::{Buf, Cost, Sample, ShapeCtx, Values, cost_n, x_buf};
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
            Buf {
                name: "gamma",
                values: Values::F32((0..len).map(|i| 0.5 + (i as f32) * 0.01).collect()),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 参考两参前置：长度一致且非空（空行没有可统计的均方根）。
fn check_ref(x: &[f32], gamma: &[f32]) -> diagnostic::Result<()> {
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

/// 内核三参前置：等长且非空。
fn check_kernel(x: &[f32], gamma: &[f32], out: &[f32]) -> diagnostic::Result<()> {
    if x.len() != gamma.len() || x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "缓冲长度不一致：x {}、gamma {}、out {}",
                x.len(),
                gamma.len(),
                out.len()
            ),
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

/// 标量参考：两趟，与内核同一公式。
///
/// # Errors
///
/// 输入为空时返回说明：空行没有可统计的均方根。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let gamma = sample.f32("gamma")?;
    check_ref(x, gamma)?;
    let rms = (sum_sq(x) / (x.len() as f32) + EPS).sqrt();
    Ok(x.iter()
        .zip(gamma.iter())
        .map(|(value, weight)| weight * value / rms)
        .collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三个缓冲等长，或输入为空时返回说明：空行没有可统计的均方根。
pub fn forward(x: View<'_, f32>, gamma: View<'_, f32>, out: ViewMut<'_, f32>) -> diagnostic::Result<()> {
    let x = x.as_slice();
    let gamma = gamma.as_slice();
    let out = out.into_slice_mut();
    check_kernel(x, gamma, out)?;
    let rms = (sum_sq(x) / (x.len() as f32) + EPS).sqrt();
    for ((out, x), gamma) in out.iter_mut().zip(x).zip(gamma) {
        *out = gamma * x / rms;
    }
    Ok(())
}

/// 批段前置：行宽非空、gamma 与行宽等长、x/out 等长且为行宽整数倍；返回批内段数。
fn rmsnorm_batch_segments(
    x: &[f32],
    gamma: &[f32],
    out: &[f32],
    row_dim: usize,
) -> Result<usize, Error> {
    if row_dim == 0 {
        return Err(Error::new(Code::Invalid, "rmsnorm 行宽不能为 0"));
    }
    if gamma.len() != row_dim {
        return Err(Error::new(
            Code::Invalid,
            format!("gamma 长度 {} 不等于行宽 {row_dim}", gamma.len()),
        ));
    }
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
        ));
    }
    check_batch_rows(x, row_dim)?;
    Ok(x.len() / row_dim)
}

/// 批行校验：缓冲非空且为行宽整数倍；空行没有可统计的均方根。
fn check_batch_rows(x: &[f32], row_dim: usize) -> Result<(), Error> {
    if x.is_empty() || !x.len().is_multiple_of(row_dim) {
        return Err(Error::new(Code::Invalid, "rmsnorm 批缓冲不是行宽整数倍"));
    }
    Ok(())
}

/// 批量入口：`[batch, row_dim]` 段连续，每段独立 RMS 归一化，gamma 按行宽复用。
///
/// # Errors
///
/// 行宽为 0、gamma 与行宽不等长、x/out 不等长、缓冲非行宽倍数或为空时返回说明。
pub fn forward_batch(
    x: View<'_, f32>,
    gamma: View<'_, f32>,
    out: ViewMut<'_, f32>,
    row_dim: usize,
) -> diagnostic::Result<()> {
    let x = x.as_slice();
    let gamma = gamma.as_slice();
    let out = out.into_slice_mut();
    let batch = rmsnorm_batch_segments(x, gamma, out, row_dim)?;
    for b in 0..batch {
        let start = b * row_dim;
        forward(
            View::one(&x[start..start + row_dim]),
            View::one(gamma),
            ViewMut::one(&mut out[start..start + row_dim]),
        )?;
    }
    Ok(())
}

/// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
pub type RmsNormF32 = RmsNorm;

crate::operator_here!(RmsNorm, "rmsnorm_f32", 2, (4096, 4096), true, reference);

/// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
pub struct RmsNorm;

#[cfg(test)]
mod tests {
    include!("../tests/kernel_rmsnorm.rs");
}
