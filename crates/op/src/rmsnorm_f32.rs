//! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
//!
//! 两趟遍历同一行：先累加平方和求均方根，再归一化并乘权重。常量 `eps = 1e-6`。本模块自带
//! 形状、代价、样本、标量参考与遍历，`#[annotation::model]` 把这几个 fn 装配成 `MODEL`。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

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
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "rmsnorm 形状缺轴 `n`"))?;
    let bytes = (n as u64) * 4;
    Ok(Cost {
        flops: (n as f64) * 4.0,
        read: bytes * 3,
        write: bytes,
    })
}

/// 确定性样本：`x`、`gamma`、`y` 三个等长缓冲。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32((0..len).map(|i| ((i as f32) * 0.25) - 2.0).collect()),
            },
            Buffer {
                name: "gamma",
                values: Values::F32((0..len).map(|i| 0.5 + (i as f32) * 0.01).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：两趟，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let gamma = sample.f32("gamma")?;
    if x.len() != gamma.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("x 与 gamma 长度不一致：{} 与 {}", x.len(), gamma.len()),
        ));
    }
    let len = x.len();
    let mut sum_sq = 0.0;
    for value in x {
        sum_sq += value * value;
    }
    let rms = (sum_sq / (len as f32) + EPS).sqrt();
    Ok(x.iter()
        .zip(gamma.iter())
        .map(|(value, weight)| weight * value / rms)
        .collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三个缓冲等长，否则返回说明。
pub fn forward(x: &[f32], gamma: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
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
    let len = x.len();
    let mut sum_sq = 0.0;
    for value in x {
        sum_sq += value * value;
    }
    let rms = (sum_sq / (len as f32) + EPS).sqrt();
    for ((out, x), gamma) in out.iter_mut().zip(x).zip(gamma) {
        *out = gamma * x / rms;
    }
    Ok(())
}

/// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
pub type RmsNormF32 = RmsNorm;

/// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
pub struct RmsNorm;

#[cfg(test)]
#[path = "rmsnorm_f32_tests.rs"]
mod tests;
