//! 2 位反量化：打包字节 `q`（每字节四个 2 位码），`y = scale * (code - 2)`。
//!
//! 与 [`crate::dequant_i4`] 的差别是每字节四个码：`n` 个输出元素对应 `n / 4` 个字节。
//! 低 2 位在前、高位在后；码是无符号 0..3，按对称零点 2 转成 -2..=1。打包骨架在
//! [`crate::dequant`]，本文件只写位宽与零点，`#[annotation::model]` 装配。

use crate::dequant;
use crate::model::{Cost, Sample, ShapeContext};
use crate::quant::I2;
use diagnostic::Error;

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = dequant::SCALE;
/// 2 位对称零点：码 0..3 减去 2 得到 -2..=1（取自配置表 [`I2`]）。
pub const ZERO: f32 = I2.zero as f32;

// 单轴定义与形状投影复用打包骨架。
pub use crate::dequant::{AXES, project_shape};

/// 标量代价：每元素一次减法一次乘法；读打包字节（n/4 个），写 n 个 f32。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    dequant::cost_packed(shape, I2.cpb)
}

/// 确定性样本：`q`（打包字节，n/4 个）、`y`（n 个 f32）。
pub fn sample(len: usize, _: usize) -> Sample {
    dequant::sample_packed(len, I2.cpb)
}

/// 标量参考：逐字节解四个码，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let q = sample.u8("q")?;
    Ok(dequant::reference_packed(q, I2.cpb, I2.zero as f32))
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 4 倍时返回说明。
pub fn forward(scale: f32, zero: f32, q: &[u8], out: &mut [f32]) -> diagnostic::Result<()> {
    dequant::forward_packed(scale, zero, q, out, I2.cpb)
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DequantI2;

/// 2 位反量化的具体语义别名。
#[annotation::model(name = "dequant_i2", shape = [4096, 4096], dtype = "f32", skeleton = "dequant_i2")]
pub type DequantI2F32 = DequantI2;

#[cfg(test)]
#[path = "../tests/dequant_i2_tests.rs"]
mod tests;
