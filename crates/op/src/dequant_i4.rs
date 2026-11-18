//! 4 位反量化：打包字节 `q`（每字节两个 nibble），`y = scale * (nibble - 8)`。
//!
//! 与 [`crate::dequant_i8`] 的差别是输入为打包字节：`n` 个输出元素对应 `n / 2` 个字节。
//! 低 4 位在前、高 4 位在后；nibble 是无符号 0..15，按对称零点 8 转有符号。打包骨架在
//! [`crate::dequant`]，本文件只写位宽与零点，`#[annotation::model]` 装配。

use crate::dequant;
use crate::model::{Cost, Sample, ShapeContext};
use crate::quant::I4;
use diagnostic::Error;

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = dequant::SCALE;
/// 4 位对称零点（取自配置表 [`I4`]）。
pub const ZERO: f32 = I4.zero as f32;

// 单轴定义与形状投影复用打包骨架。
pub use crate::dequant::{AXES, project_shape};

/// 标量代价：每元素一次减法一次乘法；读打包字节（n/2 个），写 n 个 f32。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    dequant::cost_packed(shape, I4.cpb)
}

/// 确定性样本：`q`（打包字节，n/2 个）、`y`（n 个 f32）。
pub fn sample(len: usize, _: usize) -> Sample {
    dequant::sample_packed(len, I4.cpb)
}

/// 标量参考：逐字节解两个 nibble，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let q = sample.u8("q")?;
    Ok(dequant::reference_packed(q, I4.cpb, ZERO))
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 2 倍时返回说明。
pub fn forward(scale: f32, zero: f32, q: &[u8], out: &mut [f32]) -> diagnostic::Result<()> {
    dequant::forward_packed(scale, zero, q, out, I4.cpb)
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DequantI4;

/// 4 位反量化的具体语义别名。
#[annotation::model(name = "dequant_i4", shape = [4096, 4096], dtype = "f32", skeleton = "dequant_i4")]
pub type DequantI4F32 = DequantI4;

#[cfg(test)]
#[path = "../tests/dequant_i4_tests.rs"]
mod tests;
