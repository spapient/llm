//! 8 位反量化：`y[n] = scale * (q[n] - zero)`。
//!
//! 量化值以 f32 承载（当前 `Values` 只有 f32 变体），`scale` 与 `zero` 由样本标量给出。
//! 本模块自带形状、代价、样本、参考与遍历，`#[annotation::model]` 装配。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 单轴定义复用 unary 骨架。
pub use crate::unary::AXES;

/// 样本固定的反量化参数。
pub const SCALE: f32 = 0.25;
/// 样本固定的零点。
pub const ZERO: f32 = 8.0;

// 形状投影复用 unary 骨架。
pub use crate::unary::project_shape;

/// 标量代价：每元素一次减法一次乘法，读一次写一次。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "dequant 形状缺轴 `n`"))?;
    Ok(Cost {
        flops: (n as f64) * 2.0,
        read: (n as u64) * 4,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`q`（f32 承载的量化整数）、`y`，标量 `scale`/`zero`。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            Buffer {
                name: "q",
                values: Values::F32((0..len).map(|i| (i % 16) as f32).collect()),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let q = sample.f32("q")?;
    Ok(q.iter().map(|value| SCALE * (value - ZERO)).collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward(scale: f32, zero: f32, q: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if q.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：q {}、out {}", q.len(), out.len()),
        ));
    }
    for (out, q) in out.iter_mut().zip(q) {
        *out = scale * (q - zero);
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct DequantI8;

/// 反量化的具体语义别名。
#[annotation::model(name = "dequant_i8", shape = [4096, 4096], dtype = "f32", skeleton = "dequant")]
pub type DequantI8F32 = DequantI8;

#[cfg(test)]
#[path = "../tests/dequant_i8_tests.rs"]
mod tests;
