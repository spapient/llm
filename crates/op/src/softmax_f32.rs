//! Softmax：`y[n] = exp(x[n] - max(x)) / sum_k exp(x[k] - max(x))`。
//!
//! 三趟遍历同一行：先求最大值（数值稳定，减去它再取指数不会溢出），再求指数和，最后归一化。
//! 本模块自带形状、代价、样本、标量参考与遍历，`#[annotation::model]` 装配成 `MODEL`。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用 unary 骨架：本文件只声明 softmax 特有的代价、样本与遍历。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：三趟读 `x`、一趟写 `y`，每元素约 3 次运算（max、exp、除法）。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "softmax 形状缺轴 `n`"))?;
    let bytes = (n as u64) * 4;
    Ok(Cost {
        flops: (n as f64) * 3.0,
        read: bytes * 3,
        write: bytes,
    })
}

/// 确定性样本：`x`、`y` 两个等长缓冲。
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
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：三趟，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let max = x.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    let exps: Vec<f32> = x
        .iter()
        .map(|value| {
            let e = (value - max).exp();
            sum += e;
            e
        })
        .collect();
    Ok(exps.iter().map(|e| e / sum).collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    let max = x.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    for value in x {
        sum += (value - max).exp();
    }
    for (out, x) in out.iter_mut().zip(x) {
        *out = (x - max).exp() / sum;
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Softmax;

/// Softmax 的具体语义别名。
#[annotation::model(name = "softmax_f32", shape = [4096, 4096], dtype = "f32", skeleton = "softmax")]
pub type SoftmaxF32 = Softmax;

#[cfg(test)]
#[path = "softmax_f32_tests.rs"]
mod tests;
