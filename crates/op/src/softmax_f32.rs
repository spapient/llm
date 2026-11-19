//! Softmax：`y[n] = exp(x[n] - max(x)) / sum_k exp(x[k] - max(x))`。
//!
//! 三趟遍历同一行：先求最大值（数值稳定，减去它再取指数不会溢出），再求指数和，最后归一化。
//! 本模块自带形状、代价、样本、标量参考与遍历，`#[annotation::model]` 装配成 `MODEL`。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用 unary 骨架：本文件只声明 softmax 特有的代价、样本与遍历。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：三趟读 `x`、一趟写 `y`，每元素约 3 次运算（max、exp、除法）。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    cost_n(shape, "softmax", 3.0, 3)
}

/// 确定性样本：`x`、`y` 两个等长缓冲。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            x_buffer(len),
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：三趟，与内核同一公式。
///
/// # Errors
///
/// 输入为空时返回说明：空行没有可归一化的分母。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    if x.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            "softmax 输入不能为空：归一化分母无定义",
        ));
    }
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
/// 输入与输出等长，或输入为空时返回说明：空行没有可归一化的分母。
pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    if x.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            "softmax 输入不能为空：归一化分母无定义",
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
mod tests {
    use super::*;

    #[test]
    fn softmax_matches_the_scalar_reference_and_sums_to_one() {
        let sample = sample(128, 1);
        let expected = reference(&sample).expect("参考可计算");
        let total: f32 = expected.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax 输出和为 1");
        let mut out = vec![0.0; 128];
        forward(sample.f32("x").expect("x"), &mut out).expect("等长");
        assert_eq!(out, expected);
    }

    #[test]
    fn softmax_rejects_empty_input() {
        let mut out = vec![];
        assert!(forward(&[], &mut out).is_err(), "空行没有可归一化的分母");
        assert!(reference(&sample(0, 1)).is_err(), "参考同样拒绝空行");
    }

    /// 输入与输出长度不一致时拒绝；非对齐长度不静默截断。
    #[test]
    fn softmax_rejects_mismatched_output_length() {
        let mut out = vec![0.0; 3];
        assert!(
            forward(&[1.0, 2.0], &mut out).is_err(),
            "输出比输入长必须拒绝"
        );
        let mut short = vec![0.0; 1];
        assert!(
            forward(&[1.0, 2.0, 3.0], &mut short).is_err(),
            "输出比输入短必须拒绝"
        );
    }

    /// 饱和与退化输入：全零、重复元素、同号大值与极值；每行仍归一化到 1 且不含 NaN/Inf。
    #[test]
    fn softmax_saturating_and_repeated_inputs_stay_normalized() {
        let cases: &[&[f32]] = &[
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[-2.0, -1.0, 0.0, 1.0, 2.0],
            &[f32::MAX, f32::MAX],
            &[f32::MIN, f32::MIN],
            &[3.5, -4.0, 3.5],
        ];
        for (index, input) in cases.iter().enumerate() {
            let mut out = vec![0.0; input.len()];
            forward(input, &mut out).expect("非空行可归一化");
            let total: f32 = out.iter().sum();
            assert!(total.is_finite(), "用例 {index}：归一化结果必须有限");
            assert!((total - 1.0).abs() < 1e-5, "用例 {index}：每行和为 1");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "用例 {index}：无 NaN/Inf"
            );
        }
    }
}
