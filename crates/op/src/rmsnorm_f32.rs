//! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
//!
//! 两趟遍历同一行：先累加平方和求均方根，再归一化并乘权重。常量 `eps = 1e-6`。本模块自带
//! 形状、代价、样本、标量参考与遍历，`#[annotation::model]` 把这几个 fn 装配成 `MODEL`。

use crate::model::{Buffer, Cost, Sample, ShapeContext, Values, cost_n, x_buffer};
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
    cost_n(shape, "rmsnorm", 4.0, 3)
}

/// 确定性样本：`x`、`gamma`、`y` 三个等长缓冲。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        buffers: vec![
            x_buffer(len),
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
pub fn forward(x: &[f32], gamma: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
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
    x: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    row_dim: usize,
) -> diagnostic::Result<()> {
    let batch = rmsnorm_batch_segments(x, gamma, out, row_dim)?;
    for b in 0..batch {
        let start = b * row_dim;
        forward(
            &x[start..start + row_dim],
            gamma,
            &mut out[start..start + row_dim],
        )?;
    }
    Ok(())
}

/// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
pub type RmsNormF32 = RmsNorm;

/// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
pub struct RmsNorm;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_matches_the_scalar_reference() {
        let sample = sample(128, 1);
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        forward(
            sample.f32("x").expect("x"),
            sample.f32("gamma").expect("gamma"),
            &mut out,
        )
        .expect("等长");
        assert_eq!(out, expected);
    }

    #[test]
    fn rmsnorm_rejects_empty_input() {
        let mut out = vec![];
        assert!(
            forward(&[], &[], &mut out).is_err(),
            "空行没有可统计的均方根"
        );
        assert!(reference(&sample(0, 1)).is_err(), "参考同样拒绝空行");
    }

    /// 三缓冲非对齐长度各自拒绝：x/gamma 不等、x/out 不等都不能静默截位。
    #[test]
    fn rmsnorm_rejects_each_mismatched_buffer_length() {
        let mut out = vec![0.0; 3];
        assert!(
            forward(&[1.0, 2.0], &[1.0], &mut out).is_err(),
            "x 与 gamma 不等"
        );
        assert!(
            forward(&[1.0, 2.0], &[1.0, 2.0], &mut out).is_err(),
            "x 与 out 不等"
        );
        assert!(reference(&sample(0, 1)).is_err(), "参考路径同样拒绝空行");
    }

    /// 零范数（全零行）：分母由 `EPS` 托底，输出为零而不是除零；权重按原样乘回。
    #[test]
    fn rmsnorm_zero_row_is_guided_by_epsilon() {
        let x = vec![0.0; 4];
        let gamma = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 4];
        forward(&x, &gamma, &mut out).expect("EPS 托底后全零行可归一化");
        assert!(out.iter().all(|v| v.abs() < 1e-6), "全零行输出趋近零");
    }

    /// 饱和与重复边界：大值、极值、重复元素行，结果有限且与参考一致。
    #[test]
    fn rmsnorm_saturating_and_repeated_inputs_match_reference() {
        let cases: &[&[f32]] = &[
            &[1.0, 1.0, 1.0],
            &[-1.0, 2.0, -3.0],
            &[f32::MAX; 3],
            &[f32::MIN; 3],
            &[0.25, -0.5, 0.75, -1.0],
        ];
        for (index, x) in cases.iter().enumerate() {
            let gamma: Vec<f32> = x.iter().map(|_| 0.75).collect();
            let sample = Sample {
                m: x.len(),
                n: x.len(),
                scalars: vec![],
                buffers: vec![
                    Buffer {
                        name: "x",
                        values: crate::model::Values::F32(x.to_vec()),
                    },
                    Buffer {
                        name: "gamma",
                        values: crate::model::Values::F32(gamma.clone()),
                    },
                    Buffer {
                        name: "y",
                        values: crate::model::Values::F32(vec![0.0; x.len()]),
                    },
                ],
            };
            let expected = reference(&sample).expect("参考可计算");
            let mut out = vec![0.0; x.len()];
            forward(x, &gamma, &mut out).expect("等长非空");
            assert_eq!(out, expected, "用例 {index}");
            assert!(out.iter().all(|v| v.is_finite()), "用例 {index}：结果有限");
        }
    }

    #[test]
    fn batch_normalizes_each_segment_independently() {
        let row_dim = 3;
        let batch = 2;
        let x: Vec<f32> = (0..batch * row_dim).map(|i| i as f32 * 0.5 - 2.0).collect();
        let gamma: Vec<f32> = (0..row_dim).map(|i| 0.6 + i as f32 * 0.1).collect();

        let mut out = vec![0.0; x.len()];
        forward_batch(&x, &gamma, &mut out, row_dim).expect("批量归一化");

        // 参考：每段独立走单行入口，gamma 逐段复用，段间不混均值。
        let mut expected = vec![0.0; x.len()];
        for (b, seg) in x.chunks(row_dim).enumerate() {
            let span = b * row_dim..(b + 1) * row_dim;
            forward(seg, &gamma, &mut expected[span]).expect("段内参考");
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn batch_one_entry_matches_the_single_row_kernel() {
        let x = vec![1.0, -2.0, 0.5, 3.0];
        let gamma = vec![0.7, 0.6, 0.8, 0.5];
        let mut single = vec![0.0; x.len()];
        forward(&x, &gamma, &mut single).expect("单行");
        let mut batch = vec![0.0; x.len()];
        forward_batch(&x, &gamma, &mut batch, x.len()).expect("单段经批量入口");
        assert_eq!(single, batch, "batch=1 必须与单行入口逐元素等价");
    }

    #[test]
    fn batch_rejects_bad_row_dimension_and_length() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![0.5; 2];
        let mut out = vec![0.0; 4];
        // 行宽为 0。
        assert!(forward_batch(&x, &gamma, &mut out, 0).is_err());
        // gamma 长度与行宽不符。
        let wrong_gamma = vec![0.5; 3];
        assert!(forward_batch(&x, &wrong_gamma, &mut out, 2).is_err());
        // x 与 out 不等长。
        let mut short = vec![0.0; 3];
        assert!(forward_batch(&x, &gamma, &mut short, 2).is_err());
        // 缓冲不是行宽整数倍。
        assert!(forward_batch(&x[..3], &gamma, &mut short, 2).is_err());
    }
}
