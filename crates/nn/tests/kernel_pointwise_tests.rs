mod clamp {
    pub use crate::clamp::*;

    mod tests {
use super::*;

#[test]
fn clamp_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(
        LO,
        HI,
        tensor::View::one(sample.f32("x").expect("x")),
        &mut out,
    )
    .expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn clamp_rejects_length_mismatch() {
    let mut out = vec![0.0; 3];
    assert!(
        forward(LO, HI, tensor::View::one(&[1.0, 2.0]), &mut out).is_err(),
        "x 与 out 不等长拒绝"
    );
}

/// 标量界负例：下界大于上界、任一界非有限，都在裁剪前拦下。
#[test]
fn clamp_rejects_illegal_bounds() {
    let x = [1.0f32, 2.0];
    let mut out = [0.0; 2];

    assert!(
        forward(1.0, -1.0, tensor::View::one(&x), &mut out).is_err(),
        "下界大于上界拒绝"
    );
    assert!(
        forward(f32::INFINITY, 1.0, tensor::View::one(&x), &mut out).is_err(),
        "非有限下界拒绝"
    );
    assert!(
        forward(-1.0, f32::NAN, tensor::View::one(&x), &mut out).is_err(),
        "非有限上界拒绝"
    );
}

/// 饱和与重复边界：越界值被裁到界内，极值结果有限且与参考一致。
#[test]
fn clamp_keeps_values_in_bounds_on_saturated_inputs() {
    let cases: &[&[f32]] = &[
        &[-3.0, 0.0, 3.0],
        &[f32::MAX, f32::MIN, 0.5],
        &[1.0, 1.0, -1.0],
    ];
    for (idx, x) in cases.iter().enumerate() {
        let lo = -1.0f32;
        let hi = 1.0f32;
        let mut out = vec![0.0; x.len()];
        forward(lo, hi, tensor::View::one(x), &mut out)
            .expect("等长非空");
        assert!(
            out.iter().all(|v| v.is_finite() && *v >= lo && *v <= hi),
            "用例 {idx} 裁剪到界内"
        );
        let expected: Vec<f32> = x.iter().map(|v| v.clamp(lo, hi)).collect();
        assert_eq!(out, expected, "用例 {idx} 与参考一致");
    }
}

    }
}

mod embed {
    pub use crate::embed::*;

    mod tests {
use super::*;

#[test]
fn embed_matches_the_row_reference() {
    let sample = sample(6, 4);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 6 * 4];
    forward(
        tensor::View::one(sample.f32("idx").expect("idx")),
        tensor::View::one(sample.f32("table").expect("table")),
        &mut out,
    )
    .expect("几何自洽");
    assert_eq!(out, expected);
}

#[test]
fn embed_rejects_an_out_of_range_idx() {
    let sample = sample(4, 3);
    let table = sample.f32("table").expect("table");
    let mut out = vec![0.0; 4 * 3];
    let idx = vec![100.0; 4];
    assert!(
        forward(
            tensor::View::one(idx.as_slice()),
            tensor::View::one(table),
            &mut out
        )
        .is_err(),
        "下标越界词表应被拒绝"
    );
}

#[test]
fn embed_rejects_a_table_wrong_length() {
    let sample = sample(4, 3);
    let idx = sample.f32("idx").expect("idx");
    let mut out = vec![0.0; 4 * 3];
    let table = vec![0.0; 1];
    assert!(
        forward(
            tensor::View::one(idx),
            tensor::View::one(table.as_slice()),
            &mut out
        )
        .is_err(),
        "表长不符行宽应被拒绝"
    );
}

/// 退化与重复边界：空索引拒绝；行重复取值时逐行 gather 与参考一致。
#[test]
fn embed_handles_empty_and_repeated_idxs() {
    let sample = sample(4, 3);
    let table = sample.f32("table").expect("table");
    let mut out = vec![];
    assert!(
        forward(tensor::View::one(&[]), tensor::View::one(table), &mut out).is_err(),
        "空索引必须拒绝"
    );
    let repeated = vec![0.0, 0.0, 1.0, 1.0];
    let mut out = vec![0.0; repeated.len() * 3];
    let mut repeat_sample = crate::embed_f32::sample(repeated.len(), 3);
    repeat_sample.bufs[0].values = crate::model::Values::F32(repeated.clone());
    let expected = reference(&repeat_sample).expect("参考可计算");
    forward(
        tensor::View::one(repeated.as_slice()),
        tensor::View::one(table),
        &mut out,
    )
    .expect("重复索引不越界");
    assert_eq!(out, expected);
}


#[test]
fn embed_rejection_leaves_the_output_untouched() {
    let idx = [0.0, 2.0];
    let table = [1.0, 2.0, 3.0, 4.0];
    let mut output = [9.0; 4];
    assert!(
        forward(
            tensor::View::one(&idx),
            tensor::View::one(&table),
            &mut output,
        )
        .is_err(),
        "第二个下标越界必须拒绝"
    );
    assert_eq!(output, [9.0; 4], "校验失败前不得写入第一行");
}

    }
}

mod rmsnorm {
    pub use crate::rmsnorm::*;
    use crate::model::{Buf, Sample};

    mod tests {
use super::*;

#[test]
fn rmsnorm_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(
        tensor::View::one(sample.f32("x").expect("x")),
        tensor::View::one(sample.f32("gamma").expect("gamma")),
        &mut out,
    )
    .expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn rmsnorm_rejects_empty_input() {
    let mut out = vec![];
    assert!(
        forward(tensor::View::one(&[]), tensor::View::one(&[]), &mut out).is_err(),
        "空行没有可统计的均方根"
    );
    assert!(reference(&sample(0, 1)).is_err(), "参考同样拒绝空行");
}

/// 三缓冲非对齐长度各自拒绝：x/gamma 不等、x/out 不等都不能静默截位。
#[test]
fn rmsnorm_rejects_each_mismatched_buf_length() {
    let mut out = vec![0.0; 3];
    assert!(
        forward(
            tensor::View::one(&[1.0, 2.0]),
            tensor::View::one(&[1.0]),
            &mut out
        )
        .is_err(),
        "x 与 gamma 不等"
    );
    assert!(
        forward(
            tensor::View::one(&[1.0, 2.0]),
            tensor::View::one(&[1.0, 2.0]),
            &mut out
        )
        .is_err(),
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
    forward(
        tensor::View::one(x.as_slice()),
        tensor::View::one(gamma.as_slice()),
        &mut out,
    )
    .expect("EPS 托底后全零行可归一化");
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
    for (idx, x) in cases.iter().enumerate() {
        let gamma: Vec<f32> = x.iter().map(|_| 0.75).collect();
        let sample = Sample {
            m: x.len(),
            n: x.len(),
            scalars: vec![],
            bufs: vec![
                Buf {
                    name: "x",
                    values: crate::model::Values::F32(x.to_vec()),
                },
                Buf {
                    name: "gamma",
                    values: crate::model::Values::F32(gamma.clone()),
                },
                Buf {
                    name: "y",
                    values: crate::model::Values::F32(vec![0.0; x.len()]),
                },
            ],
        };
        let expected = reference(&sample).expect("参考可计算");
        let mut out = vec![0.0; x.len()];
        forward(
            tensor::View::one(x),
            tensor::View::one(gamma.as_slice()),
            &mut out,
        )
        .expect("等长非空");
        assert_eq!(out, expected, "用例 {idx}");
        assert!(out.iter().all(|v| v.is_finite()), "用例 {idx}：结果有限");
    }
}

#[test]
fn batch_normalizes_each_segment_independently() {
    let row_dim = 3;
    let batch = 2;
    let x: Vec<f32> = (0..batch * row_dim).map(|i| i as f32 * 0.5 - 2.0).collect();
    let gamma: Vec<f32> = (0..row_dim).map(|i| 0.6 + i as f32 * 0.1).collect();

    let mut out = vec![0.0; x.len()];
    forward_batch(
        tensor::View::one(x.as_slice()),
        tensor::View::one(gamma.as_slice()),
        &mut out,
        row_dim,
    )
    .expect("批量归一化");

    // 参考：每段独立走单行入口，gamma 逐段复用，段间不混均值。
    let mut expected = vec![0.0; x.len()];
    for (b, seg) in x.chunks(row_dim).enumerate() {
        let span = b * row_dim..(b + 1) * row_dim;
        forward(
            tensor::View::one(seg),
            tensor::View::one(gamma.as_slice()),
            &mut expected[span],
        )
        .expect("段内参考");
    }
    assert_eq!(out, expected);
}

#[test]
fn batch_one_entry_matches_the_single_row_kernel() {
    let x = vec![1.0, -2.0, 0.5, 3.0];
    let gamma = vec![0.7, 0.6, 0.8, 0.5];
    let mut single = vec![0.0; x.len()];
    forward(
        tensor::View::one(x.as_slice()),
        tensor::View::one(gamma.as_slice()),
        &mut single,
    )
    .expect("单行");
    let mut batch = vec![0.0; x.len()];
    forward_batch(
        tensor::View::one(x.as_slice()),
        tensor::View::one(gamma.as_slice()),
        &mut batch,
        x.len(),
    )
    .expect("单段经批量入口");
    assert_eq!(single, batch, "batch=1 必须与单行入口逐元素等价");
}

#[test]
fn batch_rejects_bad_row_dimension_and_length() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![0.5; 2];
    let mut out = vec![0.0; 4];
    // 行宽为 0。
    assert!(
        forward_batch(
            tensor::View::one(x.as_slice()),
            tensor::View::one(gamma.as_slice()),
            &mut out,
            0
        )
        .is_err()
    );
    // gamma 长度与行宽不符。
    let wrong_gamma = vec![0.5; 3];
    assert!(
        forward_batch(
            tensor::View::one(x.as_slice()),
            tensor::View::one(wrong_gamma.as_slice()),
            &mut out,
            2
        )
        .is_err()
    );
    // x 与 out 不等长。
    let mut short = vec![0.0; 3];
    assert!(
        forward_batch(
            tensor::View::one(x.as_slice()),
            tensor::View::one(gamma.as_slice()),
            &mut short,
            2
        )
        .is_err()
    );
    // 缓冲不是行宽整数倍。
    assert!(
        forward_batch(
            tensor::View::one(&x[..3]),
            tensor::View::one(gamma.as_slice()),
            &mut short,
            2
        )
        .is_err()
    );
}


#[test]
fn rmsnorm_epsilon_is_configurable_and_validated_before_writing() {
    let x = [3.0, 4.0];
    let gamma = [1.0, 1.0];
    let mut output = [0.0; 2];
    forward_with_epsilon(
        tensor::View::one(&x),
        tensor::View::one(&gamma),
        &mut output,
        0.0,
    )
    .expect("零 epsilon 对非零行合法");
    let scale = (12.5f32).sqrt().recip();
    assert!((output[0] - 3.0 * scale).abs() < 1e-6);
    assert!((output[1] - 4.0 * scale).abs() < 1e-6);

    for epsilon in [-1.0, f32::NAN, f32::INFINITY] {
        let mut untouched = [7.0; 2];
        assert!(
            forward_with_epsilon(
                tensor::View::one(&x),
                tensor::View::one(&gamma),
                &mut untouched,
                epsilon,
            )
            .is_err()
        );
        assert_eq!(untouched, [7.0; 2], "非法 epsilon 不得留下半写状态");
    }
}

    }
}

mod softmax {
    pub use crate::softmax::*;

    mod tests {
use super::*;

#[test]
fn softmax_matches_the_scalar_reference_and_sums_to_one() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let total: f32 = expected.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax 输出和为 1");
    let mut out = vec![0.0; 128];
    forward(tensor::View::one(sample.f32("x").expect("x")), &mut out).expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn softmax_rejects_empty_input() {
    let mut out = vec![];
    assert!(
        forward(tensor::View::one(&[]), &mut out).is_err(),
        "空行没有可归一化的分母"
    );
    assert!(reference(&sample(0, 1)).is_err(), "参考同样拒绝空行");
}

/// 输入与输出长度不一致时拒绝；非对齐长度不静默截断。
#[test]
fn softmax_rejects_mismatched_output_length() {
    testkit::assert_rejects_length_mismatch(|input, out| {
        forward(tensor::View::one(input), out).is_err()
    });
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
    for (idx, input) in cases.iter().enumerate() {
        let mut out = vec![0.0; input.len()];
        forward(tensor::View::one(input), &mut out).expect("非空行可归一化");
        let total: f32 = out.iter().sum();
        assert!(total.is_finite(), "用例 {idx}：归一化结果必须有限");
        assert!((total - 1.0).abs() < 1e-5, "用例 {idx}：每行和为 1");
        assert!(out.iter().all(|v| v.is_finite()), "用例 {idx}：无 NaN/Inf");
    }
}

    }
}
