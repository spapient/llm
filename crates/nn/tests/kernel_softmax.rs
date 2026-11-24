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
