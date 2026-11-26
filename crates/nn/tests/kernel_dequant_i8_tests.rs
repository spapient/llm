use super::*;

testkit::test_cases! {
fn dequant_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(
        SCALE,
        ZERO,
        tensor::View::one(sample.f32("q").expect("q")),
        &mut out,
    )
    .expect("等长");
    assert_eq!(out, expected);
}

/// 退化与边界：空行放行；长度不齐拒绝；饱和量化值与零档不产生 NaN。
fn dequant_i8_handles_empty_mismatched_and_saturated_values() {
    let mut empty = vec![];
    forward(SCALE, ZERO, tensor::View::one(&[]), &mut empty).expect("空行放行");

    let mut out = vec![0.0; 2];
    assert!(
        forward(SCALE, ZERO, tensor::View::one(&[1.0, 2.0, 3.0]), &mut out).is_err(),
        "q 与 out 不等长拒绝"
    );

    let saturated = [0.0f32, 15.0, f32::MIN, f32::MAX];
    let mut out = vec![0.0; saturated.len()];
    forward(SCALE, ZERO, tensor::View::one(&saturated), &mut out).expect("饱和值不越界");
    assert!(out.iter().all(|v| v.is_finite()), "结果有限");
}

/// 仿射参数负例：零标度、非有限标度、越出 i4 码域（0..15）的零点都在解包前拦下。
fn dequant_i8_rejects_zero_scale_and_out_of_domain_zero() {
    let q = [8.0f32, 12.0];
    let mut out = [0.0; 2];

    assert!(
        forward(0.0, ZERO, tensor::View::one(&q), &mut out).is_err(),
        "零标度拒绝"
    );
    assert!(
        forward(f32::INFINITY, ZERO, tensor::View::one(&q), &mut out).is_err(),
        "非有限标度拒绝"
    );
    assert!(
        forward(SCALE, 16.0, tensor::View::one(&q), &mut out).is_err(),
        "零点越出 0..15 码域"
    );
    assert!(
        forward(SCALE, -1.0, tensor::View::one(&q), &mut out).is_err(),
        "负零点越出码域"
    );
}
}
