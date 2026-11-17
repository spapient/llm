use super::*;

#[test]
fn dequant_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(SCALE, ZERO, sample.f32("q").expect("q"), &mut out).expect("等长");
    assert_eq!(out, expected);
}

/// 退化与边界：空行放行；长度不齐拒绝；饱和量化值与零档不产生 NaN。
#[test]
fn dequant_i8_handles_empty_mismatched_and_saturated_values() {
    let mut empty = vec![];
    forward(SCALE, ZERO, &[], &mut empty).expect("空行放行");

    let mut out = vec![0.0; 2];
    assert!(
        forward(SCALE, ZERO, &[1.0, 2.0, 3.0], &mut out).is_err(),
        "q 与 out 不等长拒绝"
    );

    let saturated = [0.0f32, 15.0, f32::MIN, f32::MAX];
    let mut out = vec![0.0; saturated.len()];
    forward(SCALE, ZERO, &saturated, &mut out).expect("饱和值不越界");
    assert!(out.iter().all(|v| v.is_finite()), "结果有限");
}
