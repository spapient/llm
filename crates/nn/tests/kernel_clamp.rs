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
