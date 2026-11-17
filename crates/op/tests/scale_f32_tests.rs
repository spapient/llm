use super::*;

#[test]
fn scale_matches_the_scalar_reference() {
    let sample = crate::unary::sample::<Scale>(128, 1);
    let expected = crate::unary::reference::<Scale>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    forward(a, &mut out).expect("缓冲等长");
    assert_eq!(out, expected);
}

/// 输入与输出长度不一致时拒绝；空行两边都放行（逐元素，没有分母）。
#[test]
fn scale_rejects_mismatched_length_and_allows_empty() {
    let mut out = vec![0.0; 3];
    assert!(forward(&[1.0, 2.0], &mut out).is_err(), "输出比输入长");
    let mut empty = vec![];
    forward(&[], &mut empty).expect("空行逐元素放行");
}

/// 饱和与重复边界：零、正负大值、重复元素，结果与参考一致且有限。
#[test]
fn scale_saturating_and_repeated_inputs_match_reference() {
    let cases: &[&[f32]] = &[
        &[0.0, 0.0],
        &[1.0, 1.0, 1.0],
        &[-1.0, -2.0, -3.0],
        &[f32::MAX, f32::MAX],
        &[f32::MIN, f32::MIN],
    ];
    for (index, a) in cases.iter().enumerate() {
        let mut sample = crate::unary::sample::<Scale>(a.len(), 1);
        sample.buffers[0].values = crate::model::Values::F32(a.to_vec());
        let expected = crate::unary::reference::<Scale>(&sample).expect("参考可计算");
        let mut out = vec![0.0; a.len()];
        forward(a, &mut out).expect("等长");
        assert_eq!(out, expected, "用例 {index}");
        assert!(out.iter().all(|v| v.is_finite()), "用例 {index}：结果有限");
    }
}
