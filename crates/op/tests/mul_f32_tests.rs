use super::*;

#[test]
fn mul_matches_the_scalar_reference() {
    let sample = crate::elementwise::sample::<Mul>(128, 1);
    let expected = crate::elementwise::reference::<Mul>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    let b = sample.f32("b").expect("b 存在");
    forward(a, b, &mut out).expect("缓冲等长");
    assert_eq!(out, expected);
}

/// 三缓冲非对齐：a/b 不等、a/out 不等各自拒绝；空输入逐元素放行。
#[test]
fn mul_rejects_each_mismatched_pair_and_allows_empty() {
    let mut out = vec![0.0; 2];
    assert!(
        forward(&[1.0], &[1.0, 2.0], &mut out).is_err(),
        "a 与 b 不等"
    );
    assert!(
        forward(&[1.0, 2.0], &[1.0], &mut out).is_err(),
        "b 与 out 不等"
    );
    let mut empty = vec![];
    forward(&[], &[], &mut empty).expect("空输入逐元素放行");
}

/// 饱和与重复边界：零、同号大值、重复元素与符号翻转行，结果与参考一致。
#[test]
fn mul_saturating_and_repeated_inputs_match_reference() {
    let cases: &[(&[f32], &[f32])] = &[
        (&[0.0, 0.0], &[0.0, 0.0]),
        (&[1.0, 1.0], &[2.0, 2.0]),
        (&[f32::MAX, f32::MIN], &[0.0, 0.0]),
        (&[f32::MAX, f32::MAX], &[f32::MIN, f32::MIN]),
        (&[3.0, -1.0], &[-3.0, 1.0]),
    ];
    for (index, (a, b)) in cases.iter().enumerate() {
        let mut sample = crate::elementwise::sample::<Mul>(a.len(), 1);
        sample.buffers[0].values = crate::model::Values::F32(a.to_vec());
        sample.buffers[1].values = crate::model::Values::F32(b.to_vec());
        let expected = crate::elementwise::reference::<Mul>(&sample).expect("参考可计算");
        let mut out = vec![0.0; a.len()];
        forward(a, b, &mut out).expect("等长");
        assert_eq!(out, expected, "用例 {index}");
    }
}
