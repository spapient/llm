use super::*;

/// 参考对拍：默认夹具 `b = 0.5 + 0.25·i` 恒正，不含除零。
#[test]
fn matches_the_scalar_reference() {
    let sample = crate::elementwise::sample::<Div>(128, 1);
    let expected = crate::elementwise::reference::<Div>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    let b = sample.f32("b").expect("b 存在");
    forward(a, b, &mut out).expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn rejects_mismatched_lengths_and_allows_empty() {
    let mut out = vec![0.0; 2];
    assert!(forward(&[1.0, 2.0], &[1.0], &mut out).is_err(), "b 与 out 不等");
    let mut empty = vec![];
    forward(&[], &[], &mut empty).expect("空输入逐元素放行");
}

/// 除零按 IEEE-754：符号决定 `±inf`，不与 NaN 比较（NaN != NaN）。
#[test]
fn division_by_zero_matches_ieee() {
    let mut out = vec![0.0; 2];
    forward(&[1.0, -1.0], &[0.0, 0.0], &mut out).expect("等长");
    assert!(out[0].is_infinite() && out[0].is_sign_positive(), "1/0 = +inf");
    assert!(out[1].is_infinite() && out[1].is_sign_negative(), "-1/0 = -inf");
}
