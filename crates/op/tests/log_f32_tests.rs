use super::*;

/// 参考对拍：夹具已覆盖为正数（`0.5 + 0.25·i`），不会触发 `log(负数)` 的 NaN。
#[test]
fn matches_the_scalar_reference() {
    let sample = crate::unary::sample::<Log>(128, 1);
    let expected = crate::unary::reference::<Log>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    forward(a, &mut out).expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn rejects_mismatched_length_and_allows_empty() {
    let mut out = vec![0.0; 3];
    assert!(forward(&[1.0, 2.0], &mut out).is_err(), "输出比输入长");
    let mut empty = vec![];
    forward(&[], &mut empty).expect("空行逐元素放行");
}

/// 边界：ln(1)=0、ln(e)=1。
#[test]
fn log_edges_match_ieee() {
    let mut out = vec![0.0; 2];
    forward(&[1.0, std::f32::consts::E], &mut out).expect("等长");
    assert_eq!(out[0], 0.0, "ln(1)=0");
    assert!((out[1] - 1.0).abs() < 1e-6, "ln(e)=1");
}
