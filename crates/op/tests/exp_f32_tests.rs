use super::*;

/// 参考对拍：默认夹具 `a = -4 + 0.25·i`，exp 后全部有限（最大值 exp(27.75) 不上溢）。
#[test]
fn matches_the_scalar_reference() {
    let sample = crate::unary::sample::<Exp>(128, 1);
    let expected = crate::unary::reference::<Exp>(&sample).expect("参考可计算");
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

/// 边界：exp(0)=1、大负数下溢为 0、exp(1)=e。
#[test]
fn exp_edges_match_ieee() {
    let mut out = vec![0.0; 3];
    forward(&[0.0, -1000.0, 1.0], &mut out).expect("等长");
    assert_eq!(out[0], 1.0, "exp(0)=1");
    assert_eq!(out[1], 0.0, "exp(-1000) 下溢为 0");
    assert!((out[2] - std::f32::consts::E).abs() < 1e-6, "exp(1)=e");
}
