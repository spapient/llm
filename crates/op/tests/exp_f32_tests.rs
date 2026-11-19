use super::*;

crate::unary::test_support::unary_pair_tests!(Exp, forward);

/// 边界：exp(0)=1、大负数下溢为 0、exp(1)=e。
#[test]
fn exp_edges_match_ieee() {
    let mut out = vec![0.0; 3];
    forward(&[0.0, -1000.0, 1.0], &mut out).expect("等长");
    assert_eq!(out[0], 1.0, "exp(0)=1");
    assert_eq!(out[1], 0.0, "exp(-1000) 下溢为 0");
    assert!((out[2] - std::f32::consts::E).abs() < 1e-6, "exp(1)=e");
}
