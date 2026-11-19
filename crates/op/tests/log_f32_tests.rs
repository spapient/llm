use super::*;

crate::unary::test_support::unary_pair_tests!(Log, forward);

/// 边界：ln(1)=0、ln(e)=1。
#[test]
fn log_edges_match_ieee() {
    let mut out = vec![0.0; 2];
    forward(&[1.0, std::f32::consts::E], &mut out).expect("等长");
    assert_eq!(out[0], 0.0, "ln(1)=0");
    assert!((out[1] - 1.0).abs() < 1e-6, "ln(e)=1");
}
