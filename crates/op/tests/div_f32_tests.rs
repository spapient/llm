use super::*;

crate::elementwise::test_support::elementwise_pair_tests!(Div, forward);

/// 除零按 IEEE-754：符号决定 `±inf`，不与 NaN 比较（NaN != NaN）。
#[test]
fn division_by_zero_matches_ieee() {
    let mut out = vec![0.0; 2];
    forward(&[1.0, -1.0], &[0.0, 0.0], &mut out).expect("等长");
    assert!(out[0].is_infinite() && out[0].is_sign_positive(), "1/0 = +inf");
    assert!(out[1].is_infinite() && out[1].is_sign_negative(), "-1/0 = -inf");
}
