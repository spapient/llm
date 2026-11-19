use super::*;

crate::unary::test_support::unary_pair_tests!(Gelu, forward);

/// 精确 erf 的已知值钉住：Abramowitz-Stegun 7.1.26 闭式，两侧绝对误差 < 1.5e-7。
#[test]
fn erf_matches_known_values() {
    assert!((erf(0.0) - 0.0).abs() < 1e-6);
    assert!((erf(1.0) - 0.842_700_8).abs() < 1.5e-6);
    assert!((erf(-1.0) + 0.842_700_8).abs() < 1.5e-6);
}
