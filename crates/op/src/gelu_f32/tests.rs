use super::*;

#[test]
fn gelu_matches_the_scalar_reference() {
    let sample = crate::unary::sample::<Gelu>(128, 1);
    let expected = crate::unary::reference::<Gelu>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    forward(a, &mut out).expect("缓冲等长");
    assert_eq!(out, expected);
}

#[test]
fn erf_matches_known_values() {
    assert!((erf(0.0) - 0.0).abs() < 1e-6);
    assert!((erf(1.0) - 0.842_700_8).abs() < 1.5e-6);
    assert!((erf(-1.0) + 0.842_700_8).abs() < 1.5e-6);
}
