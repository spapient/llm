use super::*;

#[test]
fn silu_matches_the_scalar_reference() {
    let sample = crate::unary::sample::<Silu>(128, 1);
    let expected = crate::unary::reference::<Silu>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    forward(a, &mut out).expect("缓冲等长");
    assert_eq!(out, expected);
}
