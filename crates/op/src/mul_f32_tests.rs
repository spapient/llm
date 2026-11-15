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
