use super::*;

#[test]
fn sum_matches_the_scalar_reference() {
    let sample = crate::reduce::sample::<Sum>(128, 1);
    let expected = crate::reduce::reference::<Sum>(&sample).expect("参考可计算");
    let mut out = vec![0.0];
    forward(sample.f32("a").expect("a"), &mut out).expect("单元素输出");
    assert_eq!(out, expected);
}
