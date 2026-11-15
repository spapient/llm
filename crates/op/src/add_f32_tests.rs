use super::*;

#[test]
fn add_matches_the_scalar_reference() {
    let sample = crate::elementwise::sample::<Add>(128, 1);
    let expected = crate::elementwise::reference::<Add>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    let a = sample.f32("a").expect("a 存在");
    let b = sample.f32("b").expect("b 存在");
    forward(a, b, &mut out).expect("缓冲等长");
    assert_eq!(out, expected);
}

#[test]
fn mismatched_lengths_are_rejected() {
    let mut out = vec![0.0; 2];
    assert!(forward(&[1.0, 2.0], &[1.0], &mut out).is_err());
}
