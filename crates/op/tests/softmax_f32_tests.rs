use super::*;

#[test]
fn softmax_matches_the_scalar_reference_and_sums_to_one() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let total: f32 = expected.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax 输出和为 1");
    let mut out = vec![0.0; 128];
    forward(sample.f32("x").expect("x"), &mut out).expect("等长");
    assert_eq!(out, expected);
}
