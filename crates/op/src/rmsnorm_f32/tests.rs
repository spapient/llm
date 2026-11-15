use super::*;

#[test]
fn rmsnorm_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(
        sample.f32("x").expect("x"),
        sample.f32("gamma").expect("gamma"),
        &mut out,
    )
    .expect("等长");
    assert_eq!(out, expected);
}
