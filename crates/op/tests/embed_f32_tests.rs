use super::*;

#[test]
fn embed_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(
        sample.f32("idx").expect("idx"),
        sample.f32("table").expect("table"),
        &mut out,
    )
    .expect("等长");
    assert_eq!(out, expected);
}
