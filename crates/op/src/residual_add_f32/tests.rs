use super::*;

#[test]
fn residual_add_matches_the_scalar_reference() {
    let sample = crate::elementwise::sample::<Residual>(128, 1);
    let expected = crate::elementwise::reference::<Residual>(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(
        sample.f32("a").expect("a"),
        sample.f32("b").expect("b"),
        &mut out,
    )
    .expect("等长");
    assert_eq!(out, expected);
}
