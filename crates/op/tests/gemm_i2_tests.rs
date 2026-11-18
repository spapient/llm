use super::*;

#[test]
fn gemm_i2_matches_the_scalar_reference() {
    let sample = sample(16, 12);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 16 * 12];
    forward(
        sample.u8("a").expect("a"),
        sample.u8("b").expect("b"),
        16,
        12,
        &mut out,
    )
    .expect("尺寸符");
    assert_eq!(out, expected);
}
