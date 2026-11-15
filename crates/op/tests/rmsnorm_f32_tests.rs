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

#[test]
fn rmsnorm_rejects_empty_input() {
    let mut out = vec![];
    assert!(
        forward(&[], &[], &mut out).is_err(),
        "空行没有可统计的均方根"
    );
    assert!(reference(&sample(0, 1)).is_err(), "参考同样拒绝空行");
}
