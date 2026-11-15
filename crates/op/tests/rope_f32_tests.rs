use super::*;

#[test]
fn rope_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(7, sample.f32("x").expect("x"), &mut out).expect("等长");
    assert_eq!(out, expected);
}

#[test]
fn an_odd_length_passes_its_tail_through() {
    let out = rotate(&[1.0, 2.0, 3.0], 0.0).expect("奇数长度不报错");
    assert_eq!(out[2], 3.0, "末尾无法配对的元素原样透传");
}
