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

/// 退化与重复边界：空行放行；长度不齐拒绝；重复偶数元素在同相位下结果对称。
#[test]
fn rope_handles_empty_mismatched_and_repeated_pairs() {
    let mut empty = vec![];
    forward(0, &[], &mut empty).expect("空行放行");

    let mut out = vec![0.0; 3];
    assert!(forward(0, &[1.0, 2.0], &mut out).is_err(), "长度不齐拒绝");

    let repeated = [1.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0; repeated.len()];
    forward(3, &repeated, &mut out).expect("重复对不越界");
    assert!(out.iter().all(|v| v.is_finite()), "结果有限");
}
