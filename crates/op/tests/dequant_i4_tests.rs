use super::*;

#[test]
fn dequant_matches_the_scalar_reference() {
    let sample = sample(128, 1);
    let expected = reference(&sample).expect("参考可计算");
    let mut out = vec![0.0; 128];
    forward(SCALE, ZERO, sample.u8("q").expect("q"), &mut out).expect("等长");
    assert_eq!(out, expected);
}

/// 退化与边界：空打包行放行；输出长度不是字节数 2 倍时拒绝；极值 nibble 不产生 NaN。
#[test]
fn dequant_i4_handles_empty_mismatched_and_saturated_bytes() {
    let mut empty = vec![];
    forward(SCALE, ZERO, &[], &mut empty).expect("空行放行");

    let mut wrong = vec![0.0; 3];
    assert!(
        forward(SCALE, ZERO, &[0x12, 0x34], &mut wrong).is_err(),
        "输出长度必须是字节数的 2 倍"
    );

    let saturated = [0x00u8, 0xFF];
    let mut out = vec![0.0; saturated.len() * 2];
    forward(SCALE, ZERO, &saturated, &mut out).expect("极值字节不越界");
    assert!(out.iter().all(|v| v.is_finite()), "结果有限");
}
