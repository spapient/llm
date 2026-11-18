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

/// 仿射参数负例：零标度、非有限标度、越出 4 位码域的零点都在解包前拦下。
#[test]
fn dequant_i4_rejects_zero_scale_and_out_of_domain_zero() {
    let q = [0x52u8];
    let mut out = [0.0; 2];

    assert!(forward(0.0, ZERO, &q, &mut out).is_err(), "零标度拒绝");
    assert!(
        forward(SCALE, f32::INFINITY, &q, &mut out).is_err(),
        "非有限零点拒绝"
    );
    assert!(
        forward(SCALE, 16.0, &q, &mut out).is_err(),
        "零点越出 0..15 码域"
    );
    assert!(
        forward(SCALE, -1.0, &q, &mut out).is_err(),
        "负零点越出码域"
    );
}
