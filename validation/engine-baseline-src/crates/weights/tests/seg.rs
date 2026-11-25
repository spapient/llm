//! `seg` 模块的单元测试。
use super::*;

#[test]
fn a_short_buf_is_rejected_by_length() {
    let error = Segment::new("w", Wire::F32, vec![2], None, vec![0u8; 4]).expect_err("长度不符");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("8 字节"), "{error}");
}

#[test]
fn an_empty_name_is_rejected() {
    let error = Segment::new("", Wire::F32, vec![1], None, vec![0u8; 4]).expect_err("空名");

    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn a_zero_element_shape_accepts_an_empty_buf() {
    let segment = Segment::new("empty", Wire::F32, vec![0], None, Vec::new()).expect("零元素段");

    assert_eq!(segment.numel(), 0);

    assert_eq!(segment.expected_bytes(), 0);
}

#[test]
fn an_overlarge_rank_is_rejected_before_allocation() {
    let mut cur = Cursor::new(&[9, 0, 0, 0, 0]);

    let error = decode_shape(&mut cur, MAX_RANK + 1).expect_err("rank 超限必须报错");

    assert!(error.to_string().contains("维数"), "{error}");
}

#[test]
fn an_overflowing_shape_saturates_instead_of_wrapping_to_zero() {
    // 饱和乘积不会回绕成 0；短缓冲必须走长度不符错误，而不是被当成零元素放行。

    let error = Segment::new("huge", Wire::F32, vec![u64::MAX, 2], None, vec![0u8; 4])
        .expect_err("溢出形状");

    assert_eq!(error.code(), Code::Invalid);

    assert!(error.to_string().contains("字节"), "{error}");
}
