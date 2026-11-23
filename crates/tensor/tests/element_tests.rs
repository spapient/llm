//! 元素类型映射：每个有原生标量的 `K` 恰好钉一个 `DType`。
//!
//! 元素类型的唯一事实源是 `dtype::DType`；[`Element`] 只是把 Rust 标量接到那张表上。这里逐个
//! 钉住映射，防止某个 impl 漂移到相邻精度；亚字节精度没有原生标量，不在此列（由 `DType` 位宽表
//! 表达，见 `TestKit` 的 `no_storage` 用例）。

use crate::{DType, Element};

#[test]
fn bool_maps_to_bool() {
    assert_eq!(bool::DTYPE, DType::Bool);
}

#[test]
fn unsigned_integers_map_to_their_widths() {
    assert_eq!(u8::DTYPE, DType::U8);
    assert_eq!(u16::DTYPE, DType::U16);
    assert_eq!(u32::DTYPE, DType::U32);
    assert_eq!(u64::DTYPE, DType::U64);
}

#[test]
fn signed_integers_map_to_their_widths() {
    assert_eq!(i8::DTYPE, DType::I8);
    assert_eq!(i16::DTYPE, DType::I16);
    assert_eq!(i32::DTYPE, DType::I32);
    assert_eq!(i64::DTYPE, DType::I64);
}

#[test]
fn floats_map_to_their_widths() {
    assert_eq!(f32::DTYPE, DType::F32);
    assert_eq!(f64::DTYPE, DType::F64);
    assert_eq!(half::f16::DTYPE, DType::F16);
    assert_eq!(half::bf16::DTYPE, DType::BF16);
}

#[test]
fn sub_byte_precisions_have_no_scalar_element() {
    // 亚字节精度是 `DType` 位宽表的一等公民，但没有原生标量，因此不进 `Element`。
    // 它们的字节数由 `DType::bytes_for` 位级进位表达，而不是硬造一个打包标量。
    for dtype in [DType::U1, DType::U2, DType::U4, DType::I4, DType::F4] {
        assert!(dtype.bits() < 8, "{dtype:?} 应是亚字节精度");
    }
}
