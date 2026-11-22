//! 形状描述的不变量：元素数是各轴之积、字节数由元素宽度派生、溢出饱和。
//!
//! 这些用例原本挂在 `prop_shape` 的「按行主序枚举全部多下标」性质里，与视图代数同生共死。
//! 视图代数删除后剩下的是形状本身的三条契约，改为直接钉住——覆盖面没有变窄，
//! 只是不再经过一条已被删除的布局路径。

use crate::{DType, Shape, product};

#[test]
fn the_element_count_is_the_product_of_the_axes() {
    let shape = Shape::new([2, 3, 4]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.numel(), 24);
    assert_eq!(shape.numel(), product(shape.dims()));
    assert_eq!(shape.element_bytes(DType::F32), 96);
}

#[test]
fn a_scalar_shape_holds_one_element() {
    let shape = Shape::new([]);
    assert_eq!(shape.rank(), 0);
    assert_eq!(shape.numel(), 1, "空轴序列是标量，不是零个元素");
    assert_eq!(shape.element_bytes(DType::F32), 4);
}

#[test]
fn sub_byte_precisions_round_the_byte_count_up() {
    // 6 个元素：f16/bf16 各 2 字节，u4/i4 只占 24 位 = 3 字节。
    // 字节数小于 `numel` 是亚字节精度的正确结果，不是漏乘。
    let shape = Shape::new([2, 3]);
    assert_eq!(shape.element_bytes(DType::F16), 12);
    assert_eq!(shape.element_bytes(DType::BF16), 12);
    assert_eq!(shape.element_bytes(DType::U8), 6);
    assert_eq!(shape.element_bytes(DType::I8), 6);
    assert_eq!(shape.element_bytes(DType::U4), 3);
    assert_eq!(shape.element_bytes(DType::I4), 3);
}

#[test]
fn an_overflowing_shape_saturates_instead_of_wrapping() {
    // 轴长可能来自不受信任的权重文件头：回绕会给出一个偏小的元素数，骗过下游的容量检查。
    let shape = Shape::new([usize::MAX, 2]);
    assert_eq!(shape.numel(), usize::MAX, "饱和到上限，不回绕");
}

#[test]
fn an_axis_length_is_read_by_position() {
    let shape = Shape::new([5, 7]);
    assert_eq!(shape.dim(0), Some(5));
    assert_eq!(shape.dim(1), Some(7));
    assert_eq!(shape.dim(2), None);
}
