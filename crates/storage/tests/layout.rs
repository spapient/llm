//! `Layout` 与 `remap` 的单元测试：行↔列主序、单位变换、形状不匹配与空形状。

use super::*;

#[test]
fn row_major_is_the_tensor_default_and_col_major_is_its_mirror() {
    let shape = Shape::new([2, 3]);
    assert_eq!(
        Layout::new(shape.clone(), Order::RowMajor)
            .strides()
            .as_slice(),
        [3, 1]
    );
    assert_eq!(
        Layout::new(shape, Order::ColMajor).strides().as_slice(),
        [1, 2]
    );
}

#[test]
fn remap_transposes_a_row_major_buffer() {
    let shape = Shape::new([2, 3]);
    let from = Layout::new(shape.clone(), Order::RowMajor);
    let to = Layout::new(shape.clone(), Order::ColMajor);
    let moved = remap(&[0u8, 1, 2, 3, 4, 5], &from, &to).expect("形状一致");
    assert_eq!(moved, [0, 3, 1, 4, 2, 5]);
    let back = remap(&moved, &to, &from).expect("回到行主序");
    assert_eq!(back, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn an_identity_layout_change_is_a_copy() {
    let shape = Shape::new([4]);
    let layout = Layout::new(shape, Order::RowMajor);
    let moved = remap(&[1.0f32, 2.0, 3.0, 4.0], &layout, &layout).expect("相同布局");
    assert_eq!(moved, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn mismatched_shapes_and_short_buffers_are_rejected() {
    let from = Layout::new(Shape::new([2, 3]), Order::RowMajor);
    let to = Layout::new(Shape::new([3, 2]), Order::ColMajor);
    assert_eq!(
        remap(&[0u8; 6], &from, &to).unwrap_err().code(),
        Code::Invalid
    );
    let same = Layout::new(Shape::new([2, 3]), Order::ColMajor);
    assert_eq!(
        remap(&[0u8; 5], &from, &same).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn an_empty_shape_maps_to_nothing() {
    let shape = Shape::new([0, 3]);
    let from = Layout::new(shape.clone(), Order::RowMajor);
    let to = Layout::new(shape, Order::ColMajor);
    assert_eq!(remap::<u8>(&[], &from, &to), Ok(Vec::new()));
}
