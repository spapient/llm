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

#[test]
fn cpu_allocator_allocates_zeroed_buffer() {
    let mut alloc = CpuAllocator::new();
    let buffer = alloc.alloc(5).expect("分配");
    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.bytes(), &[0; 5]);
    alloc.free(buffer);
}

#[test]
fn allocator_buffer_is_writable() {
    let mut alloc = CpuAllocator::new();
    let mut buffer = alloc.alloc(3).expect("分配");
    buffer.bytes_mut().copy_from_slice(&[1, 2, 3]);
    assert_eq!(buffer.bytes(), &[1, 2, 3]);
    let bytes = buffer.into_bytes();
    assert_eq!(bytes, vec![1, 2, 3]);
}

#[test]
fn zero_allocation_yields_an_empty_buffer() {
    let mut alloc = CpuAllocator::new();
    let buffer = alloc.alloc(0).expect("零长度");
    assert!(buffer.is_empty());
}

#[test]
fn copy_into_matches_lengths_and_rejects_mismatch() {
    let alloc = CpuAllocator::new();
    let mut dst = vec![0u8; 4];
    alloc.copy_into(&[1, 2, 3, 4], &mut dst).expect("等长拷贝");
    assert_eq!(dst, [1, 2, 3, 4]);
    let err = alloc.copy_into(&[1, 2], &mut dst).unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
}

#[test]
fn an_unaffordable_allocation_is_an_error_not_an_abort() {
    let mut alloc = CpuAllocator::new();
    let error = alloc.alloc(usize::MAX).expect_err("地址空间装不下");
    assert_eq!(error.code(), Code::Invalid);
}
