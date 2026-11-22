use crate::typed::{Shape, Strides, Tensor};
use crate::{DType, Order};
use diagnostic::Code;

#[test]
fn typed_tensor_is_contiguous_and_typed() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.numel(), 6);
    assert!(tensor.is_contiguous());
    assert_eq!(tensor.shape().dims(), &[2, 3]);
    assert_eq!(tensor.strides().steps(), &[3, 1]);
    assert_eq!(tensor.as_slice(), data.as_slice());
    assert_eq!(tensor.at([1, 2]), Some(6.0));
    assert_eq!(tensor.at([2, 0]), None, "行越界");
}

#[test]
fn typed_tensor_rejects_mismatched_buf() {
    let error = Tensor::<2, f32>::new([2, 3], &[1.0; 5]).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn transpose_is_zero_copy_metadata() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    let transposed = tensor.transpose(0, 1).expect("轴合法");
    assert_eq!(transposed.shape().dims(), &[3, 2]);
    assert_eq!(transposed.strides().steps(), &[1, 3]);
    assert!(!transposed.is_contiguous());
    assert_eq!(transposed.at([1, 0]), Some(2.0), "原 (0,1) 变 (1,0)");
    assert_eq!(transposed.at([2, 0]), Some(3.0), "原 (0,2) 变 (2,0)");
    assert!(tensor.transpose(0, 2).is_none(), "轴越界");
}

#[test]
fn permute_reorders_axes() {
    let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let tensor = Tensor::new([2, 3, 4], data.as_slice()).expect("形状与缓冲匹配");
    let permuted = tensor.permute([2, 0, 1]).expect("合法排列");
    assert_eq!(permuted.shape().dims(), &[4, 2, 3]);
    assert_eq!(permuted.at([1, 0, 2]), tensor.at([0, 2, 1]));
    assert!(tensor.permute([0, 0, 1]).is_none(), "重复轴不是排列");
}

#[test]
fn reshape_preserves_element_count() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    let flat = tensor.reshape([6]).expect("等元素数");
    assert_eq!(flat.at([3]), Some(4.0));
    assert!(tensor.reshape([3, 3]).is_none(), "元素数不符");
}

#[test]
fn reshape_needs_contiguous_layout() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    let transposed = tensor.transpose(0, 1).expect("转置");
    assert!(transposed.reshape([6]).is_none());
}

#[test]
fn strided_rejects_out_of_reach_layout() {
    let shape = Shape::new([2, 3]);
    let strides = Strides::new([6, 2]);
    let error = Tensor::<2, f32>::strided(shape, strides, &[0.0; 6]).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn zero_element_view_needs_empty_buf() {
    let error = Tensor::<2, f32>::new([0, 3], &[1.0]).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn row_slices_a_contiguous_2d_view() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    assert_eq!(tensor.row(1), Some(&[4.0, 5.0, 6.0][..]));
    assert_eq!(tensor.row(2), None);
}

#[test]
fn shape_layout_algebra_matches_runtime_semantics() {
    let shape = Shape::<3>::new([2, 3, 4]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.numel(), 24);
    let transposed = shape.transpose(0, 2).expect("轴合法");
    assert_eq!(transposed.dims(), &[4, 3, 2]);
    let permuted = shape.permute([1, 2, 0]).expect("合法排列");
    assert_eq!(permuted.dims(), &[3, 4, 2]);
    let reshaped = shape.reshape([4, 6]).expect("等元素数");
    assert_eq!(reshaped.dims(), &[4, 6]);
    assert!(shape.reshape([3, 3]).is_none(), "元素数不符");
    assert_eq!(shape.element_bytes(DType::F32), 96);
    assert_eq!(Shape::<2>::new([2, 4]).element_bytes(DType::U4), 4);
    assert_eq!(Strides::contiguous(&shape).steps(), &[12, 4, 1]);
    assert_eq!(
        Strides::ordered(&shape, Order::ColMajor).steps(),
        &[1, 2, 6]
    );
}
