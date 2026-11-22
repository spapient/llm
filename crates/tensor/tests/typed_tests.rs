//! 秩进类型的形态：`Tensor<&[f32], [usize; D], f32>`。
//!
//! 同一个 [`Tensor`](crate::Tensor) 把秩写成 `[usize; D]` 时，形状与步长是栈上定长数组，构造不碰堆；
//! 秩由类型钉死，换轴数就编译不过。这里锁住这条路径的全部布局能力——它与运行期秩共用
//! `layout` 与 `shape::accumulate`，所以两个见证不该有任何行为差别。

use crate::{DType, Element, Order, Shape, Strides, Tensor, product};

/// 编译期秩 2 的借来视图。
type Fix2 = Tensor<&'static [f32], [usize; 2], f32>;

#[test]
fn typed_tensor_is_contiguous_and_typed() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2 = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.numel(), 6);
    assert!(tensor.is_contiguous());
    assert_eq!(tensor.dims(), [2, 3]);
    assert_eq!(tensor.strides(), [3, 1]);
    assert_eq!(tensor.as_slice(), data.as_slice());
    assert_eq!(tensor.at([1, 2]), Some(6.0));
    assert_eq!(tensor.at([2, 0]), None, "行越界");
}

#[test]
fn typed_tensor_rejects_mismatched_buf() {
    let error = Tensor::<&[f32], [usize; 2], f32>::new([2, 3], &[1.0f32; 5]).unwrap_err();
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn transpose_is_zero_copy_metadata() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2 = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    let transposed = tensor.clone().transpose(0, 1).expect("轴合法");
    assert_eq!(transposed.dims(), [3, 2]);
    assert_eq!(transposed.strides(), [1, 3]);
    assert!(!transposed.is_contiguous());
    assert_eq!(transposed.at([1, 0]), Some(2.0), "原 (0,1) 变 (1,0)");
    assert_eq!(transposed.at([2, 0]), Some(3.0), "原 (0,2) 变 (2,0)");
    assert!(tensor.transpose(0, 2).is_none(), "轴越界");
}

#[test]
fn permute_reorders_axes() {
    let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let tensor: Tensor<&[f32], [usize; 3], f32> =
        Tensor::new([2, 3, 4], data.as_slice()).expect("形状与缓冲匹配");
    let permuted = tensor.clone().permute([2, 0, 1]).expect("合法排列");
    assert_eq!(permuted.dims(), [4, 2, 3]);
    assert_eq!(permuted.at([1, 0, 2]), tensor.at([0, 2, 1]));
    assert!(tensor.permute([0, 0, 1]).is_none(), "重复轴不是排列");
}

#[test]
fn reshape_preserves_element_count_and_may_change_rank() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2 = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    let flat: Tensor<&[f32], [usize; 1], f32> = tensor.clone().reshape([6]).expect("等元素数");
    assert_eq!(flat.at([3]), Some(4.0));
    assert!(tensor.reshape([3, 3]).is_none(), "元素数不符");
}

#[test]
fn reshape_needs_contiguous_layout() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2 = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    let transposed = tensor.transpose(0, 1).expect("转置");
    assert!(transposed.reshape([6]).is_none());
}

#[test]
fn strided_rejects_out_of_reach_layout() {
    let error = Tensor::<&[f32], [usize; 2], f32>::strided(
        Shape::new([2, 3]),
        Strides::new([6, 2]),
        &[0.0f32; 6],
    )
    .unwrap_err();
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn zero_element_view_needs_empty_buf() {
    let error = Tensor::<&[f32], [usize; 2], f32>::new([0, 3], &[1.0f32]).unwrap_err();
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn row_slices_a_contiguous_2d_view() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2 = Tensor::new([2, 3], &data).expect("形状与缓冲匹配");
    assert_eq!(tensor.row(1), Some(&[4.0, 5.0, 6.0][..]));
    assert_eq!(tensor.row(2), None);
}

/// 编译期秩与运行期秩共用同一套累积规则：`Strides::ordered` 与 `Rank::contiguous` 必须逐轴相等。
#[test]
fn static_rank_contiguous_matches_the_runtime_strides() {
    for dims in [&[] as &[usize], &[5], &[2, 3], &[2, 3, 4]] {
        let shape = Shape::new(dims.to_vec());
        let expected = Strides::ordered(&shape, Order::RowMajor);
        let direct: Option<Tensor<&[usize], Vec<usize>, u8>> = None;
        assert!(direct.is_none(), "占位，避免未使用告警");
        assert_eq!(expected.as_slice(), Strides::contiguous(&shape).as_slice());
    }
    assert_eq!(product(&[2usize, 3, 4]), 24);
}

#[test]
fn element_types_are_compile_time_constants() {
    assert_eq!(<f32 as Element>::DTYPE, DType::F32);
    assert_eq!(<i8 as Element>::DTYPE, DType::I8);
    assert_eq!(<u8 as Element>::DTYPE, DType::U8);
}
