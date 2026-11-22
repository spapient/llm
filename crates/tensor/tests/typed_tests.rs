//! 秩进类型的形态：`Tensor<&[f32], [usize; D], f32>`——形状与步长是栈上定长数组，构造不碰堆，
//! 秩由类型钉死，换轴数就编译不过。
//!
//! 它与运行期秩共用 `layout` 与 `shape::accumulate`，所以两个见证不该有任何行为差别；这里既锁住
//! 编译期秩自己的全套布局能力，也锁住"两族算出来的步长逐轴相等"。

use crate::{DType, Element, Order, Shape, Tensor, product};

/// 编译期秩 2 的借来视图。
type Fix2<'a> = Tensor<&'a [f32], [usize; 2], f32>;

#[test]
fn typed_tensor_is_contiguous_and_typed() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
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
    let error = Tensor::<&[f32], [usize; 2], f32>::new([2, 3], &[1.0f32; 5][..]).unwrap_err();
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn transpose_is_zero_copy_metadata() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
    let transposed = tensor.clone().transpose(0, 1).expect("轴合法");
    assert_eq!(transposed.dims(), [3, 2]);
    assert_eq!(transposed.strides(), [1, 3]);
    assert!(!transposed.is_contiguous());
    assert_eq!(transposed.at([1, 0]), Some(2.0), "原 (0,1) 变 (1,0)");
    assert_eq!(transposed.at([2, 0]), Some(3.0), "原 (0,2) 变 (2,0)");
    assert!(tensor.transpose(0, 2).is_none(), "轴越界");
    assert!(
        std::ptr::eq(transposed.as_slice().as_ptr(), data.as_ptr()),
        "转置是元数据重排，缓冲一个字节都没动"
    );
}

#[test]
fn permute_reorders_axes() {
    let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let tensor: Tensor<&[f32], [usize; 3], f32> =
        Tensor::new([2, 3, 4], data.as_slice()).expect("形状与缓冲匹配");
    let permuted = tensor.clone().permute([2, 0, 1]).expect("合法排列");
    assert_eq!(permuted.dims(), [4, 2, 3]);
    assert_eq!(permuted.strides(), [1, 12, 4]);
    assert_eq!(permuted.at([1, 0, 2]), tensor.at([0, 2, 1]));
    assert!(
        tensor.clone().permute([0, 0, 1]).is_none(),
        "重复轴不是排列"
    );
    assert!(tensor.permute([0, 1]).is_none(), "轴数不符不是排列");
}

#[test]
fn reshape_preserves_element_count_and_may_change_rank() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
    let flat: Tensor<&[f32], [usize; 1], f32> = tensor.clone().reshape([6]).expect("等元素数");
    assert_eq!(flat.at([3]), Some(4.0));
    assert!(tensor.reshape::<[usize; 2]>([3, 3]).is_none(), "元素数不符");
}

#[test]
fn reshape_needs_contiguous_layout() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
    let transposed = tensor.transpose(0, 1).expect("转置");
    assert!(transposed.reshape::<[usize; 1]>([6]).is_none());
}

#[test]
fn strided_rejects_out_of_reach_layout() {
    let error = Tensor::<&[f32], [usize; 2], f32>::strided([2, 3], [6, 2], &[0.0f32; 6])
        .expect_err("最高偏移越界");
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn zero_element_view_needs_empty_buf() {
    let error = Tensor::<&[f32], [usize; 2], f32>::new([0, 3], &[1.0f32][..]).unwrap_err();
    assert_eq!(error.code(), diagnostic::Code::Invalid);
}

#[test]
fn row_slices_a_contiguous_2d_view() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
    assert_eq!(tensor.row(1), Some(&[4.0, 5.0, 6.0][..]));
    assert_eq!(tensor.row(2), None);
}

/// 编译期秩与运行期秩共用同一套累积规则：同一个形状在两个见证下必须算出逐轴相等的步长。
#[test]
fn static_rank_contiguous_matches_the_runtime_strides() {
    let data = [0.0f32; 24];
    let fixed: Tensor<&[f32], [usize; 3], f32> =
        Tensor::new([2, 3, 4], data.as_slice()).expect("形状与缓冲匹配");
    let runtime = Shape::new([2, 3, 4]).strides(Order::RowMajor);
    assert_eq!(fixed.strides(), runtime.as_slice());
    assert_eq!(fixed.numel(), product(&[2usize, 3, 4]));
}

#[test]
fn element_types_are_compile_time_constants() {
    assert_eq!(<f32 as Element>::DTYPE, DType::F32);
    assert_eq!(<i8 as Element>::DTYPE, DType::I8);
    assert_eq!(<u8 as Element>::DTYPE, DType::U8);
}

#[test]
fn a_shape_view_keeps_the_compile_time_rank() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
    // 返回类型写死成 Fix2：起点前移若偷偷换成 Vec 见证，这里就编译不过。
    let tail: Fix2<'_> = tensor.clone().narrow(0, 1, 1).expect("末行");
    assert_eq!(tail.dims(), [1, 3]);
    assert_eq!(tail.strides(), [3, 1]);
    assert_eq!(tail.base(), 3);
    let block: Fix2<'_> = tensor.slice([0..1, 1..3]).expect("首行后两列");
    assert_eq!(block.at([0, 1]), Some(3.0));
}

#[test]
fn rank_changing_views_need_the_target_witness_and_the_length_is_checked() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");
    let row = tensor.clone().select::<[usize; 1]>(0, 1).expect("第 1 行");
    assert_eq!(row.dims(), [3]);
    assert_eq!(row.base(), 3);
    assert!(
        tensor.clone().select::<[usize; 2]>(0, 1).is_none(),
        "去掉一轴后是秩 1，装不进秩 2 的定长数组"
    );

    let lifted: Tensor<&[f32], [usize; 3], f32> = tensor
        .clone()
        .unsqueeze::<[usize; 3]>(1)
        .expect("插入单例轴");
    assert_eq!(lifted.dims(), [2, 1, 3]);
    let back: Fix2<'_> = lifted.squeeze::<[usize; 2]>(1).expect("再压回去");
    assert_eq!(back.dims(), [2, 3]);
    assert_eq!(back.as_slice(), tensor.as_slice());
}
