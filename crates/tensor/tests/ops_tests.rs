//! 张量方法面对拍：算术 / 归约 / 统计 / 互转，参考语义与逐元素手算一致。

use crate::{DynTensor, Owned, Tensor, View, ViewMut};

fn matrix(values: &[f32], rows: usize, cols: usize) -> View<'_, f32> {
    View::matrix(rows, cols, values).expect("形状与缓冲匹配")
}

#[test]
fn add_matches_elementwise_reference() {
    let a = matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = matrix(&[4.0, 3.0, 2.0, 1.0], 2, 2);
    let sum = a.add(&b).expect("同形");
    assert_eq!(sum.as_slice(), [5.0, 5.0, 5.0, 5.0]);
}

#[test]
fn arithmetic_on_noncontiguous_view_follows_logical_order() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let view: Tensor<&[f32], [usize; 2], f32> =
        Tensor::new([2, 2], values.as_slice()).expect("视图");
    let transposed = view.transpose(0, 1).expect("转置");
    // 转置后逻辑序为 [1, 3, 2, 4]；乘标量按逻辑序走，不受存储步长影响。
    let doubled = transposed.mul_scalar(2.0);
    assert_eq!(doubled.to_vec(), [2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn shape_mismatch_is_rejected_not_broadcast() {
    let a = matrix(&[1.0, 2.0, 3.0], 3, 1);
    let b = matrix(&[1.0, 2.0], 2, 1);
    assert!(a.add(&b).is_err(), "形状不符必须返回 Invalid");
}

#[test]
fn reductions_pick_extrema_and_idx() {
    let view = matrix(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 2, 3);
    assert_eq!(view.sum(), Some(23.0));
    assert_eq!(view.max(), Some(9.0));
    assert_eq!(view.min(), Some(1.0));
    assert_eq!(view.argmax(), Some(5));
}

#[test]
fn scalar_map_and_close_check() {
    let view = matrix(&[1.0, 2.0, 3.0], 3, 1);
    let shifted = view.add_scalar(10.0);
    assert_eq!(shifted.as_slice(), [11.0, 12.0, 13.0]);
    assert!(view.all_close(&view, 0.0));
    let other = matrix(&[1.0, 2.0, 3.05], 3, 1);
    assert!(view.all_close(&other, 0.1));
    assert!(!view.all_close(&other, 0.01));
}

#[test]
fn owned_conversions_keep_elements() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let view = matrix(&values, 2, 2);
    let owned = view.clone().into_owned();
    assert_eq!(owned.as_slice(), &values[..]);

    let dyn_tensor: DynTensor<f32> = view.into_dyn();
    assert_eq!(dyn_tensor.dims(), [2, 2]);
    assert_eq!(dyn_tensor.to_vec(), values);
    assert_eq!(dyn_tensor.as_view().at([1, 1]), Some(4.0));
}

#[test]
fn mutable_view_shares_the_owned_buf() {
    let mut owned = Tensor::<Owned<f32>, Vec<usize>, f32>::zeros(vec![2, 2]).expect("分配");
    let mut write = owned.as_view_mut();
    write.as_slice_mut()[0] = 7.0;
    assert_eq!(owned.as_slice()[0], 7.0, "可写视图与自持缓冲同址");
}

#[test]
fn map_keeps_compile_time_rank() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let tensor: Tensor<&[f32], [usize; 2], f32> =
        Tensor::new([2, 2], values.as_slice()).expect("视图");
    let mapped = tensor.map(|x| x + 1.0);
    let fixed: Tensor<Owned<f32>, [usize; 2], f32> = mapped;
    assert_eq!(fixed.dims(), [2, 2]);
    assert_eq!(fixed.to_vec(), [2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn view_mut_one_writes_back_through_slice() {
    let mut buf = [1.0f32, 2.0];
    let mut view = ViewMut::one(&mut buf);
    view.as_slice_mut()[0] = 9.0;
    assert_eq!(buf[0], 9.0);
}

/// 能力分组编译期见证：数值标量同时满足数值与有序边界；`bool` 不在其内。
fn numeric<T: crate::NumericElement>() {}
fn ordered<T: crate::OrderedElement>() {}

#[test]
fn closed_groups_cover_numeric_scalars() {
    numeric::<f32>();
    ordered::<f32>();
    numeric::<i8>();
    ordered::<u32>();
    numeric::<half::f16>();
}

#[test]
fn map_can_change_the_element_type_without_changing_rank() {
    let tensor: Tensor<&[i32], [usize; 2], i32> =
        Tensor::new([2, 2], [0, 1, 2, 3].as_slice()).expect("视图");
    let positive: Tensor<Owned<bool>, [usize; 2], bool> = tensor.map(|value| value > 0);
    assert_eq!(positive.dims(), [2, 2]);
    assert_eq!(positive.to_vec(), [false, true, true, true]);
}

#[test]
fn binary_operations_accept_different_storage_and_rank_witnesses() {
    let left: Tensor<&[i32], [usize; 2], i32> =
        Tensor::new([2, 2], [1, 2, 3, 4].as_slice()).expect("定秩借用");
    let right: Tensor<Owned<i32>, Vec<usize>, i32> =
        Tensor::new(vec![2, 2], Owned::from(vec![4, 3, 2, 1])).expect("动态秩自持");
    let sum = left.add(&right).expect("形状一致");
    assert_eq!(sum.to_vec(), [5, 5, 5, 5]);
    let widened: Tensor<Owned<i64>, [usize; 2], i64> = left
        .zip_with(&right, |a, b| i64::from(a * b))
        .expect("输出类型可改变");
    assert_eq!(widened.to_vec(), [4, 6, 6, 4]);
}

#[test]
fn extreme_positions_keep_the_first_of_equal_values() {
    let view: View<'_, i32> = View::new(vec![4], &[2, 1, 2, 1]).expect("视图");
    assert_eq!(view.argmax(), Some(0));
    assert_eq!(view.argmin(), Some(1));
}

#[test]
fn integer_close_checks_never_overflow() {
    let low: View<'_, i64> = View::new(vec![1], &[i64::MIN]).expect("视图");
    let high: View<'_, i64> = View::new(vec![1], &[i64::MAX]).expect("视图");
    assert!(!low.all_close(&high, i64::MAX));
    assert!(low.all_close(&low, 0));
    assert!(!low.all_close(&low, -1), "负容差不能放行任何值");
}

#[test]
fn complete_owned_tensor_into_vec_reuses_its_allocation() {
    let tensor: Tensor<Owned<i32>, Vec<usize>, i32> =
        Tensor::new(vec![2, 2], Owned::from(vec![1, 2, 3, 4])).expect("张量");
    let pointer = tensor.as_slice().as_ptr();
    let values = tensor.into_vec();
    assert_eq!(values, [1, 2, 3, 4]);
    assert_eq!(pointer, values.as_ptr(), "完整连续张量不应再次分配");
}

#[test]
// 混合类型映射仅取 1、2 两个小整数，在 f32 精确整数范围内；允许整数到浮点的测试换算。
#[allow(clippy::cast_precision_loss)]
fn zip_with_accepts_different_input_element_types() {
    let left: View<'_, i32> = View::new(vec![2], &[1, 2]).expect("整数输入");
    let right: Tensor<Owned<f32>, [usize; 1], f32> =
        Tensor::new([2], Owned::from(vec![0.5, 0.25])).expect("浮点输入");
    let output: Tensor<Owned<f32>, Vec<usize>, f32> = left
        .zip_with(&right, |integer, float| integer as f32 + float)
        .expect("混合元素类型");
    assert_eq!(output.to_vec(), [1.5, 2.25]);
}

#[test]
fn owned_consumption_reuses_the_same_allocation() {
    let tensor: Tensor<Owned<i32>, [usize; 2], i32> =
        Tensor::new([2, 2], Owned::from(vec![1, 2, 3, 4])).expect("自持张量");
    let pointer = tensor.as_slice().as_ptr();
    let owned = tensor.into_owned();
    assert_eq!(owned.as_slice().as_ptr(), pointer, "into_owned 不应复制");
    let dynamic = owned.into_dyn();
    assert_eq!(dynamic.as_slice().as_ptr(), pointer, "into_dyn 不应复制");
    assert_eq!(dynamic.to_vec(), [1, 2, 3, 4]);
}

#[test]
fn contiguous_subview_into_vec_reuses_the_parent_allocation() {
    let tensor: Tensor<Owned<i32>, Vec<usize>, i32> =
        Tensor::new(vec![3, 2], Owned::from(vec![1, 2, 3, 4, 5, 6])).expect("父张量");
    let pointer = tensor.as_slice().as_ptr();
    let tail = tensor.narrow(0, 1, 1).expect("中间一行");
    let values = tail.into_vec();
    assert_eq!(values, [3, 4]);
    assert_eq!(
        values.as_ptr(),
        pointer,
        "连续子视图只需移动元素，不重新分配"
    );
}

struct LegacyStore<'a>(&'a [u8]);

impl crate::Store<u8> for LegacyStore<'_> {
    fn slice(&self) -> &[u8] {
        self.0
    }
}

#[test]
fn legacy_store_implementations_use_the_default_range_copy() {
    let tensor: Tensor<LegacyStore<'_>, [usize; 1], u8> =
        Tensor::new([4], LegacyStore(&[1, 2, 3, 4])).expect("旧式自定义存储");
    let owned = tensor.narrow(0, 1, 2).expect("借用子视图").into_owned();
    assert_eq!(owned.to_vec(), [2, 3], "默认实现只复制逻辑范围");
}
