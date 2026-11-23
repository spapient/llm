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
