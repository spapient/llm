//! 秩进类型的形态：`Tensor<&[f32], [usize; D], f32>`——形状与步长是栈上定长数组，构造不碰堆，
//! 秩由类型钉死，换轴数就编译不过。
//!
//! 它与运行期秩共用 `layout` 与 `shape::accumulate`，所以两个见证不该有任何行为差别；这里既锁住
//! 编译期秩自己的全套布局能力，也锁住"两族算出来的步长逐轴相等"。

use crate::{DType, Element, Order, Shape, Slice, Tensor, product};

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

#[test]
fn typed_tensor_keeps_its_rank_through_narrow_and_slice() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");

    // 类型标注本身就是断言：`narrow`/`slice` 若把秩退化成 `Vec<usize>`，这里编译不过。
    let last: Fix2<'_> = tensor.narrow(0, 1, 1).expect("第 1 行");
    assert_eq!(last.dims(), [1, 3]);
    assert_eq!(last.row(0), Some(&data[3..]), "零拷贝挪到起点 3");

    let corner: Fix2<'_> = last
        .slice([core::ops::Range { start: 0, end: 1 }, 0..2])
        .expect("左上角一块");
    assert_eq!(corner.dims(), [1, 2]);
    assert_eq!(corner.strides(), [3, 1]);
    assert_eq!(corner.base(), 3, "两次切片的起点偏移累加");
}

#[test]
fn typed_tensor_keeps_its_rank_through_flip_and_window_writes() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Fix2<'_> = Tensor::new([2, 3], data.as_slice()).expect("形状与缓冲匹配");

    // `flip` 换存储（借来 → 自持），秩见证原样留在类型上。
    let flipped: Tensor<crate::Owned<f32>, [usize; 2], f32> = tensor.flip([1]);
    assert_eq!(flipped.dims(), [2, 3]);
    assert_eq!(flipped.row(0), Some(&[3.0f32, 2.0, 1.0][..]), "整行倒读");

    let mut own: Tensor<crate::Owned<f32>, [usize; 2], f32> =
        Tensor::new([2, 3], crate::Owned::from(data.to_vec())).expect("自持张量");
    own.slice_fill([1..2, 0..3], 0.0).expect("末行清零");
    own.slice_assign([0..1, 1..3], [8.0f32, 9.0])
        .expect("首行两列");
    assert_eq!(own.row(0), Some(&[1.0f32, 8.0, 9.0][..]));
    assert_eq!(own.row(1), Some(&[0.0f32, 0.0, 0.0][..]));
}

#[test]
fn a_static_rank_owned_tensor_walks_the_whole_shape_surface() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let solid: Tensor<crate::Owned<f32>, [usize; 2], f32> =
        Tensor::new([3, 4], crate::Owned::from(data)).expect("自持张量");

    let window: Tensor<crate::Owned<f32>, [usize; 2], f32> = solid
        .slice([Slice::stepped(0, 3, 2), Slice::new(1, 4)])
        .expect("隔行、去掉首列");
    assert_eq!(window.dims(), [2, 3]);
    assert!(
        window.clone().flatten::<[usize; 1]>(0, 2).is_none(),
        "跳着读的窗口压不平：压平只能落在连续段上"
    );

    let flat: Tensor<crate::Owned<f32>, [usize; 1], f32> =
        window.to_contiguous().flatten(0, 2).expect("物化后压平");
    assert_eq!(flat.dims(), [6], "秩在类型里跟着变");
    assert_eq!(flat.to_vec(), vec![1.0f32, 2.0, 3.0, 9.0, 10.0, 11.0]);

    let wide: Tensor<crate::Owned<f32>, [usize; 2], f32> = flat
        .unsqueeze(0)
        .expect("提轴")
        .broadcast([2, 6])
        .expect("广播");
    let pieces: Vec<_> = wide.chunks(1, 3).collect();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[2].dims(), [2, 2]);

    let diag: Tensor<crate::Owned<f32>, [usize; 1], f32> = wide.diagonal(0, 1).expect("对角");
    assert_eq!(diag.dims(), [2], "2×6 的对角长度取小者");
    assert_eq!(
        diag.to_vec(),
        vec![1.0f32, 2.0],
        "对角沿列走：(0,0)=1、(1,1)=2；行方向步长为 0 不影响"
    );

    let two: Tensor<crate::Owned<f32>, [usize; 2], f32> = diag
        .unsqueeze(0)
        .expect("提轴")
        .broadcast([2, 2])
        .expect("广播");
    assert_eq!(
        two.select::<[usize; 1]>(0, 1).expect("固定首轴").dims(),
        [2],
        "静态秩下 select 同样只减一秩"
    );
}

#[derive(Clone)]
struct ExternalRank(Vec<usize>);

impl crate::Rank for ExternalRank {
    fn as_slice(&self) -> &[usize] {
        &self.0
    }

    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    fn from_slice(values: &[usize]) -> Option<Self> {
        Some(Self(values.to_vec()))
    }
}

#[test]
fn legacy_rank_implementations_use_the_generic_default_extensions() {
    let tensor: Tensor<&[u8], ExternalRank, u8> =
        Tensor::new(ExternalRank(vec![2, 2]), [1u8, 2, 3, 4].as_slice()).expect("外部秩见证");
    let row = tensor.select::<ExternalRank>(0, 1).expect("默认迭代构造");
    assert_eq!(row.dims(), [2]);
    assert_eq!(row.iter().sum::<u8>(), 7);
}
