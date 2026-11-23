use crate::{Owned, Shape, Slice, Tensor, View};

#[test]
fn a_broadcast_axis_gets_stride_zero_and_reads_the_same_element() {
    let values = [1.0f32, 2.0];
    let row: View<'_, f32> = View::new(Shape::new([1, 2]), &values).expect("视图");
    let wide = row
        .clone()
        .broadcast(Shape::new([3, 2]))
        .expect("1 轴可广播");
    assert_eq!(wide.dims(), [3, 2]);
    assert_eq!(wide.strides(), [0, 1], "被扩展的轴步长为 0");
    assert_eq!(wide.at([2, 1]), Some(2.0), "三个逻辑行读同一份数据");
    assert!(!wide.is_contiguous(), "步长 0 不是连续布局");

    // 首轴可广播、末轴 2 对 3 -> 整体拒绝。
    assert!(row.clone().broadcast(Shape::new([2, 3])).is_none());
    // 目标秩比源秩小 -> 不是广播。
    assert!(row.broadcast(Shape::new([2])).is_none());
}

#[test]
fn a_narrowed_view_still_borrows_the_callers_buf() {
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view: View<'_, f32> = View::new(Shape::new([3, 2]), &values).expect("视图");
    let tail = view.clone().narrow(0, 1, 2).expect("取第 1..3 行");
    assert_eq!(tail.dims(), [2, 2]);
    assert_eq!(tail.strides(), [2, 1], "步长不变，只是起点前移");
    assert_eq!(tail.row(0), Some(&[3.0, 4.0][..]), "借的是同一块缓冲");
    assert_eq!(tail.at([1, 1]), Some(6.0));

    let column = view.clone().narrow(1, 1, 1).expect("取第 1 列");
    assert_eq!(column.dims(), [3, 1]);
    assert_eq!(column.strides(), [2, 1]);
    assert_eq!(
        [column.at([0, 0]), column.at([1, 0]), column.at([2, 0])],
        [Some(2.0), Some(4.0), Some(6.0)]
    );

    assert!(view.clone().narrow(0, 2, 2).is_none(), "越界");
    assert!(view.clone().narrow(9, 0, 1).is_none(), "轴越界");
    let empty = view.narrow(0, 0, 0).expect("零行是合法视图");
    assert_eq!(empty.numel(), 0);
}

#[test]
fn a_permuted_rank_cannot_panic_on_a_long_order() {
    let tensor: Tensor<&[f32], [usize; 2], f32> =
        Tensor::new([2, 2], [1.0f32, 2.0, 3.0, 4.0].as_slice()).expect("视图");
    assert!(
        tensor.permute([0, 1, 0]).is_none(),
        "长度超过秩的置换返回 None，而不是在定长数组上越界"
    );
}

#[test]
fn a_broadcast_view_materialises_into_a_contiguous_tensor() {
    let buf = [1.0f32, 2.0];
    let row: View<'_, f32> = View::new(Shape::new([1, 2]), &buf[..]).expect("单行");
    let wide = row.broadcast(Shape::new([3, 2])).expect("1 轴可广播");
    assert_eq!(wide.numel(), 6);

    let dense = wide.to_contiguous();
    assert!(dense.is_contiguous(), "物化后必须连续");
    assert_eq!(dense.dims(), [3, 2]);
    assert_eq!(dense.as_slice(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    // 物化的用处就是能接着 reshape——广播视图本身不是连续的，reshape 不了。
    assert!(wide.reshape::<[usize; 1]>([6]).is_none());
    let flat = dense.reshape::<[usize; 1]>([6]).expect("6 个元素");
    assert_eq!(flat.dims(), [6]);
}

#[test]
fn materialising_a_transpose_reorders_the_elements() {
    let buf = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix: View<'_, f32> = View::new(Shape::new([2, 3]), &buf[..]).expect("2×3");
    let turned = matrix.transpose(0, 1).expect("秩 2");
    assert!(!turned.is_contiguous(), "转置后步长不是行主序");
    assert_eq!(
        turned.to_contiguous().as_slice(),
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn matrix_constructor_carries_two_axis_lengths() {
    let buf = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = View::matrix(2, 3, &buf[..]).expect("2×3");
    assert_eq!(view.dims(), [2, 3]);
    assert!(view.is_contiguous());
    assert_eq!(view.row(1), Some(&[4.0, 5.0, 6.0][..]));
    // 元素数与 `rows·cols` 不符在校验拦下。
    assert!(View::matrix(2, 2, &buf[..]).is_err());
}

#[test]
fn a_contiguous_tensor_materialises_to_an_equal_copy() {
    let buf = [1.0f32, 2.0, 3.0, 4.0];
    let dense: View<'_, f32> = View::new(Shape::new([2, 2]), &buf[..]).expect("2×2");
    let copy = dense.to_contiguous();
    assert_eq!(copy.as_slice(), dense.as_slice());
    assert_eq!(copy.strides(), dense.strides());
}

/// 0 到 11 的 3×4 连续张量：下标 = 行 * 4 + 列。
fn counted() -> Vec<f32> {
    (0..12).map(|i| i as f32).collect()
}

#[test]
fn a_slice_advances_the_base_and_keeps_the_strides() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let block = view.slice([0..2, 1..3]).expect("两轴各取一段");
    assert_eq!(block.dims(), [2, 2]);
    assert_eq!(block.strides(), [4, 1], "步长不变，只是起点前移");
    assert_eq!(block.base(), 1, "列 1 起");
    assert_eq!(block.at([0, 0]), Some(1.0));
    assert_eq!(block.at([1, 1]), Some(6.0));
}

#[test]
fn a_slice_outside_the_axes_or_the_rank_is_none() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let rows = 3;
    let inverted = core::ops::Range { start: 2, end: 1 };
    assert!(
        view.clone().slice([1..rows + 1, 0..1]).is_none(),
        "3 行里取不到 1..4"
    );
    assert!(
        view.clone().slice([0..1, inverted]).is_none(),
        "起点在终点之后"
    );
    assert!(
        view.clone().slice([0..1, 0..1, 0..1]).is_none(),
        "轴数超过秩"
    );
    assert!(view.slice([0..3, 0..4]).is_some(), "满段是合法的");
}

#[test]
fn a_selected_axis_disappears_and_the_base_follows_the_idx() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let row = view.select::<Vec<usize>>(0, 1).expect("第 1 行");
    assert_eq!(row.dims(), [4]);
    assert_eq!(row.base(), 4);
    assert_eq!(row.at([2]), Some(6.0));
    assert_eq!(row.numel(), 4);
}

#[test]
fn narrow_works_on_a_self_held_tensor_not_only_on_a_borrowed_view() {
    let tensor: Tensor<Owned<f32>> = Tensor::zeros(vec![3usize, 4]).expect("分配");
    let tail = tensor.narrow(0, 1, 2).expect("自持张量也能零拷贝切");
    assert_eq!(tail.dims(), [2, 4]);
    assert_eq!(tail.base(), 4);
    assert_eq!(tail.strides(), [4, 1]);
    assert_eq!(tail.numel(), 8);
}

#[test]
fn materialising_a_narrowed_view_starts_at_its_base() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let solid = view.narrow(0, 1, 1).expect("第 1 行").to_contiguous();
    assert_eq!(solid.dims(), [1, 4]);
    assert_eq!(solid.base(), 0, "物化后的起点归零");
    assert_eq!(solid.as_slice(), [4.0f32, 5.0, 6.0, 7.0]);
}

#[test]
fn squeeze_only_drops_an_axis_that_is_really_a_singleton() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let wide = view.clone().narrow(0, 0, 1).expect("第 0 行");
    assert_eq!(wide.dims(), [1, 4]);
    assert!(
        wide.clone().squeeze::<Vec<usize>>(1).is_none(),
        "列不是单例"
    );
    let flat = wide.squeeze::<Vec<usize>>(0).expect("去掉单例行轴");
    assert_eq!(flat.dims(), [4]);
    assert_eq!(flat.base(), 0);
}

#[test]
fn unsqueezing_adds_a_singleton_axis_without_moving_anything() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let lifted = view.clone().unsqueeze::<Vec<usize>>(1).expect("插入单例轴");
    assert_eq!(lifted.dims(), [3, 1, 4]);
    assert_eq!(lifted.strides(), [4, 0, 1], "单例轴步长 0，不贡献偏移");
    assert!(lifted.is_contiguous(), "单例轴不破坏连续性");
    assert_eq!(lifted.at([2, 0, 3]), view.at([2, 3]), "元素一个没动");
    assert!(view.unsqueeze::<Vec<usize>>(3).is_none(), "轴不能超过秩");
}

#[test]
fn flatten_merges_a_run_that_is_contiguous_in_storage_and_refuses_otherwise() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let flat = view.clone().flatten::<Vec<usize>>(0, 2).expect("整块连续");
    assert_eq!(flat.dims(), [12]);
    assert_eq!(flat.at([7]), Some(7.0));

    let block = view.clone().narrow(1, 0, 2).expect("前两列");
    assert_eq!(block.dims(), [3, 2]);
    assert!(
        block.clone().flatten::<Vec<usize>>(0, 2).is_none(),
        "行之间隔着两列"
    );
    assert_eq!(
        block.flatten::<Vec<usize>>(1, 2).map(|f| f.dims().to_vec()),
        Some(vec![3, 2]),
        "单轴段是空操作"
    );
}

#[test]
fn a_transposed_pair_of_axes_cannot_be_flattened() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), values.as_slice()).expect("视图");
    let turned = view.transpose(0, 1).expect("转置");
    assert_eq!(turned.strides(), [1, 4]);
    assert!(turned.clone().flatten::<Vec<usize>>(0, 2).is_none());
    assert_eq!(
        turned
            .flatten::<Vec<usize>>(0, 1)
            .map(|f| f.dims().to_vec()),
        Some(vec![4, 3]),
        "单轴段仍然可用"
    );
}

#[test]
fn a_broadcast_axis_is_honestly_not_contiguous() {
    let values = [1.0f32, 2.0];
    let view: View<'_, f32> = View::new(Shape::new([2]), &values).expect("视图");
    let wide = view.broadcast(Shape::new([4, 2])).expect("广播新增前导轴");
    assert_eq!(wide.at([3, 1]), Some(2.0), "步长 0 复用同一个元素");
    assert!(!wide.is_contiguous(), "步长 0 的重复轴不可能是行主序紧凑");
}

#[test]
fn slice_fill_touches_only_the_window() {
    let mut tensor: Tensor<Owned<f32>> = Tensor::zeros(vec![3usize, 4]).expect("分配");
    tensor
        .slice_fill([1..2, 1..3], 9.0)
        .expect("第 1 行第 1、2 列");
    let flat = tensor.to_vec();
    assert_eq!(&flat[0..4], [0.0f32; 4], "窗口外的行原样");
    assert_eq!(&flat[4..8], [0.0f32, 9.0, 9.0, 0.0], "只有窗口被写");
    assert_eq!(&flat[8..12], [0.0f32; 4], "窗口外的行原样");
    assert!(
        tensor.slice_fill([2..4, 0..1], 1.0).is_none(),
        "窗口越界不静默收窄"
    );
}

#[test]
fn a_stepped_window_can_be_written_in_place() {
    let mut tensor: Tensor<Owned<f32>> = Tensor::zeros(vec![2usize, 4]).expect("分配");
    // 每 2 列写一个：两行的第 0、2 列。
    tensor
        .slice_fill([Slice::new(0, 2), Slice::stepped(0, 4, 2)], 9.0)
        .expect("跳着写的窗");
    assert_eq!(
        tensor.to_vec(),
        [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0],
        "只有偶数列被写"
    );
    // 跳着赋值：第 1 行的第 1、3 列，按逻辑顺序逐位对应。
    tensor
        .slice_assign([Slice::new(1, 2), Slice::stepped(1, 4, 2)], [1.0f32, 2.0])
        .expect("窗口元素数为 2");
    assert_eq!(
        tensor.to_vec(),
        [9.0, 0.0, 9.0, 0.0, 9.0, 1.0, 9.0, 2.0],
        "跳着写也按逻辑顺序"
    );
    assert!(
        tensor.slice_fill([Slice::stepped(0, 4, 0)], 1.0).is_none(),
        "步长 0 的段无意义"
    );
}

#[test]
fn slice_assign_follows_the_logical_order_even_when_strided() {
    let solid: Tensor<Owned<f32>> =
        Tensor::new(vec![3usize, 4], Owned::from(counted())).expect("自持张量");
    let mut turned = solid.transpose(0, 1).expect("转置");
    turned
        .slice_assign([0..1, 0..2], [91.0f32, 92.0])
        .expect("窗口大小对得上");
    assert_eq!(turned.at([0, 0]), Some(91.0), "逻辑第 0 位");
    assert_eq!(turned.at([0, 1]), Some(92.0), "逻辑第 1 位，跨步长");
    assert_eq!(turned.at([1, 0]), Some(1.0), "窗口外原样");
    assert_eq!(turned.at([0, 2]), Some(8.0), "窗口外原样");
    assert_eq!(turned.to_vec()[0..2], [91.0, 92.0], "读回来还是逻辑顺序");
    assert!(
        turned.slice_assign([0..1, 0..2], [1.0f32]).is_none(),
        "长度不符不静默截断"
    );
}

#[test]
fn flip_reverses_only_the_named_axes() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let rows = view.flip([0]);
    assert_eq!(rows.dims(), [3, 4]);
    assert_eq!(rows.strides(), [4, 1], "翻转后是紧凑缓冲");
    assert_eq!(rows.at([0, 0]), Some(8.0), "行序倒过来");
    assert_eq!(rows.at([2, 3]), Some(3.0), "末行就是原来的首行");
    let cols = view.clone().flip([1]);
    assert_eq!(cols.at([0, 0]), Some(3.0), "列序倒过来");
    assert_eq!(cols.at([2, 3]), Some(8.0));
    let both = view.flip([0, 1]).to_vec();
    assert_eq!(both[0], 11.0, "两轴都翻等于整段倒读");
    assert_eq!(both[11], 0.0);
}

#[test]
fn flip_is_idempotent_where_it_should_be() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    assert_eq!(view.flip([] as [usize; 0]).to_vec(), values, "不翻任何轴");
    assert_eq!(view.flip([9]).to_vec(), values, "轴号超出秩的忽略");
    assert_eq!(
        view.flip([0, 0]).to_vec(),
        view.flip([0]).to_vec(),
        "重复给出的轴只翻一次"
    );
    assert_eq!(view.flip([0]).flip([0]).to_vec(), values, "翻两次复原");
}

#[test]
fn flip_of_a_strided_view_reads_the_logical_order() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let turned = view.transpose(0, 1).expect("转置");
    assert_eq!(
        turned.flip([0]).at([0, 0]),
        turned.at([3, 0]),
        "翻的是逻辑轴，不是缓冲顺序"
    );
    assert!(turned.flip([0]).is_contiguous(), "物化后紧凑");
}

#[test]
fn a_stepped_slice_reads_every_other_element_without_moving_data() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let every_other_row = view.slice([Slice::stepped(0, 3, 2)]).expect("隔行");
    assert_eq!(every_other_row.dims(), [2, 4], "3 行里隔行取 2 行");
    assert_eq!(every_other_row.strides(), [8, 1], "步长翻倍，不是复制");
    assert!(!every_other_row.is_contiguous(), "跳着读就不再紧凑");
    assert_eq!(
        every_other_row.to_vec(),
        vec![0.0f32, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn a_stepped_view_writes_through_its_own_steps() {
    let solid: Tensor<Owned<f32>> =
        Tensor::new(vec![3, 4], Owned::from(counted())).expect("自持张量");
    let mut every_other_row = solid
        .slice([Slice::stepped(0, 3, 2), Slice::new(0, 4)])
        .expect("隔行");
    every_other_row
        .slice_assign([1..2, 0..4], [90.0f32, 91.0, 92.0, 93.0])
        .expect("写第二行");
    assert_eq!(
        every_other_row.to_vec(),
        vec![0.0f32, 1.0, 2.0, 3.0, 90.0, 91.0, 92.0, 93.0],
        "写入落在第 2 行（缓冲偏移 8），步长被尊重"
    );
}

#[test]
fn a_step_of_zero_or_an_inverted_range_is_none() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    assert!(
        view.clone().slice([Slice::stepped(0, 3, 0)]).is_none(),
        "步长 0 会原地打转"
    );
    assert!(
        view.slice([Slice::stepped(3, 0, 1)]).is_none(),
        "倒序段只能靠 flip"
    );
}

#[test]
fn chunks_split_into_borrowed_views_without_copying() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let parts: Vec<_> = view.chunks(0, 2).collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].dims(), [2, 4]);
    assert_eq!(parts[1].dims(), [1, 4], "3 行等分 2 段：2 + 1");
    assert_eq!(
        parts[0].to_vec(),
        vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(parts[1].base(), 8, "第二段只是起点前移");
    assert_eq!(parts[1].to_vec(), vec![8.0f32, 9.0, 10.0, 11.0]);
}

#[test]
fn chunks_of_a_transposed_view_keep_the_layout() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let turned = view.transpose(0, 1).expect("转置");
    let parts: Vec<_> = turned.chunks(1, 3).collect();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[2].dims(), [4, 1]);
    assert_eq!(parts[2].strides(), [1, 4], "分段不改步长");
    assert_eq!(parts[2].to_vec(), vec![8.0f32, 9.0, 10.0, 11.0], "第 3 列");
}

#[test]
fn diagonal_folds_two_axes_by_adding_their_strides() {
    let values = counted();
    let matrix: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let raw = matrix.as_slice().as_ptr();
    let diag: View<'_, f32> = matrix.diagonal(0, 1).expect("对角");
    assert_eq!(diag.dims(), [3], "秩减一，长度取两轴较小者");
    assert_eq!(diag.strides(), [5], "步长相加：4 + 1");
    assert_eq!(diag.to_vec(), vec![0.0f32, 5.0, 10.0], "一个元素都没搬");
    assert_eq!(diag.as_slice().as_ptr(), raw, "同一块存储");
}

#[test]
fn diagonal_of_a_transposed_view_reads_the_same_elements() {
    let values = counted();
    let matrix: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let turned = matrix.transpose(0, 1).expect("转置");
    let diag: View<'_, f32> = turned.diagonal(1, 0).expect("轴序无关");
    assert_eq!(diag.dims(), [3]);
    assert_eq!(diag.to_vec(), vec![0.0f32, 5.0, 10.0]);
}

#[test]
fn diagonal_rejects_the_same_axis_or_an_axis_out_of_range() {
    let values = counted();
    let square = || -> View<'_, f32> { View::new(Shape::new([3, 4]), &values).expect("视图") };
    assert!(
        square().diagonal::<[usize; 1]>(1, 1).is_none(),
        "同一轴无几何含义"
    );
    assert!(square().diagonal::<[usize; 1]>(0, 9).is_none(), "轴越界");
}

#[test]
fn diagonal_through_a_window_keeps_the_parent_offset() {
    let values = counted();
    let matrix: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let window = matrix.narrow(1, 1, 3).expect("取中三列");
    let diag: View<'_, f32> = window.diagonal(0, 1).expect("对角");
    assert_eq!(diag.to_vec(), vec![1.0f32, 6.0, 11.0], "起点已前移一位");
}

#[test]
fn split_takes_the_sizes_it_is_given_and_clamps_the_last_one() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let parts = view.split(0, [1, 2]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].dims(), [1, 4]);
    assert_eq!(parts[1].dims(), [2, 4], "段长按给定值，不是等分");
    assert_eq!(parts[1].base(), 4);
    assert_eq!(
        parts[1].to_vec(),
        vec![4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
    let clamped = view.split(0, [2, 5]);
    assert_eq!(clamped.len(), 2, "越过轴长的那段被夹到剩余长度");
    assert_eq!(clamped[1].dims(), [1, 4], "3 - 2 = 1 行");
    assert_eq!(view.split(0, [9]).len(), 1, "首段就超长：夹成整轴");
}

#[test]
fn unbind_gives_rank_minus_one_views_over_the_same_storage() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    let lines: Vec<View<'_, f32>> = view.unbind(0);
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[2].dims(), [4], "秩减一");
    assert_eq!(lines[2].strides(), [1], "步长原样，只是去掉一轴");
    assert_eq!(lines[2].to_vec(), vec![8.0f32, 9.0, 10.0, 11.0]);
    assert_eq!(
        view.unbind::<[usize; 1]>(0).len(),
        3,
        "编译期秩见证同样可用"
    );
    assert!(
        view.unbind::<[usize; 2]>(0).is_empty(),
        "见证秩与结果秩对不上：一份也给不出，不 panic"
    );
}

#[test]
fn unbind_views_write_through_to_the_source() {
    let owned: Tensor<Owned<f32>, Vec<usize>, f32> =
        Tensor::new(vec![3usize, 4], Owned::from(counted())).expect("自持张量");
    let columns: Vec<View<'_, f32>> = owned.unbind(1);
    assert_eq!(columns.len(), 4, "列数");
    assert_eq!(columns[2].to_vec(), vec![2.0f32, 6.0, 10.0], "第 3 列");
    let rows: Vec<View<'_, f32>> = owned.unbind(0);
    assert_eq!(rows[0].to_vec(), vec![0.0f32, 1.0, 2.0, 3.0]);
}

#[test]
fn movedim_relocates_one_axis_and_leaves_the_others_in_order() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([2, 2, 3]), &values).expect("视图");
    let moved = view.clone().movedim(0, 2).expect("搬轴");
    assert_eq!(moved.dims(), [2, 3, 2], "第一轴搬到末尾，其余轴次序不动");
    assert_eq!(
        moved.strides(),
        [3, 1, 6],
        "只重排步长，一块存储、一次分配都不多"
    );
    // 与 permute 的同一条置换逐元素相等：movedim 是 permute 的命名特例，不是第二套几何。
    let permuted = view.clone().permute([1, 2, 0]).expect("置换");
    assert_eq!(moved.to_vec(), permuted.to_vec());
}

#[test]
fn movedim_is_identity_when_the_axis_stays_and_none_when_out_of_range() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");
    assert_eq!(
        view.clone().movedim(1, 1).expect("恒等").to_vec(),
        view.to_vec()
    );
    assert!(view.clone().movedim(2, 0).is_none(), "源轴越界");
    assert!(view.movedim(0, 2).is_none(), "目标位置越界");
}

#[test]
fn shape_operators_reject_out_of_range_axes_without_panicking() {
    let values = counted();
    let view: View<'_, f32> = View::new(Shape::new([3, 4]), &values).expect("视图");

    // 轴号 / 下标越界：重排与减秩算子一律返回 `None`，不在定长数组上越界读。
    assert!(
        view.clone().transpose(0, 9).is_none(),
        "transpose 目标轴越界"
    );
    assert!(
        view.clone().select::<Vec<usize>>(9, 0).is_none(),
        "select 轴越界"
    );
    assert!(
        view.clone().select::<Vec<usize>>(0, 9).is_none(),
        "select 下标越界"
    );
    assert!(
        view.clone().squeeze::<Vec<usize>>(9).is_none(),
        "squeeze 轴越界"
    );
    assert!(view.clone().permute([0, 9]).is_none(), "permute 取值超出秩");

    // 只产出集合的切分算子：轴越界给空集合，越界读不发生。
    assert!(view.chunks(9, 2).next().is_none(), "chunks 轴越界为空");
    assert!(view.unbind::<Vec<usize>>(9).is_empty(), "unbind 轴越界为空");
}
