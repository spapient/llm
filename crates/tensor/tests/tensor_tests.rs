use crate::{DType, Element, Order, Owned, Shape, Tensor, View};
use diagnostic::Code;

#[test]
fn a_scalar_has_one_element_and_no_strides() {
    let shape = Shape::new([]);
    assert_eq!(shape.rank(), 0);
    assert_eq!(shape.numel(), 1);
    assert_eq!(shape.strides(Order::RowMajor).as_slice(), [] as [usize; 0]);
    let scalar: View<'_, f32> = View::new(shape, &[0.0f32]).expect("标量");
    assert_eq!(scalar.numel(), 1);
    assert!(scalar.is_contiguous());
}

#[test]
fn a_product_saturates_instead_of_wrapping() {
    assert_eq!(crate::product(&[] as &[u64]), 1, "空轴序列是标量");
    assert_eq!(crate::product(&[2u64, 3]), 6);
    assert_eq!(
        crate::product(&[u64::MAX, 2]),
        u64::MAX,
        "溢出饱和，不回绕成 0"
    );
    assert_eq!(crate::product(&[usize::MAX, 2]), usize::MAX);
}

#[test]
fn row_major_order_matches_the_continuous_strides() {
    for dims in [&[] as &[usize], &[5], &[2, 3], &[2, 3, 4]] {
        let shape = Shape::new(dims.to_vec());
        assert_eq!(
            shape.strides(Order::RowMajor).as_slice(),
            shape.strides(Order::RowMajor).as_slice(),
            "行主序必须与连续步长同源：{dims:?}"
        );
    }
}

#[test]
fn col_major_order_accumulates_along_the_first_axis() {
    assert_eq!(
        Shape::new([2, 3]).strides(Order::ColMajor).as_slice(),
        [1, 2]
    );
    assert_eq!(
        Shape::new([2, 3, 4]).strides(Order::ColMajor).as_slice(),
        [1, 2, 6]
    );
}

#[test]
fn a_scalar_has_one_element_and_no_strides_col_major_too() {
    let shape = Shape::new([]);
    assert_eq!(shape.strides(Order::ColMajor).as_slice(), [] as [usize; 0]);
}

#[test]
fn sub_byte_dtypes_need_fewer_bytes_than_elements() {
    let shape = Shape::new([3, 2]);
    assert_eq!(shape.element_bytes(DType::F32), 24);
    assert_eq!(shape.element_bytes(DType::U4), 3, "6 个 int4 = 3 字节");
    assert_eq!(shape.element_bytes(DType::I8), 6);
}

#[test]
fn a_short_buf_is_rejected_with_the_required_length() {
    let error = View::new(Shape::new([2]), &[0.0f32]).expect_err("应报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("8 字节"), "{error}");
}

/// 形状可以来自文件头：乘积溢出必须饱和，而不是回绕成一个"看起来很小"的数。
#[test]
fn a_shape_whose_product_overflows_saturates_instead_of_wrapping() {
    let huge = Shape::new([usize::MAX, 2]);
    assert_eq!(huge.numel(), usize::MAX, "溢出饱和，不回绕");
    // 回绕会得到 0，于是"空缓冲 + 巨大形状"会被当成合法张量。
    let error = View::new(huge.clone(), &[0.0f32]).expect_err("必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        View::new(huge.clone(), &[0u8; 4]).is_err(),
        "元素视图同样不能放行"
    );
    assert!(
        View::strided(huge, [1, 1], &[0u8; 4]).is_err(),
        "按步长算最高偏移时必须看见溢出"
    );
    assert_eq!(
        Shape::new([usize::MAX, 0, 2]).numel(),
        0,
        "真有一轴为 0 才是 0 个元素"
    );
}

#[test]
fn a_transposed_stride_is_not_contiguous() {
    let shape = Shape::new([2, 3]);
    let tensor: View<'_, i8> = View::new(shape.clone(), &[0i8; 6]).expect("视图");
    assert!(tensor.is_contiguous());
    let transposed = vec![1usize, 2];
    assert_ne!(transposed, shape.strides(Order::RowMajor));
}

#[test]
fn a_view_borrows_the_callers_buf_instead_of_copying_it() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let view: View<'_, f32> = View::new(Shape::new([2, 2]), &values).expect("视图");
    assert_eq!(view.dtype(), DType::F32);
    assert_eq!(view.numel(), 4);
    assert_eq!(view.strides(), [2, 1]);
    assert!(
        std::ptr::eq(view.as_slice().as_ptr(), values.as_ptr()),
        "视图必须借用原缓冲，而不是复制一份"
    );
}

#[test]
fn a_view_is_bounds_checked_without_panicking() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let view: View<'_, f32> = View::new(Shape::new([2, 2]), &values).expect("视图");
    assert_eq!(view.at([0, 1]), Some(2.0));
    assert_eq!(view.at([1, 0]), Some(3.0));
    assert_eq!(view.at([1]), None, "秩不符");
    assert_eq!(view.at([2, 0]), None, "越界");
    assert_eq!(view.row(1), Some(&[3.0, 4.0][..]));
    assert_eq!(view.row(2), None);
}

#[test]
fn a_view_rejects_a_buf_of_the_wrong_length() {
    let error = View::<f32>::new(Shape::new([2, 2]), &[1.0f32, 2.0]).expect_err("长度不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("需要 4 个"), "{error}");
}

#[test]
fn a_row_of_a_higher_rank_view_is_not_guessed() {
    let values = [0u8; 8];
    let view: View<'_, u8> = View::new(Shape::new([2, 2, 2]), &values).expect("视图");
    assert_eq!(view.row(0), None, "只有二维视图才有“行”");
}

#[test]
fn element_types_stay_inside_the_dtype_closed_set() {
    for dtype in [
        <f32 as Element>::DTYPE,
        <i8 as Element>::DTYPE,
        <u8 as Element>::DTYPE,
    ] {
        assert!(DType::ALL.contains(&dtype), "{dtype:?} 不在闭集里");
    }
    assert_eq!(<f32 as Element>::DTYPE, DType::F32);
    assert_eq!(<i8 as Element>::DTYPE, DType::I8);
    // 无符号标量不许冒充有符号类型：`U8` 与 `I8` 在闭集里是两个变体。
    assert_eq!(<u8 as Element>::DTYPE, DType::U8);
    assert_ne!(<u8 as Element>::DTYPE, <i8 as Element>::DTYPE);
}

const SIX: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

#[test]
fn a_transposed_view_reads_through_its_strides() {
    // 2×3 行主序缓冲，按 3×2 转置读：`(i, j)` 落在 `i + 3j`。
    let view: View<'_, f32> = View::strided(Shape::new([3, 2]), [1, 3], &SIX).expect("转置视图");
    assert_eq!(view.at([0, 0]), Some(1.0));
    assert_eq!(view.at([0, 1]), Some(4.0));
    assert_eq!(view.at([2, 1]), Some(6.0));
    assert!(!view.is_contiguous(), "转置后的步长不是行主序");
    assert_eq!(view.numel(), 6);
}

#[test]
fn a_strided_view_rejects_a_step_count_that_does_not_match_the_rank() {
    let error = View::<f32>::strided(Shape::new([2, 2]), [4], &SIX).expect_err("轴数不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("2 个轴"), "{error}");
}

#[test]
fn a_strided_view_rejects_steps_that_run_past_the_buf() {
    let error =
        View::<f32>::strided(Shape::new([2, 2]), [2, 1], &[0.0f32; 3]).expect_err("最高偏移越界");
    assert!(error.to_string().contains("最高偏移 3"), "{error}");
}

#[test]
fn a_row_of_a_transposed_view_is_not_returned_as_a_slice() {
    let view: View<'_, f32> = View::strided(Shape::new([3, 2]), [1, 3], &SIX).expect("转置视图");
    assert_eq!(
        view.row(0),
        None,
        "列不连续时不能把跳着取的元素当成一片内存"
    );
    assert_eq!(view.at([0, 1]), Some(4.0), "逐点读仍然可用");
}

#[test]
fn a_row_strided_view_still_serves_contiguous_rows() {
    // 隔行取样：形状 2×3，行步长 2、列步长 1。
    let view: View<'_, f32> = View::strided(Shape::new([2, 3]), [2, 1], &SIX).expect("行切片");
    assert_eq!(view.row(0), Some(&[1.0, 2.0, 3.0][..]));
    assert_eq!(view.row(1), Some(&[3.0, 4.0, 5.0][..]));
    assert!(!view.is_contiguous(), "行之间跳过了元素，整体不连续");
}

/// 属性骨架：随机小形状保持 numel、连续步长与字节数公式不变量。
///
/// 维度由 PRNG 串行生成（下一个轴依赖上次状态）；溢出饱和、零元素与越界偏移由既有单独用例锁定，
/// 这里只覆盖普通形状的几何一致性。
#[test]
fn random_shapes_keep_geometry_invariants() {
    let mut rng = testkit::Rng::new(0x7E_12);
    for _ in 0..32 {
        let rank = rng.below(4) as usize;
        let dims: Vec<usize> = (0..rank).map(|_| rng.below(8) as usize + 1).collect();
        let shape = Shape::new(dims);
        let strides = shape.strides(Order::RowMajor);

        assert_eq!(shape.numel(), shape.dims().iter().product::<usize>());
        for axis in 0..rank {
            let expected = shape.dims()[axis + 1..].iter().product::<usize>();
            assert_eq!(strides.get(axis).copied(), Some(expected));
        }
        for dtype in [DType::F32, DType::I8, DType::U4] {
            let bits = shape.numel() * dtype.bits() as usize;
            assert_eq!(shape.element_bytes(dtype), bits.div_ceil(8));
        }
    }
}

/// 属性骨架：随机长度的一维视图，缓冲短于形状必须被拒绝，等长才放行。
#[test]
fn random_views_reject_a_short_buf() {
    let mut rng = testkit::Rng::new(0x91);
    for _ in 0..16 {
        let n = rng.below(16) as usize + 1;
        let shape = Shape::new([n]);
        let values: Vec<f32> = vec![0.0; n];
        assert!(View::<f32>::new(shape.clone(), &values).is_ok());
        let short: Vec<f32> = vec![0.0; n - 1];
        assert!(View::<f32>::new(shape, &short).is_err());
    }
}

#[test]
fn a_zero_element_view_needs_an_empty_buf() {
    let empty: [f32; 0] = [];
    assert!(
        View::<f32>::strided(Shape::new([0, 3]), [3, 1], &empty).is_ok(),
        "零元素视图与 `Tensor::new` 同一口径"
    );
    let error = View::<f32>::strided(Shape::new([0, 3]), [3, 1], &[1.0f32]).expect_err("非空缓冲");
    assert!(error.to_string().contains("零元素"), "{error}");
}

/// 自己持有的张量：`B = Owned<f32>`，`R` 与 `K` 走默认（运行期秩 + `f32`）。
type OwnedF32 = Tensor<Owned<f32>>;

#[test]
fn owned_tensor_zeros_allocates_the_required_elements() {
    let owned = OwnedF32::zeros(Shape::new([2, 3])).expect("分配");
    assert_eq!(owned.numel(), 6);
    assert_eq!(owned.as_slice().len(), 6, "元素数，不是字节数");
    assert!(owned.as_slice().iter().all(|elem| *elem == 0.0));
    assert!(owned.is_contiguous());
    assert_eq!(owned.dtype(), DType::F32);
}

#[test]
fn owned_tensor_writes_through_the_mutable_path_and_reads_it_back() {
    let mut owned = OwnedF32::zeros(Shape::new([2])).expect("分配");
    owned.as_slice_mut().copy_from_slice(&[1.0, 2.0]);
    assert_eq!(owned.as_slice(), [1.0, 2.0]);
    assert_eq!(owned.at([1]), Some(2.0));
}

#[test]
fn an_unaffordable_owned_tensor_is_an_error_not_an_abort() {
    let error = OwnedF32::zeros(Shape::new([usize::MAX])).expect_err("地址空间装不下");
    assert_eq!(error.code(), Code::Invalid);
}

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
