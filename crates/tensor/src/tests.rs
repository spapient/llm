use super::*;

#[test]
fn a_scalar_has_one_element_and_no_strides() {
    let shape = Shape::new([]);
    assert_eq!(shape.rank(), 0);
    assert_eq!(shape.numel(), 1);
    assert_eq!(Strides::contiguous(&shape).as_slice(), [] as [usize; 0]);
    let tensor = Tensor::new(DType::F32, shape, &[0u8; 4]).expect("标量");
    assert_eq!(tensor.numel(), 1);
    assert!(tensor.is_contiguous());
}

#[test]
fn sub_byte_dtypes_need_fewer_bytes_than_elements() {
    let shape = Shape::new([3, 2]);
    assert_eq!(shape.element_bytes(DType::F32), 24);
    assert_eq!(shape.element_bytes(DType::U4), 3, "6 个 int4 = 3 字节");
    assert_eq!(shape.element_bytes(DType::I8), 6);
}

#[test]
fn a_short_buffer_is_rejected_with_the_required_length() {
    let error = Tensor::new(DType::F32, Shape::new([2]), &[0u8; 4]).expect_err("应报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("8 字节"), "{error}");
}

/// 形状可以来自文件头：乘积溢出必须饱和，而不是回绕成一个"看起来很小"的数。
#[test]
fn a_shape_whose_product_overflows_saturates_instead_of_wrapping() {
    let huge = Shape::new([usize::MAX, 2]);
    assert_eq!(huge.numel(), usize::MAX, "溢出饱和，不回绕");
    // 回绕会得到 0，于是"空缓冲 + 巨大形状"会被当成合法张量。
    let error = Tensor::new(DType::F32, huge.clone(), &[0u8; 4]).expect_err("必须拒绝");
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        View::new(huge.clone(), &[0u8; 4]).is_err(),
        "元素视图同样不能放行"
    );
    assert!(
        View::strided(huge, Strides::new([1, 1]), &[0u8; 4]).is_err(),
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
    let tensor = Tensor::new(DType::I8, shape.clone(), &[0u8; 6]).expect("视图");
    assert!(tensor.is_contiguous());
    let transposed = Strides::new([1, 2]);
    assert_ne!(transposed, Strides::contiguous(&shape));
}

#[test]
fn a_view_borrows_the_callers_buffer_instead_of_copying_it() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let view = View::new(Shape::new([2, 2]), &values).expect("视图");
    assert_eq!(view.dtype(), DType::F32);
    assert_eq!(view.numel(), 4);
    assert_eq!(view.strides().as_slice(), [2, 1]);
    assert!(
        std::ptr::eq(view.as_slice().as_ptr(), values.as_ptr()),
        "视图必须借用原缓冲，而不是复制一份"
    );
}

#[test]
fn a_view_is_bounds_checked_without_panicking() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let view = View::new(Shape::new([2, 2]), &values).expect("视图");
    assert_eq!(view.at(&[0, 1]), Some(2.0));
    assert_eq!(view.at(&[1, 0]), Some(3.0));
    assert_eq!(view.at(&[1]), None, "秩不符");
    assert_eq!(view.at(&[2, 0]), None, "越界");
    assert_eq!(view.row(1), Some(&[3.0, 4.0][..]));
    assert_eq!(view.row(2), None);
}

#[test]
fn a_view_rejects_a_buffer_of_the_wrong_length() {
    let error = View::new(Shape::new([2, 2]), &[1.0f32, 2.0]).expect_err("长度不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("需要 4 个"), "{error}");
}

#[test]
fn a_row_of_a_higher_rank_view_is_not_guessed() {
    let values = [0u8; 8];
    let view = View::new(Shape::new([2, 2, 2]), &values).expect("视图");
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
    let view = View::strided(Shape::new([3, 2]), Strides::new([1, 3]), &SIX).expect("转置视图");
    assert_eq!(view.at(&[0, 0]), Some(1.0));
    assert_eq!(view.at(&[0, 1]), Some(4.0));
    assert_eq!(view.at(&[2, 1]), Some(6.0));
    assert!(!view.is_contiguous(), "转置后的步长不是行主序");
    assert_eq!(view.numel(), 6);
}

#[test]
fn a_strided_view_rejects_a_step_count_that_does_not_match_the_rank() {
    let error = View::strided(Shape::new([2, 2]), Strides::new([4]), &SIX).expect_err("轴数不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("2 个轴"), "{error}");
}

#[test]
fn a_strided_view_rejects_steps_that_run_past_the_buffer() {
    let error = View::strided(Shape::new([2, 2]), Strides::new([2, 1]), &[0.0f32; 3])
        .expect_err("最高偏移越界");
    assert!(error.to_string().contains("最高偏移 3"), "{error}");
}

#[test]
fn a_row_of_a_transposed_view_is_not_returned_as_a_slice() {
    let view = View::strided(Shape::new([3, 2]), Strides::new([1, 3]), &SIX).expect("转置视图");
    assert_eq!(
        view.row(0),
        None,
        "列不连续时不能把跳着取的元素当成一片内存"
    );
    assert_eq!(view.at(&[0, 1]), Some(4.0), "逐点读仍然可用");
}

#[test]
fn a_row_strided_view_still_serves_contiguous_rows() {
    // 隔行取样：形状 2×3，行步长 2、列步长 1。
    let view = View::strided(Shape::new([2, 3]), Strides::new([2, 1]), &SIX).expect("行切片");
    assert_eq!(view.row(0), Some(&[1.0, 2.0, 3.0][..]));
    assert_eq!(view.row(1), Some(&[3.0, 4.0, 5.0][..]));
    assert!(!view.is_contiguous(), "行之间跳过了元素，整体不连续");
}

#[test]
fn a_zero_element_view_needs_an_empty_buffer() {
    let empty: [f32; 0] = [];
    assert!(
        View::strided(Shape::new([0, 3]), Strides::new([3, 1]), &empty).is_ok(),
        "零元素视图与 `View::new` 同一口径"
    );
    let error =
        View::strided(Shape::new([0, 3]), Strides::new([3, 1]), &[1.0f32]).expect_err("非空缓冲");
    assert!(error.to_string().contains("零元素"), "{error}");
}
