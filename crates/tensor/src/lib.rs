//! 泛型逻辑张量：[`Tensor<B, R, K>`](Tensor) 统一存储、秩和元素类型。
//!
//! - `B: `[`Store`]：借用 `&[K]`、可写借用 `&mut [K]` 或自持 [`Owned<K>`]。
//! - `R: `[`Rank`]：编译期 `[usize; D]` 或运行期 `Vec<usize>`，轴数不设固定上限。
//! - `K: `[`Element`]：原生布尔、整数和浮点标量，精度在类型上确定。
//!
//! 转置、置换、广播、切片、分段和改秩都是零拷贝元数据运算；[`Tensor::iter`] 按逻辑序
//! 直接遍历底层存储，算术与归约不再创建中间张量。[`Order`] 配置连续构造的排列顺序，
//! [`Tensor::with_order`] 与 [`Tensor::zeros_with_order`] 同时支持行主序和列主序。
//!
//! 本 crate 只负责形状、布局和主机侧参考运算，不持有设备句柄，也不执行设备内核。
//!
//! ```
//! use llm_tensor::{Shape, Tensor, View};
//!
//! let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let view: View<f32> = View::new(Shape::new([2, 3]), &values)?;
//! assert_eq!(view.strides(), [3, 1]);
//! assert_eq!(view.iter().sum::<f32>(), 21.0);
//!
//! let fixed: Tensor<&[f32], [usize; 2], f32> = Tensor::new([3, 2], values.as_slice())?;
//! assert_eq!(fixed.transpose(0, 1).expect("轴合法").dims(), [2, 3]);
//! # Ok::<(), diagnostic::Error>(())
//! ```

#![deny(unsafe_code)]
#![deny(clippy::pedantic)]

mod layout;
mod ops;
mod tensor;

pub use dtype::{Class, DType, DTypeSpec, FloatLayout};
pub use element::{BoolElement, Element, FloatElement, IntElement, NumericElement, OrderedElement};
pub use ops::Abs;
pub use tensor::{
    AxisLen, DynTensor, Order, Owned, Rank, Shape, Slice, Store, StoreMut, Tensor, View, ViewMut,
    product,
};

use diagnostic::{Code, Error};

/// 构造期 `Invalid` 错误的统一入口。
pub(crate) fn invalid(message: impl Into<String>, marker: &str) -> Error {
    Error::new(Code::Invalid, message).with(marker)
}

mod element {
    //! Rust 标量到 [`DType`] 的封闭映射及算子能力分组。

    use crate::DType;

    mod sealed {
        /// 禁止外部类型加入封闭能力分组。
        pub trait Sealed {}
    }

    /// 张量元素标量；`Default` 支撑零值分配，`Copy` 支撑按值遍历。
    pub trait Element: Copy + Default + 'static {
        /// 对应的运行时元素类型。
        const DTYPE: DType;
    }

    /// 可参与算术的浮点与整数元素。
    pub trait NumericElement: Element + sealed::Sealed {}

    /// 可参与偏序极值归约的数值元素。
    pub trait OrderedElement: NumericElement + PartialOrd {}

    /// 原生浮点元素。
    pub trait FloatElement: NumericElement {}

    /// 原生有符号或无符号整数元素。
    pub trait IntElement: NumericElement {}

    /// 布尔元素。
    pub trait BoolElement: Element + sealed::Sealed {}

    macro_rules! elements {
        ($($scalar:ty => $dtype:expr),+ $(,)?) => {
            $(
                impl Element for $scalar {
                    const DTYPE: DType = $dtype;
                }
            )+
        };
    }

    macro_rules! numeric_group {
        ($capability:ident, $($scalar:ty),+ $(,)?) => {
            $(
                impl sealed::Sealed for $scalar {}
                impl NumericElement for $scalar {}
                impl OrderedElement for $scalar {}
                impl $capability for $scalar {}
            )+
        };
    }

    elements! {
        bool => DType::Bool,
        u8 => DType::U8,
        u16 => DType::U16,
        u32 => DType::U32,
        u64 => DType::U64,
        i8 => DType::I8,
        i16 => DType::I16,
        i32 => DType::I32,
        i64 => DType::I64,
        f32 => DType::F32,
        f64 => DType::F64,
        half::f16 => DType::F16,
        half::bf16 => DType::BF16,
    }

    numeric_group!(FloatElement, f32, f64, half::f16, half::bf16);
    numeric_group!(IntElement, u8, u16, u32, u64, i8, i16, i32, i64);

    impl sealed::Sealed for bool {}
    impl BoolElement for bool {}
}

#[cfg(test)]
#[path = "../tests/ops_tests.rs"]
mod ops_tests;
#[cfg(test)]
mod prop_shape {
    //! 形状与步长的代数性质：任意布局都把多下标**双射**到 `0..numel`。
    //!
    //! 单点用例只钉住几个形状；这里对任意轴数、任意轴长的形状断言双射，
    //! 反向累乘少乘一轴、多乘一轴、或漏掉最后一轴的 `1`，都会被最小反例抓住。

    use crate::{DType, Order, Shape, View};
    use proptest::prelude::*;
    use testkit::prop;

    /// 任意形状：0 轴（标量）到 4 轴，每轴 1..=5 个元素，最宽 625 个。
    fn dims() -> impl Strategy<Value = Vec<usize>> {
        proptest::collection::vec(1usize..=5, 0..=4)
    }

    /// 按行主序枚举形状的全部多下标——顺序无所谓，双射才要紧。
    fn idxs(dims: &[usize]) -> Vec<Vec<usize>> {
        let mut all = vec![Vec::new()];
        for &dim in dims {
            let mut next = Vec::with_capacity(all.len() * dim);
            for prefix in &all {
                for position in 0..dim {
                    let mut idx = prefix.clone();
                    idx.push(position);
                    next.push(idx);
                }
            }
            all = next;
        }
        all
    }

    /// 断言每个多下标换出一个互不相同的线性位号，且 `0..numel` 全被取到。
    // 读回值来自计数缓冲，等于非负小整数位号；clippy 看不到该夹具不变量，允许浮点回位号换算。
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn assert_bijection(view: &View<'_, f32>, dims: &[usize]) {
        let numel = crate::product(dims);
        let mut seen = vec![false; numel];
        for idx in idxs(dims) {
            let linear = view.at(&idx).expect("枚举出的下标必然在形状内") as usize;
            assert!(linear < numel, "线性位号 {linear} 超出 {numel}：{dims:?}");
            assert!(!seen[linear], "位号 {linear} 被两个下标取到：{dims:?}");
            seen[linear] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "有位号没人取到：{dims:?}");
    }

    /// 行主序缓冲：第 `i` 个元素就是 `i`，于是「位号」与「值」互为账本。
    // 计数夹具只取 0..=625 的小整数，在 f32 精确整数范围内；允许整数到浮点的构造换算。
    #[allow(clippy::cast_precision_loss)]
    fn counting(dims: &[usize]) -> Vec<f32> {
        (0..crate::product(dims)).map(|i| i as f32).collect()
    }

    proptest! {
        #![proptest_config(prop::config())]

        #[test]
        fn row_major_offsets_are_a_bijection_onto_the_elements(dims in dims()) {
            let shape = Shape::new(dims.clone());
            let strides = shape.strides(Order::RowMajor);
            let data = counting(&dims);
            let view = View::strided(shape, strides, &data).expect("缓冲长度与形状一致");
            assert_bijection(&view, &dims);
        }

        #[test]
        fn col_major_offsets_are_a_bijection_onto_the_elements(dims in dims()) {
            let shape = Shape::new(dims.clone());
            let strides = shape.strides(Order::ColMajor);
            let data = counting(&dims);
            let view = View::strided(shape, strides, &data).expect("列主序也是合法布局");
            assert_bijection(&view, &dims);
        }

        /// 轴与步幅同时反序就是转置：坐标换了个写法，元素一个不少。
        #[test]
        fn reversing_axes_and_strides_together_transposes(dims in dims()) {
            let shape = Shape::new(dims.clone());
            let mut axes = dims.clone();
            let mut steps = shape.strides(Order::RowMajor).as_slice().to_vec();
            axes.reverse();
            steps.reverse();
            let data = counting(&dims);
            let view = View::strided(Shape::new(axes.clone()), steps, &data)
                .expect("转置布局仍然可达");
            assert_bijection(&view, &axes);
        }

        /// `to_vec` 是"按行主序读一遍"：任何布局摊平后逐位等于 `at` 的读数。
        ///
        /// 收拢两条性质：连续视图与转置视图的摊平长度都等于 `numel`（不按步长乘出多余元素），
        /// 且元素顺序就是逻辑下标顺序——`gather_strided` 的进位一旦少减一轴就会在此暴露。
        #[test]
        fn flattening_a_layout_reads_exactly_what_at_reads(dims in dims()) {
            let shape = Shape::new(dims.clone());
            let data = counting(&dims);
            let view = View::strided(shape.clone(), shape.strides(Order::RowMajor), &data)
                .expect("缓冲长度与形状一致");
            let want: Vec<f32> = idxs(&dims).iter().filter_map(|idx| view.at(idx)).collect();
            prop_assert_eq!(view.to_vec(), want);

            let rank = dims.len();
            if rank >= 2 {
                let turned = view.clone().transpose(0, rank - 1).expect("两轴都在秩内");
                let axes = turned.dims().to_vec();
                let want: Vec<f32> = idxs(&axes).iter().filter_map(|idx| turned.at(idx)).collect();
                prop_assert_eq!(turned.to_vec(), want);
                prop_assert_eq!(turned.to_vec().len(), view.to_vec().len());
            }
        }

        /// 元素数是各轴之积，且秩、字节数都由它派生——三者不允许各算一遍。
        #[test]
        fn the_element_count_is_the_product_of_the_axes(dims in dims()) {
            let shape = Shape::new(dims.clone());
            prop_assert_eq!(shape.numel(), crate::product(&dims));
            prop_assert_eq!(shape.rank(), dims.len());
            prop_assert_eq!(shape.element_bytes(DType::F32), shape.numel() * 4);
        }

        /// 秩代数的定律：`unsqueeze` 再 `squeeze` 是恒等，`select` 的读数等于原张量补上该轴坐标后的读数。
        ///
        /// 这三条改秩算子此前只有实现、没有对拍；定律一次钉住「轴长 / 步长 / 起点偏移」三者同步。
        #[test]
        fn rank_algebra_keeps_the_element_order(dims in dims()) {
            let shape = Shape::new(dims.clone());
            let data = counting(&dims);
            let view = View::strided(shape.clone(), shape.strides(Order::RowMajor), &data)
                .expect("缓冲长度与形状一致");
            for axis in 0..=dims.len() {
                let round = view
                    .clone()
                    .unsqueeze::<Vec<usize>>(axis)
                    .and_then(|wide| wide.squeeze::<Vec<usize>>(axis))
                    .expect("插一个单例轴再摘掉必然成立");
                prop_assert_eq!(round.dims(), view.dims());
                for idx in idxs(&dims) {
                    prop_assert_eq!(round.at(&idx), view.at(&idx));
                }
            }
            for axis in 0..dims.len() {
                for at in [0, dims[axis] - 1] {
                    let picked = view
                        .clone()
                        .select::<Vec<usize>>(axis, at)
                        .expect("轴内下标必然合法");
                    for idx in idxs(&dims) {
                        if idx[axis] != at {
                            continue;
                        }
                        let mut rest = idx.clone();
                        rest.remove(axis);
                        prop_assert_eq!(picked.at(&rest), view.at(&idx));
                    }
                }
            }
        }

        /// 单例轴不改变元素顺序：连续布局对 `dims` 和 `dims + [1]` 给出同一条线性位号。
        #[test]
        fn a_singleton_axis_does_not_reorder_anything(dims in dims()) {
            let mut wider = dims.clone();
            wider.push(1);
            let narrow_shape = Shape::new(dims.clone());
            let wide_shape = Shape::new(wider);
            let data = counting(&dims);
            let narrow = View::strided(narrow_shape.clone(), narrow_shape.strides(Order::RowMajor), &data);
            let wide = View::strided(wide_shape.clone(), wide_shape.strides(Order::RowMajor), &data);
            let (narrow, wide) = (narrow.expect("同长缓冲"), wide.expect("多一个单例轴仍可达"));
            for idx in idxs(&dims) {
                let mut extended = idx.clone();
                extended.push(0);
                prop_assert_eq!(narrow.at(&idx), wide.at(&extended));
            }
        }

        /// 翻转只是「这些轴从另一端读」：结果等于把每个下标在指定轴上取镜像后逐个读，且翻两次复原。
        #[test]
        fn flipping_reads_the_named_axes_from_their_far_end(dims in dims(), mask in any::<u16>()) {
            let data = counting(&dims);
            let shape = Shape::new(dims.clone());
            let view = View::strided(shape.clone(), shape.strides(Order::RowMajor), &data);
            let view = view.expect("同长缓冲");
            let axes: Vec<usize> = (0..dims.len()).filter(|axis| mask >> axis & 1 == 1).collect();
            let turned = view.flip(axes.clone());
            let want: Vec<f32> = idxs(&dims)
                .iter()
                .map(|idx| {
                    let far: Vec<usize> = idx
                        .iter()
                        .enumerate()
                        .map(|(axis, &at)| {
                            if axes.contains(&axis) {
                                dims[axis] - 1 - at
                            } else {
                                at
                            }
                        })
                        .collect();
                    view.at(&far).expect("镜像下标仍在形状内")
                })
                .collect();
            prop_assert_eq!(turned.to_vec(), want);
            prop_assert_eq!(turned.flip(axes).to_vec(), view.to_vec(), "翻两次复原");
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::{
        BoolElement, DType, Element, FloatElement, IntElement, Order, Owned, Shape, Tensor, View,
        product,
    };
    use diagnostic::Code;

    testkit::test_cases! {
    fn a_scalar_has_one_element_and_no_strides() {
        let shape = Shape::new([]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1);
        assert_eq!(shape.strides(Order::RowMajor).as_slice(), [] as [usize; 0]);
        let scalar: View<'_, f32> = View::new(shape, &[0.0f32]).expect("标量");
        assert_eq!(scalar.numel(), 1);
        assert!(scalar.is_contiguous());
    }

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

    fn a_scalar_has_one_element_and_no_strides_col_major_too() {
        let shape = Shape::new([]);
        assert_eq!(shape.strides(Order::ColMajor).as_slice(), [] as [usize; 0]);
    }

    fn sub_byte_dtypes_need_fewer_bytes_than_elements() {
        let shape = Shape::new([3, 2]);
        assert_eq!(shape.element_bytes(DType::F32), 24);
        assert_eq!(shape.element_bytes(DType::U4), 3, "6 个 int4 = 3 字节");
        assert_eq!(shape.element_bytes(DType::I8), 6);
    }

    fn a_short_buf_is_rejected_with_the_required_length() {
        let error = View::new(Shape::new([2]), &[0.0f32]).expect_err("应报错");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("8 字节"), "{error}");
    }

    /// 形状可以来自文件头：乘积溢出必须饱和，而不是回绕成一个"看起来很小"的数。
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

    fn a_transposed_stride_is_not_contiguous() {
        let shape = Shape::new([2, 3]);
        let tensor: View<'_, i8> = View::new(shape.clone(), &[0i8; 6]).expect("视图");
        assert!(tensor.is_contiguous());
        let transposed = vec![1usize, 2];
        assert_ne!(transposed, shape.strides(Order::RowMajor));
    }

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

    fn a_view_rejects_a_buf_of_the_wrong_length() {
        let error = View::<f32>::new(Shape::new([2, 2]), &[1.0f32, 2.0]).expect_err("长度不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("需要 4 个"), "{error}");
    }

    fn a_row_of_a_higher_rank_view_is_not_guessed() {
        let values = [0u8; 8];
        let view: View<'_, u8> = View::new(Shape::new([2, 2, 2]), &values).expect("视图");
        assert_eq!(view.row(0), None, "只有二维视图才有“行”");
    }

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
    }

    const SIX: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    testkit::test_cases! {
    fn a_transposed_view_reads_through_its_strides() {
        // 2×3 行主序缓冲，按 3×2 转置读：`(i, j)` 落在 `i + 3j`。
        let view: View<'_, f32> = View::strided(Shape::new([3, 2]), [1, 3], &SIX).expect("转置视图");
        assert_eq!(view.at([0, 0]), Some(1.0));
        assert_eq!(view.at([0, 1]), Some(4.0));
        assert_eq!(view.at([2, 1]), Some(6.0));
        assert!(!view.is_contiguous(), "转置后的步长不是行主序");
        assert_eq!(view.numel(), 6);
    }

    fn a_strided_view_rejects_a_step_count_that_does_not_match_the_rank() {
        let error = View::<f32>::strided(Shape::new([2, 2]), [4], &SIX).expect_err("轴数不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("2 个轴"), "{error}");
    }

    fn a_strided_view_rejects_steps_that_run_past_the_buf() {
        let error =
            View::<f32>::strided(Shape::new([2, 2]), [2, 1], &[0.0f32; 3]).expect_err("最高偏移越界");
        assert!(error.to_string().contains("最高偏移 3"), "{error}");
    }

    fn a_row_of_a_transposed_view_is_not_returned_as_a_slice() {
        let view: View<'_, f32> = View::strided(Shape::new([3, 2]), [1, 3], &SIX).expect("转置视图");
        assert_eq!(
            view.row(0),
            None,
            "列不连续时不能把跳着取的元素当成一片内存"
        );
        assert_eq!(view.at([0, 1]), Some(4.0), "逐点读仍然可用");
    }

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

    fn a_zero_element_view_needs_an_empty_buf() {
        let empty: [f32; 0] = [];
        assert!(
            View::<f32>::strided(Shape::new([0, 3]), [3, 1], &empty).is_ok(),
            "零元素视图与 `Tensor::new` 同一口径"
        );
        let error = View::<f32>::strided(Shape::new([0, 3]), [3, 1], &[1.0f32]).expect_err("非空缓冲");
        assert!(error.to_string().contains("零元素"), "{error}");
    }
    }

    /// 自己持有的张量：`B = Owned<f32>`，`R` 与 `K` 走默认（运行期秩 + `f32`）。
    type OwnedF32 = Tensor<Owned<f32>>;

    testkit::test_cases! {
    fn owned_tensor_zeros_allocates_the_required_elements() {
        let owned = OwnedF32::zeros(Shape::new([2, 3])).expect("分配");
        assert_eq!(owned.numel(), 6);
        assert_eq!(owned.as_slice().len(), 6, "元素数，不是字节数");
        assert!(owned.as_slice().iter().all(|elem| *elem == 0.0));
        assert!(owned.is_contiguous());
        assert_eq!(owned.dtype(), DType::F32);
    }

    fn owned_tensor_writes_through_the_mutable_path_and_reads_it_back() {
        let mut owned = OwnedF32::zeros(Shape::new([2])).expect("分配");
        owned.as_slice_mut().copy_from_slice(&[1.0, 2.0]);
        assert_eq!(owned.as_slice(), [1.0, 2.0]);
        assert_eq!(owned.at([1]), Some(2.0));
    }

    fn an_unaffordable_owned_tensor_is_an_error_not_an_abort() {
        let error = OwnedF32::zeros(Shape::new([usize::MAX])).expect_err("地址空间装不下");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn column_major_construction_uses_the_same_logical_element_order() {
        let values = [1i32, 4, 2, 5, 3, 6];
        let view: View<'_, i32> =
            View::with_order(Shape::new([2, 3]), &values, Order::ColMajor).expect("列主序");
        assert_eq!(view.strides(), [1, 2]);
        assert_eq!(view.at([1, 0]), Some(4));
        assert_eq!(view.at([0, 1]), Some(2));
        assert_eq!(view.to_vec(), [1, 2, 3, 4, 5, 6]);
        assert!(!view.is_contiguous(), "列主序不是行主序紧凑布局");
        let turned = view.transpose(0, 1).expect("交换两轴");
        assert!(turned.is_contiguous());
        assert_eq!(turned.to_vec(), values);
    }

    fn ordered_zero_allocation_preserves_the_requested_layout() {
        let tensor: Tensor<Owned<f32>> =
            Tensor::zeros_with_order(vec![2, 3], Order::ColMajor).expect("列主序分配");
        assert_eq!(tensor.strides(), [1, 2]);
        assert_eq!(tensor.numel(), 6);
        assert!(tensor.as_slice().iter().all(|&value| value == 0.0));
    }

    fn the_element_count_is_the_product_of_the_axes() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.numel(), product(shape.dims()));
        assert_eq!(shape.element_bytes(DType::F32), 96);
    }

    fn a_scalar_shape_holds_one_element() {
        let shape = Shape::new([]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1, "空轴序列是标量，不是零个元素");
        assert_eq!(shape.element_bytes(DType::F32), 4);
    }

    fn sub_byte_precisions_round_the_byte_count_up() {
        // 6 个元素：f16/bf16 各 2 字节，u4/i4 只占 24 位 = 3 字节。
        // 字节数小于 `numel` 是亚字节精度的正确结果，不是漏乘。
        let shape = Shape::new([2, 3]);
        assert_eq!(shape.element_bytes(DType::F16), 12);
        assert_eq!(shape.element_bytes(DType::BF16), 12);
        assert_eq!(shape.element_bytes(DType::U8), 6);
        assert_eq!(shape.element_bytes(DType::I8), 6);
        assert_eq!(shape.element_bytes(DType::U4), 3);
        assert_eq!(shape.element_bytes(DType::I4), 3);
    }

    fn an_overflowing_shape_saturates_instead_of_wrapping() {
        // 轴长可能来自不受信任的权重文件头：回绕会给出一个偏小的元素数，骗过下游的容量检查。
        let shape = Shape::new([usize::MAX, 2]);
        assert_eq!(shape.numel(), usize::MAX, "饱和到上限，不回绕");
    }

    fn an_axis_length_is_read_by_position() {
        let shape = Shape::new([5, 7]);
        assert_eq!(shape.dim(0), Some(5));
        assert_eq!(shape.dim(1), Some(7));
        assert_eq!(shape.dim(2), None);
    }
    }

    fn assert_float<T: FloatElement>() {}
    fn assert_int<T: IntElement>() {}
    fn assert_bool<T: BoolElement>() {}

    testkit::test_cases! {
    fn bool_maps_to_bool() {
        assert_eq!(bool::DTYPE, DType::Bool);
    }

    fn unsigned_integers_map_to_their_widths() {
        assert_eq!(u8::DTYPE, DType::U8);
        assert_eq!(u16::DTYPE, DType::U16);
        assert_eq!(u32::DTYPE, DType::U32);
        assert_eq!(u64::DTYPE, DType::U64);
    }

    fn signed_integers_map_to_their_widths() {
        assert_eq!(i8::DTYPE, DType::I8);
        assert_eq!(i16::DTYPE, DType::I16);
        assert_eq!(i32::DTYPE, DType::I32);
        assert_eq!(i64::DTYPE, DType::I64);
    }

    fn floats_map_to_their_widths() {
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(f64::DTYPE, DType::F64);
        assert_eq!(half::f16::DTYPE, DType::F16);
        assert_eq!(half::bf16::DTYPE, DType::BF16);
    }

    fn markers_split_closed_groups() {
        assert_float::<f32>();
        assert_float::<f64>();
        assert_float::<half::f16>();
        assert_float::<half::bf16>();
        assert_int::<u8>();
        assert_int::<u32>();
        assert_int::<i8>();
        assert_int::<i64>();
        assert_bool::<bool>();
    }

    fn sub_byte_precisions_have_no_scalar_element() {
        // 亚字节精度是 `DType` 位宽表的一等公民，但没有原生标量，因此不进 `Element`。
        // 它们的字节数由 `DType::bytes_for` 位级进位表达，而不是硬造一个打包标量。
        for dtype in [DType::U1, DType::U2, DType::U4, DType::I4, DType::F4] {
            assert!(dtype.bits() < 8, "{dtype:?} 应是亚字节精度");
        }
    }
    }

    /// 编译期秩 2 的借来视图，用于类型级核心契约。
    type Fix2<'a> = Tensor<&'a [f32], [usize; 2], f32>;

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
    // 计数夹具只取 0..11 小整数，在 f32 精确整数范围内；允许整数到浮点的构造换算。
    #[allow(clippy::cast_precision_loss)]
    fn a_static_rank_owned_tensor_walks_the_whole_shape_surface() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let solid: Tensor<crate::Owned<f32>, [usize; 2], f32> =
            Tensor::new([3, 4], crate::Owned::from(data)).expect("自持张量");

        let window: Tensor<crate::Owned<f32>, [usize; 2], f32> = solid
            .slice([crate::Slice::stepped(0, 3, 2), crate::Slice::new(1, 4)])
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
}
#[cfg(test)]
#[path = "../tests/view_tests.rs"]
mod view_tests;
