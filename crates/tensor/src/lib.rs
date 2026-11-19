//! 逻辑张量：`DType` + [`Shape`] + [`Strides`] + 只读字节视图 [`Tensor`]、元素视图
//! [`View`] 与拥有形态 [`OwnedTensor`]。
//!
//! **拥有**：[`Tensor`]、[`OwnedTensor`]、[`View`]、[`Shape`]、[`Strides`]、[`Order`]、
//! [`Element`] 与计算层子集 [`SUPPORTED`]。**不拥有**：元素类型闭集（L1 `dtype`，这里只再导出
//! `DType`）、kernel 侧的操作数视图（DSL 的宿主 API，两者语义不同，见
//! `docs/adr/0003-kernel-and-runtime-types.md`）、布局搬运（`storage`）、设备句柄（ADR 0025）。
//! **依赖**：`dtype`、`diagnostic`。**被依赖**：`storage`、`weights`、`kv`、`autograd`、
//! `data`、`nn`、`engine`、`llm`。
//!
//! 这个类型只描述"数据长什么样、放在哪几个字节里"，**不做 `&mut` 分区**——分区是 DSL 的所有权
//! 规则（cutile README §4.4，P0 尚未实现），两套并发规则会互相打架。本层也不做任何执行：
//! 矩阵乘、归约、激活、形状重排都在算子/内核层；这里只提供与后端无关、可构造期校验的元数据。
//!
//! # 例子
//!
//! ```
//! use llm_tensor::{DType, Shape, Tensor};
//!
//! let shape = Shape::new([2, 3]);
//! let bytes = [0u8; 24]; // 2 * 3 个 f32
//! let tensor = Tensor::new(DType::F32, shape, &bytes).expect("形状与缓冲匹配");
//! assert_eq!(tensor.numel(), 6);
//! assert_eq!(tensor.strides().as_slice(), [3, 1], "行主序连续布局");
//! ```

#![deny(unsafe_code)]

/// 三个张量形态（[`OwnedTensor`] / [`Tensor`] / [`View`]）共用的元数据访问器。
///
/// `shape` / `strides` / `numel` 三者完全一致，收敛在 [`tensor_meta_shape!`]；只有 `dtype`
/// 的来源不同——拥有/借用形态读 `self.dtype` 字段，元素视图读泛型 [`Element::DTYPE`]。用
/// `field` / `elem` 两臂区分，避免把 `self` 传进宏（`self` 无法作为宏参数穿透）。
macro_rules! tensor_meta {
    (field) => {
        /// 元素类型（`self.dtype` 字段）。
        pub fn dtype(&self) -> DType {
            self.dtype
        }
        tensor_meta_shape!();
    };
    (elem) => {
        /// 元素类型（泛型 `E::DTYPE`）。
        pub fn dtype(&self) -> DType {
            E::DTYPE
        }
        tensor_meta_shape!();
    };
}

/// [`tensor_meta!`] 里三种形态完全一致的部分：形状 / 步长 / 元素个数。
macro_rules! tensor_meta_shape {
    () => {
        /// 形状。
        pub fn shape(&self) -> &Shape {
            &self.shape
        }

        /// 步长。
        pub fn strides(&self) -> &Strides {
            &self.strides
        }

        /// 元素个数。
        pub fn numel(&self) -> usize {
            self.shape.numel()
        }
    };
}

mod element;
mod order;
mod owned;
mod shape;
mod strides;
mod tensor;
mod validate;
mod view;

// 元素类型的唯一事实源在 L1 `dtype`：全精度 24 变体，这里不再自定义枚举。
pub use dtype::DType;

pub use element::Element;
pub use order::Order;
pub use owned::OwnedTensor;
pub use shape::Shape;
pub use strides::Strides;
pub use tensor::Tensor;
pub use view::View;

/// 计算层当前可消费的元素子集：闭集里的其他类型可以被容器存储，却不能构造进运行时张量。
///
/// 旧设计里 `DType` 本身只有 5 个变体、类型系统天然收口；枚举上移到 L1 后，改用这张
/// 子集表在构造点显式校验，新增一种被内核支持的类型只需在这里加一项。
pub const SUPPORTED: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::I8, DType::U4];

/// 判定某个 `dtype` 是否属于计算层支持子集。
#[must_use]
pub fn supports(dtype: DType) -> bool {
    SUPPORTED.contains(&dtype)
}

#[cfg(test)]
mod tests {
use crate::{DType, Element, Order, OwnedTensor, Shape, Strides, Tensor, View};
use diagnostic::Code;

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
fn row_major_order_matches_the_continuous_strides() {
    for dims in [&[] as &[usize], &[5], &[2, 3], &[2, 3, 4]] {
        let shape = Shape::new(dims.to_vec());
        assert_eq!(
            Strides::ordered(&shape, Order::RowMajor).as_slice(),
            Strides::contiguous(&shape).as_slice(),
            "行主序必须与连续步长同源：{dims:?}"
        );
    }
}

#[test]
fn col_major_order_accumulates_along_the_first_axis() {
    assert_eq!(
        Strides::ordered(&Shape::new([2, 3]), Order::ColMajor).as_slice(),
        [1, 2]
    );
    assert_eq!(
        Strides::ordered(&Shape::new([2, 3, 4]), Order::ColMajor).as_slice(),
        [1, 2, 6]
    );
}

#[test]
fn a_scalar_has_one_element_and_no_strides_col_major_too() {
    let shape = Shape::new([]);
    assert_eq!(
        Strides::ordered(&shape, Order::ColMajor).as_slice(),
        [] as [usize; 0]
    );
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
        let strides = Strides::contiguous(&shape);

        assert_eq!(shape.numel(), shape.dims().iter().product::<usize>());
        for axis in 0..rank {
            let expected = shape.dims()[axis + 1..].iter().product::<usize>();
            assert_eq!(strides.get(axis), Some(expected));
        }
        for dtype in [DType::F32, DType::I8, DType::U4] {
            let bits = shape.numel() * dtype.bits() as usize;
            assert_eq!(shape.element_bytes(dtype), bits.div_ceil(8));
        }
    }
}

/// 属性骨架：随机长度的一维视图，缓冲短于形状必须被拒绝，等长才放行。
#[test]
fn random_views_reject_a_short_buffer() {
    let mut rng = testkit::Rng::new(0x91);
    for _ in 0..16 {
        let n = rng.below(16) as usize + 1;
        let shape = Shape::new([n]);
        let values: Vec<f32> = vec![0.0; n];
        assert!(View::new(shape.clone(), &values).is_ok());
        let short: Vec<f32> = vec![0.0; n - 1];
        assert!(View::new(shape, &short).is_err());
    }
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

#[test]
fn owned_tensor_zeros_allocates_the_required_bytes() {
    let shape = Shape::new([2, 3]);
    let owned = OwnedTensor::zeros(DType::F32, shape).expect("分配");
    assert_eq!(owned.numel(), 6);
    assert_eq!(owned.data().len(), 24);
    assert!(owned.data().iter().all(|byte| *byte == 0));
    assert!(owned.is_contiguous());
}

#[test]
fn owned_tensor_rejects_a_short_or_unsupported_buffer() {
    let short = OwnedTensor::from_bytes(DType::F32, Shape::new([2]), vec![0u8; 4]);
    assert_eq!(short.unwrap_err().code(), Code::Invalid);
    let unsupported = OwnedTensor::zeros(DType::I32, Shape::new([1]));
    assert_eq!(unsupported.unwrap_err().code(), Code::Invalid);
}

#[test]
fn owned_tensor_writes_through_the_mutable_path_and_loans_a_read_view() {
    let mut owned = OwnedTensor::zeros(DType::F32, Shape::new([2])).expect("分配");
    owned.data_mut()[..4].copy_from_slice(&1.0f32.to_ne_bytes());
    let view = owned.view().expect("借出只读视图");
    assert_eq!(view.dtype(), DType::F32);
    assert_eq!(view.numel(), 2);
    assert_eq!(owned.data()[..4], 1.0f32.to_ne_bytes());
}

#[test]
fn an_unaffordable_owned_tensor_is_an_error_not_an_abort() {
    let error =
        OwnedTensor::zeros(DType::F32, Shape::new([usize::MAX])).expect_err("地址空间装不下");
    assert_eq!(error.code(), Code::Invalid);
}
}

