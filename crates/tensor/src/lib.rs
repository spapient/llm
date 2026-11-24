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

mod layout;
mod ops;
mod shape;
mod tensor;
mod view;

pub use dtype::{Class, DType, DTypeSpec, FloatLayout};
pub use element::{BoolElement, Element, FloatElement, IntElement, NumericElement, OrderedElement};
pub use ops::Abs;
pub use shape::{AxisLen, Order, Rank, Shape, Slice, product};
pub use tensor::{DynTensor, Owned, Store, StoreMut, Tensor, View, ViewMut};

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
#[path = "../tests/tensor_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/view_tests.rs"]
mod view_tests;

#[cfg(test)]
#[path = "../tests/typed_tests.rs"]
mod static_rank_tests;

#[cfg(test)]
#[path = "../tests/prop_shape.rs"]
mod prop_shape;

#[cfg(test)]
#[path = "../tests/shape_tests.rs"]
mod shape_tests;

#[cfg(test)]
#[path = "../tests/element_tests.rs"]
mod element_tests;

#[cfg(test)]
#[path = "../tests/ops_tests.rs"]
mod ops_tests;
