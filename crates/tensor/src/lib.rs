//! 逻辑张量：`DType` + [`Shape`] + [`Strides`] + 只读字节视图 [`Tensor`]、元素视图
//! [`View`] 与拥有形态 [`OwnedTensor`]。
//!
//! **拥有**：[`Tensor`]、[`OwnedTensor`]、[`View`]、[`Shape`]、[`Strides`]、[`Order`]、
//! [`Element`] 与计算层子集 [`SUPPORTED`]。**不拥有**：元素类型闭集（L1 `dtype`，这里只再导出
//! `DType`）、kernel 侧的操作数视图（DSL 的宿主 API，两者语义不同，见
//! `docs/ARCHITECTURE.md` 第三节）、布局搬运（`storage`）、设备句柄（ADR 0025）。
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

mod shape;
mod tensor;
mod view;

// 元素类型的唯一事实源在 L1 `dtype`：全精度 24 变体，这里不再自定义枚举。
pub use dtype::DType;

pub use element::Element;
pub use shape::{AxisLen, Order, Shape, Strides, product};
pub use tensor::{OwnedTensor, Tensor};
pub use view::View;

/// 计算层当前可消费的元素子集：闭集里的其他类型可以被容器存储，却不能构造进运行时张量。
pub const SUPPORTED: &[DType] = &[DType::F32, DType::F16, DType::BF16, DType::I8, DType::U4];

/// 判定某个 `dtype` 是否属于计算层支持子集。
#[must_use]
pub fn supports(dtype: DType) -> bool {
    SUPPORTED.contains(&dtype)
}

mod element {
    //! 元素类型：把 Rust 标量映射到 [`DType`] 闭集，供 [`crate::View`] 做单态化视图。

    use crate::DType;

    /// 元素类型：把 Rust 标量映射到 [`DType`] 闭集，供 [`crate::View`] 做单态化视图。
    pub trait Element: Copy + 'static {
        /// 对应的元素类型。
        const DTYPE: DType;
    }

    impl Element for f32 {
        const DTYPE: DType = DType::F32;
    }

    impl Element for i8 {
        const DTYPE: DType = DType::I8;
    }

    impl Element for u8 {
        const DTYPE: DType = DType::U8;
    }
}

#[cfg(test)]
#[path = "../tests/tensor_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/prop_shape.rs"]
mod prop_shape;
