//! 逻辑张量：一个 [`Tensor<B, R, K>`](Tensor) 覆盖全部形态。
//!
//! | 泛型位 | 约束 | 管什么 | 形态 |
//! |---|---|---|---|
//! | `B` | [`Store`] | 元素存在哪 | 借来的 `&'a [K]` / 自己持有的 [`Owned<K>`] |
//! | `R` | [`Rank`] | 秩从哪来 | `[usize; D]` 编译期定长 / `Vec<usize>` 运行期长度 |
//! | `K` | [`Element`] | 元素是什么 | `f32` / `i8` / `u8`，精度钉在类型上 |
//!
//! 这与 Burn 的 `Tensor<B, const D, K>` 同形，两处差异都是刻意的：`B` 是**存储**而不是后端算子集
//! （算子由 `op` 的数据驱动注册表给出，`B: Backend` 会带来 `FloatTensorOps` 那种百方法接口并把
//! 算子选择从运行期搬到泛型，见 ADR 0053），`R` 是 `[usize; D]` 的**超集**（运行期 batch /
//! segment 几何也要进同一个类型，ADR 0046）。其余全部对齐：连续与非连续布局、stride、transpose、
//! permute、reshape 全是零拷贝的元数据重排；形状、元素数、可达性与溢出在构造期一次拦下。
//!
//! 只有一个 `Tensor`，于是"字节视图 / 元素视图 / 拥有形态"不再是三份平行实现。偏移、可达性、
//! 连续性与取行只有 [`layout`] 一份实现；连续步长只有 [`shape::accumulate`] 一份定义。
//!
//! 布局代数比"只有重排"宽：元数据是 `(起点偏移, 各轴长度, 各轴步长)` 三元组，所以整套形状算子
//! 都是 O(轴数) 的整数运算，且**与存储形态无关**（借来视图与自持张量一视同仁）：
//! [`Tensor::broadcast`] 用步长 0 表达广播，[`Tensor::narrow`] / [`Tensor::slice`] 把起点前移
//! 成子视图，[`Tensor::select`] / [`Tensor::squeeze`] / [`Tensor::unsqueeze`] / [`Tensor::flatten`]
//! 改秩，[`Tensor::transpose`] / [`Tensor::permute`] 换轴序，[`Tensor::slice_fill`] /
//! [`Tensor::slice_assign`] 往同一个窗口里就地写（自持存储才可写）。改秩的四个要写目标见证
//! （`t.select::<[usize; 1]>(0, 3)`），因为稳定 Rust 算不出 `D - 1`——Burn 的 `.squeeze::<2>()`
//! 同理，只是那边只能是定长数组，这里 `Vec<usize>` 也可以。
//!
//! 唯一搬数据的形状算子是 [`Tensor::flip`]：步长是无符号量，"从远端起倒走"表达不出来，所以它
//! 按镜像下标重算偏移并返回 `Tensor<Owned<K>, R, K>`——借来的视图也能翻，代价是一趟 O(numel)。
//!
//! **拥有**：`Tensor`、`Owned`、`Shape`、`Order`、`AxisLen`、`Element` 与 `DType`
//! 的再导出。**不拥有**：元素类型闭集（L1 `dtype`，这里是唯一事实源的下游）、dtype-运行期的
//! 字节视图（权重层自己的描述符已经承担该契约，形状与字节数由 `Shape::element_bytes` 给出）、
//! kernel 侧操作数视图（DSL 的宿主 API，语义不同）、布局搬运（`storage`）、设备句柄（ADR 0025）。
//! **依赖**：`dtype`、`diagnostic`。**被依赖**：`storage`、`weights`、`kv`、`autograd`、`data`、
//! `nn`、`engine`、`llm`、`cli`。
//!
//! 本层只描述"数据长什么样、放在哪几个字节里"，**不做 `&mut` 分区**（那是 DSL 的所有权规则，
//! cutile README §4.4），也不做任何执行——矩阵乘、归约、激活都在算子 / 内核层。窗口写入
//! （`slice_fill` / `slice_assign`）写的是自己的缓冲，不涉及 kernel 操作数分区。
//!
//! # 例子
//!
//! ```
//! use llm_tensor::{Shape, Tensor, View};
//!
//! let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let view: View<f32> = View::new(Shape::new([2, 3]), &values).expect("形状与缓冲匹配");
//! assert_eq!(view.numel(), 6);
//! assert_eq!(view.strides(), [3, 1], "行主序连续布局");
//! assert_eq!(view.at([1, 2]), Some(6.0));
//!
//! // 秩进类型：形状写成 `[usize; 2]`，换成三轴就编译不过。
//! let fixed: Tensor<&[f32], [usize; 2], f32> =
//!     Tensor::new([3, 2], values.as_slice()).expect("形状与缓冲匹配");
//! assert_eq!(fixed.transpose(0, 1).expect("轴合法").dims(), [2, 3]);
//! ```

#![deny(unsafe_code)]

mod shape;
mod tensor;

// 秩见证（[usize; D] / Vec<usize>）与元素存储（&[K] / Owned<K>）。
mod rank;
mod store;

// 布局算式的唯一实现：偏移 / 可达性 / 连续性 / 取行 + 全部形状代数，两种秩、两种存储共用。
mod layout;

// 元素类型的唯一事实源在 L1 `dtype`：全精度 24 变体，这里不再自定义枚举。
pub use dtype::DType;

pub use element::Element;
pub use rank::Rank;
pub use shape::{AxisLen, Order, Shape, Slice, product};
pub use store::{Owned, Store, StoreMut};
pub use tensor::{Tensor, View};

use diagnostic::{Code, Error};

/// 构造期 `Invalid` 错误：全 crate 只有这一条构造路径。
pub(crate) fn invalid(message: impl Into<String>, marker: &str) -> Error {
    Error::new(Code::Invalid, message).with(marker)
}

mod element {
    //! 元素类型：把 Rust 标量映射到 [`DType`] 闭集。这是 [`Tensor`](crate::Tensor) 的第三个泛型位。

    use crate::DType;

    /// 元素类型：`K` 的约束。
    ///
    /// `Default` 是 [`Tensor::zeros`](crate::Tensor::zeros) 的前提；`'static` 让 `K` 能进
    /// `PhantomData` 而不拖住借用。
    pub trait Element: Copy + Default + 'static {
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
#[path = "../tests/typed_tests.rs"]
mod static_rank_tests;

#[cfg(test)]
#[path = "../tests/prop_shape.rs"]
mod prop_shape;
