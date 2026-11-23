//! 逻辑张量：一个 [`Tensor<B, R, K>`](Tensor) 覆盖全部形态。
//!
//! | 泛型位 | 约束 | 管什么 | 形态 |
//! |---|---|---|---|
//! | `B` | [`Store`] | 元素存在哪 | 借来的 `&'a [K]` / 自己持有的 [`Owned<K>`] |
//! | `R` | [`Rank`] | 秩从哪来 | `[usize; D]` 编译期定长 / `Vec<usize>` 运行期长度 |
//! | `K` | [`Element`] | 元素是什么 | `f32` / `f16` / `bf16` / `i8` / `u8` …，精度钉在类型上 |
//!
//! 这与 Burn 的 `Tensor<B, const D, K>` 同形，两处差异都是刻意的：`B` 是**存储**而不是后端算子集
//! （算子由 `nn` 的数据驱动注册表给出，`B: Backend` 会带来 `FloatTensorOps` 那种百方法接口并把
//! 算子选择从运行期搬到泛型，见 ADR 0053），`R` 是 `[usize; D]` 的**超集**（运行期 batch /
//! segment 几何也要进同一个类型，ADR 0046）。其余全部对齐：连续与非连续布局、stride、transpose、
//! permute、reshape 全是零拷贝的元数据重排；形状、元素数、可达性与溢出在构造期一次拦下。
//!
//! 只有一个 `Tensor`，于是"字节视图 / 元素视图 / 拥有形态"不再是三份平行实现。偏移、可达性、
//! 连续性与取行只有 `layout` 一份实现；连续步长只有 `shape::accumulate` 一份定义。
//!
//! 布局代数比"只有重排"宽：元数据是 `(起点偏移, 各轴长度, 各轴步长)` 三元组，所以整套形状算子
//! 都是 O(轴数) 的整数运算，且**与存储形态无关**（借来视图与自持张量一视同仁）：
//! [`Tensor::broadcast`] 用步长 0 表达广播，[`Tensor::narrow`] / [`Tensor::slice`] 把起点前移
//! 成子视图，[`Tensor::chunks`] / [`Tensor::split`] 借同一存储分段，[`Tensor::select`] /
//! [`Tensor::unbind`] / [`Tensor::squeeze`] / [`Tensor::unsqueeze`] / [`Tensor::flatten`] /
//! [`Tensor::diagonal`] 改秩，[`Tensor::transpose`] / [`Tensor::permute`] / [`Tensor::movedim`] 换轴序（`movedim`
//! 只搬一轴、其余轴次序不动，是 `transpose` 表达不出的那种），[`Tensor::slice_fill`] /
//! [`Tensor::slice_assign`] 往同一个窗口里就地写（自持存储才可写）。改秩的六个要写目标见证
//! （`t.select::<[usize; 1]>(0, 3)`），因为稳定 Rust 算不出 `D - 1`——Burn 的 `.squeeze::<2>()`
//! 同理，只是那边只能是定长数组，这里 `Vec<usize>` 也可以。
//!
//! 唯一搬数据的形状算子是 [`Tensor::flip`]：步长是无符号量，"从远端起倒走"表达不出来，所以它
//! 按镜像下标重算偏移并返回 `Tensor<Owned<K>, R, K>`——借来的视图也能翻，代价是一趟 O(numel)。
//! 带步长的切片（[`Slice::stepped`]，对应 Burn 的 `s![..;2]`）与 [`Tensor::chunks`] / [`Tensor::split`]
//! 分段都只改元数据、只借存储：降采样和"切成 N 份"同样是零拷贝视图。
//!
//! **与 Burn 同宽、并在三处更硬**（都可复核）：
//!
//! 1. **亚字节元素是一等公民**：唯一事实源 `dtype::DType` 的 24 变体里 `U1`/`U2`/`U4`/`I4`/`F4`
//!    与 `F32` 同表，`bytes_for` 做位级向上取整；Burn 的 `Element` 要求 `size_of` 整除，亚字节
//!    只能靠外部打包约定。这里 [`DType::bytes_for`] 是唯一的位级进位算法。
//! 2. **能力层是分类表的一列**：`Class::{Bool,UInt,SInt,Float}` 住在 `DTypeSpec` 行表里，
//!    `is_float`/`is_integer`/`is_signed` 全是它的 `const fn` 投影；Burn 的 `Kind` 每多一档就
//!    多一层泛型参数，这里多一档只是行表加一列，调用方签名不变。
//! 3. **不支持的「算子 × 元素」不可表示**：注册名自带精度档（`add_f32`/`gemv_i2`/`dequant_i8`），
//!    于是本层不需要运行期 `supports(dtype)` 开关；Burn 靠 `Kind` 在编译期拦，这里靠注册表在
//!    构造期拦，两种都对，但后者不要求调用方在签名里背 `Kind`。
//!
//! **拥有**：`Tensor`、`Owned`、`Shape`、`Slice`、`Order`、`AxisLen`、`Element`，以及 `dtype` 的
//! 再导出（`DType`/`Class`/`DTypeSpec`/`FloatLayout`）。**不拥有**：元素类型闭集（L1 `dtype`，这里是
//! 唯一事实源的下游）、dtype-运行期的字节视图（权重层自己的描述符已经承担该契约，形状与字节数由
//! [`Shape::element_bytes`] 给出）、kernel 侧操作数视图（DSL 的宿主 API，语义不同）、布局搬运
//! （`storage`）、设备句柄（ADR 0025）。**依赖**：`dtype`、`diagnostic`。**被依赖**：`storage`、
//! `weights`、`nn`、`train`、`engine`、`autograd`、`snapshot`、`testkit`、`benchmarks`、`cli`、`fuzz`。
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

// 张量方法面：算术 / 归约 / 统计，按元素能力分组收紧 bound。
mod ops;

// 元素类型、能力分类与浮点位域的唯一事实源在 L1 `dtype`：全精度 24 变体，这里不再自定义枚举。
pub use dtype::{Class, DType, DTypeSpec, FloatLayout};

pub use element::{BoolElement, Element, FloatElement, IntElement, NumericElement, OrderedElement};
pub use rank::Rank;
pub use shape::{AxisLen, Order, Shape, Slice, product};
pub use store::{Owned, Store, StoreMut};
pub use tensor::{DynTensor, Tensor, View, ViewMut};

use diagnostic::{Code, Error};

/// 构造期 `Invalid` 错误：全 crate 只有这一条构造路径。
pub(crate) fn invalid(message: impl Into<String>, marker: &str) -> Error {
    Error::new(Code::Invalid, message).with(marker)
}

mod element {
    //! 元素类型：把 Rust 标量映射到 [`DType`] 闭集。这是 [`Tensor`](crate::Tensor) 的第三个泛型位。
    //!
    //! 覆盖全部有 Rust 原生标量对应的精度：`bool`、`u8`–`u64`、`i8`–`i64`、`f32`/`f64` 以及
    //! `half::f16`/`half::bf16`。亚字节精度（`U1`/`U2`/`U4`/`I1`/`I2`/`I4`/`F4`/`F8*`）没有原生
    //! 标量，**不进 `Element`**——它们由 [`DType`] 的位宽表一等公民地表达，需要时由上层反量化成
    //! 原生标量，不在类型层硬造打包标量。
    //!
    //! 能力标记 [`FloatElement`]/[`IntElement`]/[`BoolElement`] 是 sealed 层级：只把原生标量按
    //! 浮点 / 整数 / 布尔封闭分组，供算子族 trait 把 `Elem` bound 收紧到对应分组。它**不进**
    //! [`Tensor`](crate::Tensor) 的公开签名位——张量三泛型仍是 `B`/`R`/`K`，能力域的注册名精度档
    //! 拦阻（`gemv_i4` 这类）仍在构造期，见 ADR 0053 §2/§3。

    use crate::DType;

    mod sealed {
        /// 封闭标记：只有本 crate 内能 impl，外部类型无法把自己接进能力分组。
        pub trait Sealed {}
    }

    /// 元素类型：`K` 的约束。
    ///
    /// `Default` 是 [`Tensor::zeros`](crate::Tensor::zeros) 的前提；`'static` 让 `K` 能进
    /// `PhantomData` 而不拖住借用。
    pub trait Element: Copy + Default + 'static {
        /// 对应的元素类型。
        const DTYPE: DType;
    }

    /// 数值元素能力标记：可参与乘加的浮点与整数封闭分组；布尔不在其列。
    pub trait NumericElement: Element + sealed::Sealed {}

    /// 可有序比较的数值元素：`min` / `max` / `argmax` / 闭包比较以此为 bound。
    ///
    /// 只要求 `PartialOrd`：浮点含 NaN，不满足全序 `Ord`，但归约取极值仍有偏序语义可用。
    pub trait OrderedElement: NumericElement + PartialOrd {}

    /// 浮点元素能力标记：原生浮点标量的封闭分组。
    pub trait FloatElement: NumericElement {}

    /// 整数元素能力标记：原生有符号 / 无符号标量的封闭分组。
    pub trait IntElement: NumericElement {}

    /// 布尔元素能力标记。
    pub trait BoolElement: Element + sealed::Sealed {}

    macro_rules! element {
        ($($scalar:ty => $dtype:expr),+ $(,)?) => {
            $(
                impl Element for $scalar {
                    const DTYPE: DType = $dtype;
                }
            )+
        };
    }

    macro_rules! sealed_group {
        ($trait_:ident, $($scalar:ty),+ $(,)?) => {
            $(
                impl sealed::Sealed for $scalar {}
                impl NumericElement for $scalar {}
                impl OrderedElement for $scalar {}
                impl $trait_ for $scalar {}
            )+
        };
    }

    element! {
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

    sealed_group!(FloatElement, f32, f64, half::f16, half::bf16);
    sealed_group!(IntElement, u8, u16, u32, u64, i8, i16, i32, i64);

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
