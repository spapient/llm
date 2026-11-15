//! 元素类型：把 Rust 标量映射到 [`DType`] 闭集，供 [`crate::View`] 做单态化视图。

use crate::DType;

/// 元素类型：把 Rust 标量映射到 [`DType`] 闭集，供 [`crate::View`] 做单态化视图。
///
/// 只实现到 `DType` 已有的宽度：`f16` / `bf16` 没有稳定的 Rust 标量类型，
/// `U4` 是按位打包的存储形态（不是一种元素宽度），两者都不适合做视图的元素类型。
///
/// 映射必须**逐个精确**：`u8` 就是 [`DType::U8`]，不是 `I8`。量化权重常以 `u8` 缓冲搬运，
/// 但那是装载方的字节视图（[`crate::Tensor`]），"语义是有符号 int8" 得在反量化时显式说出来；
/// 让无符号标量冒充有符号类型，会让上层按键取内核时静默选错分支。
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
