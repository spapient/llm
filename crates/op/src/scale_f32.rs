//! 标量缩放：`y[n] = alpha * a[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Scale`] 单态化。
//! 标量参数框架只支持整数，故 `alpha` 写成与内核模板一致的常量。

/// 样本固定的缩放系数（内核模板里写同一常量）。
pub const ALPHA: f32 = 0.5;

crate::unary_f32!(
    Scale,
    ScaleF32,
    "scale_f32",
    "../tests/scale_f32_tests.rs",
    |x| ALPHA * x,
    |len| (0..len).map(|i| ((i as f32) * 0.25) - 2.0).collect()
);
