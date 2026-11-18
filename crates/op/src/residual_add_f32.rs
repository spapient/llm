//! 残差相加：`y[n] = a[n] + b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Residual`]
//! 单态化。与 `add` 同一段加法，独立成类型以对应网络残差层，样本夹具不同。

crate::binary_f32!(
    Residual,
    ResidualAddF32,
    "residual_add_f32",
    "../tests/residual_add_f32_tests.rs",
    |x, y| x + y,
    |len| (0..len).map(|i| (i as f32) * 0.5).collect(),
    |len| (0..len).map(|i| (i as f32) * 0.25).collect()
);
