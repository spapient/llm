//! 逐元素取大：`y[n] = max(a[n], b[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Max`] 单态化。

crate::binary_f32!(
    Max,
    MaxF32,
    "max_f32",
    "../tests/max_f32_tests.rs",
    |x, y| x.max(y)
);
