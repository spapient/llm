//! 逐元素取小：`y[n] = min(a[n], b[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Min`] 单态化。

crate::binary_f32!(
    Min,
    MinF32,
    "min_f32",
    "../tests/min_f32_tests.rs",
    |x, y| x.min(y)
);
