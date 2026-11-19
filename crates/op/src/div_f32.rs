//! 逐元素除法：`y[n] = a[n] / b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Div`] 单态化。
//! 除零按 IEEE-754 语义（`±inf` / `NaN`），不在算子层额外判错。

crate::binary_f32!(
    Div,
    DivF32,
    "div_f32",
    "../tests/div_f32_tests.rs",
    |x, y| x / y
);
