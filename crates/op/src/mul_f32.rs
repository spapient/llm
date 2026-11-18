//! 逐元素乘法：`y[n] = a[n] * b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Mul`] 单态化。

crate::binary_f32!(
    Mul,
    MulF32,
    "mul_f32",
    "../tests/mul_f32_tests.rs",
    |x, y| x * y
);
