//! 逐元素 SiLU 激活：`y[n] = a[n] * sigmoid(a[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Silu`] 单态化。

crate::unary_f32!(
    Silu,
    SiluF32,
    "silu_f32",
    "../tests/silu_f32_tests.rs",
    |x| x / (1.0 + (-x).exp())
);
