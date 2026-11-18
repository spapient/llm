//! 求和归约：`y[0] = sum(a[0..n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::reduce`] 骨架按 [`Sum`] 单态化。

crate::reduce_f32!(
    Sum,
    ReduceSumF32,
    "reduce_sum_f32",
    "../tests/reduce_sum_f32_tests.rs",
    0.0,
    |acc, x| acc + x
);
