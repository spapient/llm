//! 最大值归约：`y[0] = max(a[0..n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::reduce`] 骨架按 [`Max`] 单态化。

crate::reduce_f32!(
    Max,
    ReduceMaxF32,
    "reduce_max_f32",
    "../tests/reduce_max_f32_tests.rs",
    f32::MIN,
    |acc, x| if x > acc { x } else { acc }
);
