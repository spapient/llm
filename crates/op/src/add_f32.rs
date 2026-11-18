//! 逐元素加法：`y[n] = a[n] + b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成（`tests/contracts.rs` 要求语义表只有这一个生产者），
//! 算术与遍历由 [`crate::elementwise`] 骨架按 [`Add`] 类型单态化。

crate::binary_f32!(
    Add,
    AddF32,
    "add_f32",
    "../tests/add_f32_tests.rs",
    |x, y| x + y
);
