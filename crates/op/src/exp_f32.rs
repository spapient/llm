//! 逐元素指数：`y[n] = exp(a[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Exp`] 单态化。
//! 上溢/下溢按 IEEE-754 语义（`+inf` / `0`），不在算子层额外判错。

crate::unary_f32!(
    Exp,
    ExpF32,
    "exp_f32",
    "../tests/exp_f32_tests.rs",
    |x| x.exp()
);
