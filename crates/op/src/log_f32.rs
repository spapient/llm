//! 逐元素自然对数：`y[n] = ln(a[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Log`] 单态化。
//! 定义域为 `(0, +∞)`；`x <= 0` 按 IEEE-754 语义（`-inf` / `NaN`），不在算子层额外判错。
//! 样本夹具覆盖为正数，避免参考对拍里 `log(负数)` 产生 `NaN`。

crate::unary_f32!(
    Log,
    LogF32,
    "log_f32",
    "../tests/log_f32_tests.rs",
    |x| x.ln(),
    |len| (0..len).map(|i| 0.5 + i as f32 * 0.25).collect()
);
