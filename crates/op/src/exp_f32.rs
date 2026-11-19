//! 逐元素指数：`y[n] = exp(a[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Exp`] 单态化。
//! 上溢/下溢按 IEEE-754 语义（`+inf` / `0`），不在算子层额外判错。
//! 样本夹具覆盖到 `[-8, 8]`：默认线性夹具在 4096x4096 形状下会 `exp(4.19e6)` 溢出成 `inf`，
//! 让真机对拍报 `max_rel_err inf`；有界周期夹具避开上溢，同时仍遍历正负两侧。

crate::unary_f32!(
    Exp,
    ExpF32,
    "exp_f32",
    "../tests/exp_f32_tests.rs",
    |x| x.exp(),
    |len| (0..len).map(|i| ((i % 65) as f32) * 0.25 - 8.0).collect()
);
