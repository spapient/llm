//! 逐元素 GELU 激活（精确 erf 版）：`y[n] = 0.5*a[n]*(1 + erf(a[n]/sqrt(2)))`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Gelu`] 单态化。
//! C 侧与 Rust 侧用同一份 Abramowitz-Stegun 7.1.26 系数（绝对误差上界 1.5e-7），两侧只剩
//! f32 舍入级差异，对拍 max_rel_err 在 1e-10 量级。

use crate::einsum::FormulaCache;
use crate::unary::Unary;

/// GELU 实现。
pub struct Gelu;

/// 标量误差函数：Abramowitz-Stegun 手册公式 7.1.26（`|误差| < 1.5e-7`）。
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.327_591_1;
    // 字面量写成 f32 可精确往返的最短形式（高精度写法在 f32 下会舍入到同一个值）。
    let a1 = 0.254_829_6;
    let a2 = -0.284_496_72;
    let a3 = 1.421_413_8;
    let a4 = -1.453_152_1;
    let a5 = 1.061_405_4;
    let t = 1.0 / (1.0 + p * x);
    let series = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let y = 1.0 - series * (-x * x).exp();
    sign * y
}

impl Unary for Gelu {
    fn cache() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn apply(x: f32) -> f32 {
        0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
    }

    fn input(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25) - 4.0).collect()
    }
}

/// GELU 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "gelu_f32", shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
pub type GeluF32 = Gelu;

/// 内核入口：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::unary::forward::<Gelu>(a, out)
}

#[cfg(test)]
#[path = "../tests/gelu_f32_tests.rs"]
mod tests;
