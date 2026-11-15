//! 残差相加：`y[n] = a[n] + b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Residual`] 单态化。

use crate::einsum::FormulaCache;
use crate::elementwise::Binary;

/// 残差实现：与 `add` 同一段加法，独立成类型以对应网络残差层。
pub struct Residual;

impl Binary for Residual {
    const EXPR: &'static str = "n,n,n->n";
    const OPERANDS: &'static [crate::einsum::OperandSpec] = &crate::elementwise::f32_operands();

    fn cache() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn apply(a: f32, b: f32) -> f32 {
        a + b
    }

    fn input_a(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32) * 0.5).collect()
    }

    fn input_b(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32) * 0.25).collect()
    }
}

/// 残差相加的具体语义别名。
#[annotation::model(name = "residual_add_f32", shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
pub type ResidualAddF32 = Residual;

/// 内核入口：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三个缓冲等长，否则返回说明。
pub fn forward(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::elementwise::forward::<Residual>(a, b, out)
}

#[cfg(test)]
#[path = "residual_add_f32_tests.rs"]
mod tests;
