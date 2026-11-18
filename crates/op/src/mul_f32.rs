//! 逐元素乘法：`y[n] = a[n] * b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Mul`] 单态化。

use crate::einsum::FormulaCache;
use crate::elementwise::Binary;

/// 乘法实现：与 [`crate::add_f32::Add`] 同构，只改逐元素运算。
pub struct Mul;

impl Binary for Mul {
    fn cache() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn apply(x: f32, y: f32) -> f32 {
        x * y
    }
}

/// 乘法的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "mul_f32", shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
pub type MulF32 = Mul;

/// 内核入口：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三个缓冲等长，否则返回说明。
pub fn forward(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::elementwise::forward::<Mul>(a, b, out)
}

#[cfg(test)]
#[path = "../tests/mul_f32_tests.rs"]
mod tests;
