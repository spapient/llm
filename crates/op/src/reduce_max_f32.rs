//! 最大值归约：`y[0] = max(a[0..n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::reduce`] 骨架按 [`Max`] 单态化。

use crate::einsum::FormulaCache;
use crate::reduce::Reduce;

/// 最大值实现：初值取 f32 最小值（与内核 `-FLT_MAX` 同档），折叠取较大者。
pub struct Max;

impl Reduce for Max {
    fn cache() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn start() -> f32 {
        f32::MIN
    }

    fn fold(acc: f32, x: f32) -> f32 {
        if x > acc { x } else { acc }
    }
}

/// 最大值归约的具体语义别名。
#[annotation::model(name = "reduce_max_f32", shape = [4096, 1], dtype = "f32", skeleton = "reduce")]
pub type ReduceMaxF32 = Max;

/// 内核入口：结果写进调用方给的单元素缓冲。
///
/// # Errors
///
/// 输出不是单元素时返回说明。
pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::reduce::forward::<Max>(a, out)
}

#[cfg(test)]
#[path = "../tests/reduce_max_f32_tests.rs"]
mod tests;
