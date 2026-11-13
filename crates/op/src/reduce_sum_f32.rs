//! 求和归约：`y[0] = sum(a[0..n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::reduce`] 骨架按 [`Sum`] 单态化。

use crate::reduce::{Reduce, ReduceCache};
use diagnostic::Error;

/// 求和实现：初值 0，折叠用加法。
pub struct Sum;

static CACHE: ReduceCache = ReduceCache::new();

impl Reduce for Sum {
    const OPERANDS: &'static [crate::gemv::OperandSpec] = &crate::reduce::f32_operands();

    fn formula() -> Result<&'static crate::einsum::Formula, Error> {
        CACHE.get::<Self>()
    }

    fn start() -> f32 {
        0.0
    }

    fn fold(acc: f32, x: f32) -> f32 {
        acc + x
    }

    fn input(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.5) - 1.0).collect()
    }
}

/// 求和归约的具体语义别名。
#[annotation::model(name = "reduce_sum_f32", shape = [4096, 1], dtype = "f32", skeleton = "reduce")]
pub type ReduceSumF32 = Sum;

/// 内核入口：结果写进调用方给的单元素缓冲。
///
/// # Errors
///
/// 输出不是单元素时返回说明。
pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::reduce::forward::<Sum>(a, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_matches_the_scalar_reference() {
        let sample = crate::reduce::sample::<Sum>(128, 1);
        let expected = crate::reduce::reference::<Sum>(&sample).expect("参考可计算");
        let mut out = vec![0.0];
        forward(sample.f32("a").expect("a"), &mut out).expect("单元素输出");
        assert_eq!(out, expected);
    }
}
