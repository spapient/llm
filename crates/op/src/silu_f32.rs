//! 逐元素 SiLU 激活：`y[n] = a[n] * sigmoid(a[n])`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Silu`] 单态化。

use crate::unary::{Unary, UnaryCache};
use diagnostic::Error;

/// SiLU 实现：`x / (1 + exp(-x))`。
pub struct Silu;

static CACHE: UnaryCache = UnaryCache::new();

impl Unary for Silu {
    const EXPR: &'static str = "n,n->n";
    const OPERANDS: &'static [crate::gemv::OperandSpec] = &crate::unary::f32_operands();

    fn formula() -> Result<&'static crate::einsum::Formula, Error> {
        CACHE.get::<Self>()
    }

    fn apply(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn input(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25) - 4.0).collect()
    }
}

/// SiLU 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "silu_f32", shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
pub type SiluF32 = Silu;

/// 内核入口：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::unary::forward::<Silu>(a, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_matches_the_scalar_reference() {
        let sample = crate::unary::sample::<Silu>(128, 1);
        let expected = crate::unary::reference::<Silu>(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        let a = sample.f32("a").expect("a 存在");
        forward(a, &mut out).expect("缓冲等长");
        assert_eq!(out, expected);
    }
}
