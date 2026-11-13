//! 逐元素乘法：`y[n] = a[n] * b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::elementwise`] 骨架按 [`Mul`] 单态化。

use crate::elementwise::{BinCache, Binary};
use diagnostic::Error;

/// 乘法实现：与 [`crate::add_f32::Add`] 同构，只改逐元素运算。
pub struct Mul;

static CACHE: BinCache = BinCache::new();

impl Binary for Mul {
    const EXPR: &'static str = "n,n,n->n";
    const OPERANDS: &'static [crate::gemv::OperandSpec] = &crate::elementwise::f32_operands();

    fn formula() -> Result<&'static crate::einsum::Formula, Error> {
        CACHE.get::<Self>()
    }

    fn apply(x: f32, y: f32) -> f32 {
        x * y
    }

    fn input_a(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.5) - 1.0).collect()
    }

    fn input_b(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25) + 0.5).collect()
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
mod tests {
    use super::*;

    #[test]
    fn mul_matches_the_scalar_reference() {
        let sample = crate::elementwise::sample::<Mul>(128, 1);
        let expected = crate::elementwise::reference::<Mul>(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        let a = sample.f32("a").expect("a 存在");
        let b = sample.f32("b").expect("b 存在");
        forward(a, b, &mut out).expect("缓冲等长");
        assert_eq!(out, expected);
    }
}
