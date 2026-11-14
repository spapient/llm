//! 逐元素加法：`y[n] = a[n] + b[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成（`tests/contracts.rs` 要求语义表只有这一个生产者），
//! 算术与遍历由 [`crate::elementwise`] 骨架按 [`Add`] 类型单态化。

use crate::einsum::FormulaCache;
use crate::elementwise::Binary;

/// 加法实现：给 [`Binary`] 提供表达式、操作数与逐元素运算。
pub struct Add;

impl Binary for Add {
    const EXPR: &'static str = "n,n,n->n";
    const OPERANDS: &'static [crate::einsum::OperandSpec] = &crate::elementwise::f32_operands();

    fn cache() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn apply(x: f32, y: f32) -> f32 {
        x + y
    }

    fn input_a(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.5) - 1.0).collect()
    }

    fn input_b(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25) + 0.5).collect()
    }
}

/// 加法的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "add_f32", shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
pub type AddF32 = Add;

/// 内核入口：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三个缓冲等长，否则返回说明。
pub fn forward(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::elementwise::forward::<Add>(a, b, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_the_scalar_reference() {
        let sample = crate::elementwise::sample::<Add>(128, 1);
        let expected = crate::elementwise::reference::<Add>(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        let a = sample.f32("a").expect("a 存在");
        let b = sample.f32("b").expect("b 存在");
        forward(a, b, &mut out).expect("缓冲等长");
        assert_eq!(out, expected);
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let mut out = vec![0.0; 2];
        assert!(forward(&[1.0, 2.0], &[1.0], &mut out).is_err());
    }
}
