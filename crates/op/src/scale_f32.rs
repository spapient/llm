//! 标量缩放：`y[n] = alpha * a[n]`。
//!
//! 语义表由 `#[annotation::model]` 生成，算术与遍历由 [`crate::unary`] 骨架按 [`Scale`] 单态化。
//! 标量参数框架只支持整数，故 `alpha` 写成与内核模板一致的常量。

use crate::unary::{Unary, UnaryCache};
use diagnostic::Error;

/// 样本固定的缩放系数（内核模板里写同一常量）。
pub const ALPHA: f32 = 0.5;

/// 缩放实现：给 [`Unary`] 提供表达式、操作数与逐元素运算。
pub struct Scale;

static CACHE: UnaryCache = UnaryCache::new();

impl Unary for Scale {
    const EXPR: &'static str = "n,n->n";
    const OPERANDS: &'static [crate::gemv::OperandSpec] = &crate::unary::f32_operands();

    fn formula() -> Result<&'static crate::einsum::Formula, Error> {
        CACHE.get::<Self>()
    }

    fn apply(x: f32) -> f32 {
        ALPHA * x
    }

    fn input(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25) - 2.0).collect()
    }
}

/// 缩放的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
#[annotation::model(name = "scale_f32", shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
pub type ScaleF32 = Scale;

/// 内核入口：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，否则返回说明。
pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    crate::unary::forward::<Scale>(a, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_matches_the_scalar_reference() {
        let sample = crate::unary::sample::<Scale>(128, 1);
        let expected = crate::unary::reference::<Scale>(&sample).expect("参考可计算");
        let mut out = vec![0.0; 128];
        let a = sample.f32("a").expect("a 存在");
        forward(a, &mut out).expect("缓冲等长");
        assert_eq!(out, expected);
    }
}
