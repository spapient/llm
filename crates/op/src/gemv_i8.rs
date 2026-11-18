//! `gemv_i8` 的算子语义：对称 8 位权重（不打包、零点 0）+ 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"对称 int8"这一份逐元素配置：表达式、操作数位宽、声明粒度、
//! 标度取值与元素反量化。模板在 `ops/gemv_i8.k`，片段只与这里对拍。`u8` 承载 int8 位模式，
//! 算术时按 `i8` 解释。

use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Plain, PlainFmt};
use crate::quant::I8;

/// 量化分组（声明粒度）。
pub const GROUP: usize = 32;

/// 逐元素对称 int8 的配置：`u8` 承载位模式、按 `i8` 反量化、每 [`GROUP`] 个元素一条标度。
impl<const G: usize> Plain for gemv::SymI8<G> {
    type Elem = u8;

    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出的派生轴。
    const EXPR: &'static str = "nm,m,ng,n->n";

    /// 权重是 8 位（每元素 1 字节），其余同 4 位格式。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", I8.bits),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::read("scales", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    const GRAIN: usize = G;

    const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', G));

    const SCALES: Option<fn(usize) -> f32> = Some(gemv::scale_of);

    const STEP: usize = G;

    fn cell() -> &'static FormulaCache {
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn weights(m: usize, n: usize) -> Vec<u8> {
        // 取对称小整数（-8..7）的 int8 位模式，避免大整数在 f32 网格外引入舍入。
        (0..n * m)
            .map(|index| ((index * 37) % 16) as i8 - 8)
            .map(|value| value as u8)
            .collect()
    }

    fn to_f32(elem: u8) -> f32 {
        (elem as i8) as f32
    }
}

/// 对称 int8 格式的语义类型：声明粒度 [`GROUP`]、每行 `m` 个字节。
///
/// `#[model]` 由这份注解生成 [`MODEL`]；别名不实现 [`Format`](crate::gemv::Format) 就编译不过，
/// 所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_i8", shape = [4096, 4096], dtype = "f32")]
pub type I8 = PlainFmt<gemv::SymI8<GROUP>>;

/// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
///
/// # Errors
///
/// 与 [`Format::forward`](crate::gemv::Format::forward) 返回的说明一致。
pub fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    <I8 as gemv::Format>::forward(x, weights, scales, out)
}

#[cfg(test)]
#[path = "../tests/gemv_i8_tests.rs"]
mod tests;
