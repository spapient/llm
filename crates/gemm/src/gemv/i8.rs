//! `gemv_i8` 的算子语义：对称 8 位权重（不打包、零点 0）+ 每 [`GROUP`] 个元素一条标度。
use crate::einsum::FormulaCache;
use crate::gemv::{self, Plain, PlainFmt};
use crate::quant::I8;

/// 量化分组（声明粒度）：唯一事实源在 [`GROUP`](crate::gemv::GROUP)。
pub const GROUP: usize = gemv::GROUP;

/// 逐元素对称 int8 的配置：`u8` 承载位模式、按 `i8` 反量化、每 [`GROUP`] 个元素一条标度。
impl<const G: usize> crate::formula::FormulaOwner for gemv::SymI8<G> {
    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出的派生轴。
    const EXPR: &'static str = "nm,m,ng,n->n";

    // 权重是 8 位（每元素 1 字节），其余同打包格式：操作数表走同一条展开。
    scaled_operands!(I8.bits);

    const FRAME: &'static str = "op::gemv::formula";

    fn cache() -> &'static FormulaCache {
        crate::formula_cell!()
    }
}

/// 逐元素对称 int8 的配置：样本取对称小整数，反量化按 `i8` 解释位模式。
impl<const G: usize> Plain for gemv::SymI8<G> {
    type Elem = u8;

    scaled_grain!(G);

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
#[annotation::model(name = "gemv_i8", shape = [4096, 4096], dtype = "f32")]
pub type I8 = PlainFmt<gemv::SymI8<GROUP>>;

// 基座契约（ADR 0050）：粒度与三条行为从格式契约转发；量化算子不可微。
crate::operator_via!(grained I8, "gemv_i8", 3, (4096, 4096), false, super, gemv::AXES);

/// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
pub fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    <I8 as gemv::Format>::forward(x, weights, scales, out)
}

/// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只读一遍。
pub fn forward_batch(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    geo: gemv::Geometry,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    gemv::forward_plain_batch::<gemv::SymI8<GROUP>>(x, weights, scales, geo, out)
}

#[cfg(test)]
mod tests {
    include!("../../tests/gemv_i8.rs");
}
