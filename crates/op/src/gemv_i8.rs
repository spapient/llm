//! `gemv_i8` 的算子语义：对称 8 位权重（不打包、零点 0）+ 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"对称 int8"这一份格式契约：表达式、操作数位宽、声明粒度、
//! 标度取值与内核本体。模板在 `ops/gemv_i8.k`，片段只与这里对拍。`u8` 承载 int8 位模式，
//! 算术时按 `i8` 解释。

use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Format};

/// 量化分组（声明粒度，与 `gemv_i4` 同档）。
pub const GROUP: usize = 32;

/// 标度取值：`0.03125` 的 1..=4 倍，落在 2 的负幂网格，f32 精确。
fn scale_of(index: usize) -> f32 {
    0.031_25 * (1 + (index % 4)) as f32
}

/// 对称 int8 格式的语义类型：声明粒度 [`GROUP`]、每行 `m` 个字节。
#[annotation::model(name = "gemv_i8", shape = [4096, 4096], dtype = "f32")]
pub type I8 = gemv::SymI8<GROUP>;

impl<const G: usize> Format for gemv::SymI8<G> {
    type Elem = u8;

    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出的派生轴。
    const EXPR: &'static str = "nm,m,ng,n->n";

    /// 权重是 8 位（每元素 1 字节），其余同 4 位格式。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", Bits::Eight),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::read("scales", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    const GRAIN: usize = G;

    const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', G));

    const SCALES: Option<fn(usize) -> f32> = Some(scale_of);

    const STEP: usize = G;

    fn cache() -> &'static FormulaCache {
        // 一份缓存覆盖所有 SymI8<G> 实例：表达式与粒度无关。
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn row(m: usize) -> usize {
        // 不打包：每元素一个字节。
        m
    }

    fn weights(m: usize, n: usize) -> Vec<u8> {
        // 取对称小整数（-8..7）的 int8 位模式，避免大整数在 f32 网格外引入舍入。
        (0..n * m)
            .map(|index| ((index * 37) % 16) as i8 - 8)
            .map(|value| value as u8)
            .collect()
    }

    fn forward(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let m = x.len();
        let n = out.len();
        let (groups, group) = gemv::scaled_groups(m, n, scales.len())?;
        gemv::required_weights(weights.len(), n, m)?;

        // 每行只看自己那一段；每组只乘一次 scale。u8 按 i8 解释（对称零点 0）。
        for (row, value) in out.iter_mut().enumerate() {
            let w_row = &weights[row * m..][..m];
            let s_row = &scales[row * groups..][..groups];
            let mut acc = 0.0f32;
            for ((w_chunk, scale), x_chunk) in w_row.chunks(group).zip(s_row).zip(x.chunks(group)) {
                // u8 按 i8 解释（对称零点 0）：先点积，整组只乘一次标度。
                let dot: f32 = w_chunk
                    .iter()
                    .zip(x_chunk)
                    .map(|(packed, xv)| ((*packed as i8) as f32) * xv)
                    .sum();
                acc += dot * scale;
            }
            *value = acc;
        }
        Ok(())
    }
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
///
/// # Errors
///
/// 与 [`Format::forward`] 返回的说明一致。
pub fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    <I8 as Format>::forward(x, weights, scales, out)
}
