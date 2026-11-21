//! `gemv_f32` 的算子语义：稠密 f32 权重投影。
//!
//! 骨架在 [`crate::gemv`]，本模块只有"稠密权重"这一份逐元素配置：表达式、操作数位宽、分块宽度
//! 与元素反量化。模板在 `ops/gemv_f32.k`，本模块是它的第二份实现：片段只与这里对拍。
//! 内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。
use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Plain, PlainFmt};

/// 分块宽度：模板每轮处理这么多元素，标量参考按同一个宽度分块累加。
pub const BLOCK: usize = 32;

/// 稠密逐元素的配置：`f32` 权重、无标度、任意输入宽度，按 [`BLOCK`] 分块累加。
impl<const B: usize> crate::formula::FormulaOwner for gemv::Dense<B> {
    /// `w[nm] · x[m] → y[n]`：无派生轴，宽度不参与分组。
    const EXPR: &'static str = "nm,m,n->n";

    /// 三个操作数都是 f32：`w` 是 `m·n` 个元素、`x` 是 `m` 个、`y` 是 `n` 个。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", Bits::ThirtyTwo),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    const FRAME: &'static str = "op::gemv::formula";

    fn cache() -> &'static FormulaCache {
        crate::formula_cell!()
    }
}

/// 稠密 f32 的样本与反量化（恒等）：骨架的一切算术都由 [`PlainFmt`] 那一份给出。
impl<const B: usize> Plain for gemv::Dense<B> {
    type Elem = f32;

    /// 稠密算子的形状粒度是 1：一条只为对齐 nibble 的约束不该溢出到 f32 权重上。
    const GRAIN: usize = 1;

    const DERIVED: Option<(char, char, usize)> = None;

    const SCALES: Option<fn(usize) -> f32> = None;

    /// 模板每轮推进的元素数就是分块宽度（样本里的 `gsz` 与分块累加的宽度同源）。
    const STEP: usize = B;

    fn weights(m: usize, n: usize) -> Vec<f32> {
        (0..n * m)
            .map(|index| ((index * 37) % 251) as f32 * 0.125 - 8.0)
            .collect()
    }

    fn to_f32(elem: f32) -> f32 {
        elem
    }
}

/// 稠密格式的语义类型：分块宽度 [`BLOCK`]、任意输入宽度都合法。
#[annotation::model(name = "gemv_f32", shape = [4096, 4096], dtype = "f32")]
pub type F32 = PlainFmt<gemv::Dense<BLOCK>>;

// 基座契约（ADR 0050）：粒度与三条行为从格式契约转发。
crate::operator_via!(grained F32, "gemv_f32", 2, (4096, 4096), true, gemv, gemv::AXES);

/// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
pub fn forward(x: &[f32], weights: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    <F32 as gemv::Format>::forward(x, weights, &[], out)
}

#[cfg(test)]
mod tests {
    include!("../../tests/gemv.rs");
}
