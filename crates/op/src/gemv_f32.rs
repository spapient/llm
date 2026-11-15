//! `gemv_f32` 的算子语义：稠密 f32 权重投影。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"稠密权重"这一份格式契约：表达式、操作数位宽、分块宽度
//! 与内核本体。模板在 `ops/gemv_f32.k`，本模块是它的第二份实现（cutile README §11.1）：片段只与这里
//! 对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。
//!
//! 与 [`crate::gemv_i4`] 的差别只有两点，而且两点都是**类型参数**：权重元素是 `f32` 而不是
//! 打包字节，形状粒度是 1 而不是量化分组。代价、样本、校验、遍历四条路径一份代码也没有重写。

use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Format};
use diagnostic::{Code, Error};

/// 分块宽度：模板每轮处理这么多元素，标量参考按同一个宽度分块累加。
///
/// 它不是"形状粒度"：稠密权重没有分组约束，最后一块可以短一点（模板用 `len` 截断）。
/// 参考实现必须用同一个宽度分块，否则对拍比的是两种不同的舍入顺序。
pub const BLOCK: usize = 32;

/// 稠密格式的语义类型：分块宽度 [`BLOCK`]、任意输入宽度都合法。
///
/// `#[model]` 由这份注解生成 [`MODEL`]：生产形状、输出类型与三个单态化入口。别名不实现
/// [`Format`] 就编译不过，所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_f32", shape = [4096, 4096], dtype = "f32")]
pub type F32 = gemv::Dense<BLOCK>;

impl<const B: usize> Format for gemv::Dense<B> {
    type Elem = f32;

    /// `w[nm] · x[m] → y[n]`：f32 权重没有派生轴，所以这里比 int4 少一个 `g`。
    const EXPR: &'static str = "nm,m,n->n";

    /// 三个操作数都是 f32：两个读、一个写。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", Bits::ThirtyTwo),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    /// 稠密权重没有分组约束：任意输入宽度都合法。
    const GRAIN: usize = 1;

    const DERIVED: Option<(char, char, usize)> = None;

    const SCALES: Option<fn(usize) -> f32> = None;

    const STEP: usize = B;

    fn cache() -> &'static FormulaCache {
        // 缓存落在这个 impl 里：`Dense<B>` 的表达式与块宽无关，一份缓存覆盖所有实例。
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn row(m: usize) -> usize {
        m
    }

    fn weights(m: usize, n: usize) -> Vec<f32> {
        (0..n * m)
            .map(|index| ((index * 37) % 251) as f32 * 0.125 - 8.0)
            .collect()
    }

    fn forward(
        x: &[f32],
        weights: &[f32],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        // 块宽进源码就是常数：单态化后编译器看到的是 `chunks(32)` 而不是 `chunks(gsz)`。
        const { assert!(B > 0, "累加块宽必须是正数：`chunks(0)` 会 panic") };
        let m = x.len();
        let n = out.len();

        dense_checks(scales, weights.len(), m, n)?;

        // 每行只看自己那一段：切片一次算好，内层就不再做 `row × 列数 + base` 的乘加。
        for (row, value) in out.iter_mut().enumerate() {
            *value = dot_row::<B>(&weights[row * m..][..m], x);
        }
        Ok(())
    }
}

/// 稠密格式的前置校验：输出非空、没有标度、权重够长。
///
/// 「没有标度」是**格式**判据而不是骨架判据：拿着标度调用说明调用方把两条格式搞混了。
/// 静默忽略会让"换错算子"表现成数值正确，那比报错难查得多。
fn dense_checks(scales: &[f32], weights: usize, m: usize, n: usize) -> diagnostic::Result<()> {
    gemv::non_empty(n)?;
    if scales.is_empty() {
        gemv::required_weights(weights, n, m)?;
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("稠密格式没有标度，收到 {} 个", scales.len()),
    ))
}

/// 一行的点积：按块宽推进，块内交给迭代器展开。
fn dot_row<const B: usize>(w_row: &[f32], x: &[f32]) -> f32 {
    w_row
        .chunks(B)
        .zip(x.chunks(B))
        .map(|(w_block, x_block)| {
            w_block
                .iter()
                .zip(x_block)
                .map(|(weight, input)| weight * input)
                .sum::<f32>()
        })
        .sum()
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`gemv::reference`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配。
///
/// # Errors
///
/// `out` 为空或 `weights` 长度不足 `n · m` 时返回说明。
pub fn forward(x: &[f32], weights: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    <F32 as Format>::forward(x, weights, &[], out)
}

#[cfg(test)]
#[path = "../tests/gemv_f32_tests.rs"]
mod tests;
