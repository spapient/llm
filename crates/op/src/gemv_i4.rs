//! `gemv_i4` 的算子语义：4 位打包权重 + 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"4 位打包"这一份格式契约：表达式、操作数位宽、声明粒度、
//! 标度取值与内核本体。模板在 `ops/gemv_i4.k`，片段只与这里对拍。`u8` 承载打包 nibble，
//! 低 4 位在前、高 4 位在后。
//!
//! # 哪些是编译期的，哪些是数据
//!
//! 配置表 [`I4`] 把**声明粒度**放进类型（`Packed<I4Conf, 32>`）：它决定代价里的派生轴除数、
//! 生产形状的合法性与样本里每行的标度个数，三件都在编译期定死。实际分组大小由模板 `gsz`
//! 实参给出，[`forward`] 仍按 `scales` 长度推分组。

use crate::gemv::{self, Format, Packed};
use crate::quant::I4;

/// 量化分组。
///
/// 声明里 `needs.quant.group = ["32"]` 是**能力请求**（原语必须支持多大的组），这里是样本与
/// 代价实际采用的取值。`cutile` 的 `the_model_group_matches_the_declaration` 测试保证两者
/// 不漂移，`#[model]` 生成的编译期断言保证生产形状是它的整数倍。
pub const GROUP: usize = 32;

/// 4 位打包配置标记：把 [`I4`] 一行配置挂进 [`Packed`] 泛型格式。
pub struct I4Conf;

impl gemv::PackedRow for I4Conf {
    const ROW: crate::quant::Row = I4;

    fn cell() -> &'static crate::einsum::FormulaCache {
        static CACHE: crate::einsum::FormulaCache = crate::einsum::FormulaCache::new();
        &CACHE
    }
}

/// 4 位打包格式的语义类型：声明粒度 [`GROUP`]、每行 `m / 2` 个字节。
///
/// `#[model]` 由这份注解生成 [`MODEL`]：生产形状、输出类型与三个单态化入口。别名不实现
/// [`Format`] 就编译不过，所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_i4", shape = [4096, 4096], dtype = "f32")]
pub type I4 = Packed<I4Conf, GROUP>;

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`gemv::reference`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配。分组仍由
/// `scales` 的长度给出，所以同一份权重按 8 或 32 分组调用都能算。
///
/// # Errors
///
/// `out` 为空、`scales` 长度不是 `n` 的整数倍、分组数为 0、分组不是偶数、`m` 不能被分组
/// 整除（奇数尾在此被拦，进不到解包），或 `weights` 长度不足 `n · m / 2` 时返回说明。
pub fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    <I4 as Format>::forward(x, weights, scales, out)
}

#[cfg(test)]
#[path = "../tests/gemv_i4_tests.rs"]
mod tests;
