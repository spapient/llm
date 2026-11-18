//! `gemv_i2` 的算子语义：2 位打包权重 + 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"2 位打包"这一份格式契约：表达式、操作数位宽、声明粒度、
//! 标度取值与内核本体。模板在 `ops/gemv_i2.k`，本模块是它的第二份实现（cutile README §11.1）：
//! 片段只与这里对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生
//! （[`crate::sig`]）。
//!
//! # 哪些是编译期的，哪些是数据
//!
//! 配置表 [`const@I2`] 把**声明粒度**放进类型（`Packed<I2Conf, 32>`）：它决定代价里的派生轴除数、
//! 生产形状的合法性与样本里每行的标度个数，三件都在编译期定死。实际的**分组大小**不在这里——
//! 模板把它当 `int gsz` 实参收，所以 [`forward`] 仍按 `scales` 的长度推分组：一份 32 分组的权重
//! 与一份 8 分组的权重是同一个类型的两次调用，而不是两份单态化副本。
//!
//! # 打包边界：非 4 倍数的尾码在解包之前就被拦下
//!
//! 四个 2 位码共用一个字节，因此输入宽度 `m` 必须为 4 的倍数、且整除分组：分组不是 4 的倍数、
//! 或 `m` 不能被分组整除时，[`forward`] 在进入解包循环前返回错误，解包端永远拿不到非空余数。
//! 这里不做运行期补齐：weights 层把 2 位当作一整块紧贴的位数组（按行不补齐），按行补零码是
//! 另一种打包格式，会与那份字节自证口径冲突。病态的非整宽度在边界处报错，而不是让残缺码
//! 参与点积。

use crate::gemv::{self, Format, Packed};
use crate::quant::I2;

/// 量化分组。
///
/// 声明里 `needs.quant.group = ["32"]` 是**能力请求**（原语必须支持多大的组），这里是样本与
/// 代价实际采用的取值。`cutile` 的 `the_model_group_matches_the_declaration` 测试保证两者
/// 不漂移，`#[model]` 生成的编译期断言保证生产形状是它的整数倍。
pub const GROUP: usize = 32;

/// 2 位打包配置标记：把 [`const@I2`] 一行配置挂进 [`Packed`] 泛型格式。
pub struct I2Conf;

impl gemv::PackedRow for I2Conf {
    const ROW: crate::quant::Row = I2;

    fn cell() -> &'static crate::einsum::FormulaCache {
        static CACHE: crate::einsum::FormulaCache = crate::einsum::FormulaCache::new();
        &CACHE
    }
}

/// 2 位打包格式的语义类型：声明粒度 [`GROUP`]、每行 `m / 4` 个字节。
///
/// `#[model]` 由这份注解生成 [`MODEL`]：生产形状、输出类型与三个单态化入口。别名不实现
/// [`Format`] 就编译不过，所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_i2", shape = [4096, 4096], dtype = "f32")]
pub type I2 = Packed<I2Conf, GROUP>;

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`gemv::reference`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配：重放时
/// 每一步都要算，稳定态不该再向分配器要一块新的 `Vec`。分组仍由 `scales` 的长度给出，所以
/// 同一份权重按 8 或 32 分组调用都能算。
///
/// # Errors
///
/// `out` 为空、`scales` 长度不是 `n` 的整数倍、分组数为 0、分组不是 4 的倍数、`m` 不能被分组
/// 整除（非 4 倍数尾码在此被拦，进不到解包），或 `weights` 长度不足 `n · m / 4` 时返回说明。
pub fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    <I2 as Format>::forward(x, weights, scales, out)
}

#[cfg(test)]
#[path = "../tests/gemv_i2_tests.rs"]
mod tests;
