//! `gemv_i4` 的算子语义：4 位打包权重 + 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"4 位打包"这一份格式契约：表达式、操作数位宽、声明粒度、
//! 标度取值与内核本体。模板在 `ops/gemv_i4.k`，本模块是它的第二份实现（cutile README §11.1）：
//! 片段只与这里对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生
//! （[`crate::sig`]）。
//!
//! # 哪些是编译期的，哪些是数据
//!
//! [`I4`] 把**声明粒度**放进类型（`Quantized<32>`）：它决定代价里的派生轴除数、生产形状的
//! 合法性与样本里每行的标度个数，三件都在编译期定死。实际的**分组大小**不在这里——模板把
//! 它当 `int gsz` 实参收，所以 [`forward`] 仍按 `scales` 的长度推分组：一份 32 分组的权重与
//! 一份 8 分组的权重是同一个类型的两次调用，而不是两份单态化副本。
//!
//! # 打包边界：奇数尾在解包之前就被拦下
//!
//! 两个 4 位码共用一个字节，因此输入宽度 `m` 必须为偶数、且整除分组：分组为奇数、或 `m`
//! 不能被分组整除时，[`forward`] 在进入解包循环前返回错误，`as_chunks::<2>()` 永远拿不到
//! 非空余数。这里不做运行期补齐：weights 层把 4 位当作一整块紧贴的位数组（按行不补齐），
//! 按行补零 nibble 是另一种打包格式，会与那份字节自证口径冲突。病态的奇数宽度在边界处报错，
//! 而不是让半个 nibble 参与点积。

use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Format};
use diagnostic::{Code, Error};

/// 量化分组。
///
/// 声明里 `needs.quant.group = ["32"]` 是**能力请求**（原语必须支持多大的组），这里是样本与
/// 代价实际采用的取值。`cutile` 的 `the_model_group_matches_the_declaration` 测试保证两者
/// 不漂移，`#[model]` 生成的编译期断言保证生产形状是它的整数倍。
pub const GROUP: usize = 32;

/// 四位无符号码的解包结果：`nibble − 8`。
///
/// 查表而不是每次做 `i32 → f32` 转换：解包是这条内核最内层的循环，每元素省一次转换就是
/// 整个 GEMV 的省；16 个取值都是精确的小整数，表里没有舍入。
const DEQUANT: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];

/// 4 位打包格式的语义类型：声明粒度 [`GROUP`]、每行 `m / 2` 个字节。
///
/// `#[model]` 由这份注解生成 [`MODEL`]：生产形状、输出类型与三个单态化入口。别名不实现
/// [`Format`] 就编译不过，所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_i4", shape = [4096, 4096], dtype = "f32")]
pub type I4 = gemv::Quantized<GROUP>;

impl<const G: usize> Format for gemv::Quantized<G> {
    type Elem = u8;

    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出来的派生轴，
    /// 只计流量、不计运算次数。
    const EXPR: &'static str = "nm,m,ng,n->n";

    /// 三个输入都是读，只有输出是写；位宽就是解包前后各占多少。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", Bits::Four),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::read("scales", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    const GRAIN: usize = G;

    const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', G));

    const SCALES: Option<fn(usize) -> f32> = Some(gemv::scale_of);

    const STEP: usize = G;

    fn cache() -> &'static FormulaCache {
        // 缓存落在这个 impl 里：`Quantized<G>` 的表达式与粒度无关，一份缓存覆盖所有实例。
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn row(m: usize) -> usize {
        // 两个 4 位码共用一个字节：粒度必须是正偶数，否则"每行 m / 2 字节"没有意义。
        // 单态化时求值，运行期一条指令都不留。
        const {
            assert!(
                G > 0 && G.is_multiple_of(2),
                "4 位打包的粒度必须是正偶数：两个元素共用一个字节"
            )
        };
        m / 2
    }

    fn weights(m: usize, n: usize) -> Vec<u8> {
        (0..n * (m / 2))
            .map(|index| ((index * 37) % 251) as u8)
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

        let (groups, group) = quantized_groups(m, n, scales.len(), weights.len())?;

        // 每行只看自己那一段：切片一次算好，内层就不再做 `row × 列数 + g` 的乘加与边界检查。
        let half = m / 2;
        let per_group = group / 2;
        for (row, value) in out.iter_mut().enumerate() {
            let w_row = &weights[row * half..][..half];
            let s_row = &scales[row * groups..][..groups];
            *value = dot_row(w_row, x, s_row, per_group, group);
        }
        Ok(())
    }
}

/// 量化格式的前置约束：骨架的分组推导 + 本节格式自己的"分组必须是偶数"。
///
/// 两个 4 位码共用一个字节，所以分组为奇数时"每行 m / 2 字节"没有意义；分组为偶且整除 `m`
/// 时 `m` 必为偶，每组元素成对，解包的余数由此被排除在循环之外。
fn quantized_groups(
    m: usize,
    n: usize,
    scales: usize,
    weights: usize,
) -> diagnostic::Result<(usize, usize)> {
    let (groups, group) = gemv::scaled_groups(m, n, scales)?;
    if !group.is_multiple_of(2) {
        return Err(Error::new(
            Code::Invalid,
            format!("分组 {group} 必须是偶数"),
        ));
    }
    gemv::required_weights(weights, n, m / 2)?;
    Ok((groups, group))
}

/// 一行的点积：按分组推进，组内解包两个 4 位码，整组共用同一个标度。
///
/// 累加顺序与朴素写法逐位一致（同一个字节的低半字节在前），否则舍入顺序一变，
/// 对拍就不再是同一个算术。每组 `group` 为偶（前置约束），`as_chunks::<2>()` 余数为空，
/// 解包只读成对的 nibble，不触碰补齐位。
fn dot_row(packed_row: &[u8], x: &[f32], s_row: &[f32], per_group: usize, group: usize) -> f32 {
    let mut acc = 0.0f32;
    for ((packed, scale), xs) in packed_row
        .chunks_exact(per_group)
        .zip(s_row)
        .zip(x.chunks_exact(group))
    {
        let mut dot = 0.0f32;
        for (byte, pair) in packed.iter().zip(xs.as_chunks::<2>().0) {
            dot += DEQUANT[usize::from(*byte & 0x0F)] * pair[0];
            dot += DEQUANT[usize::from(*byte >> 4)] * pair[1];
        }
        acc += dot * scale;
    }
    acc
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`gemv::reference`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配：重放时
/// 每一步都要算，稳定态不该再向分配器要一块新的 `Vec`。分组仍由 `scales` 的长度给出，所以
/// 同一份权重按 8 或 32 分组调用都能算。
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
mod tests;
