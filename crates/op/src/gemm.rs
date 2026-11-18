//! GEMM 家族共享的轴契约、形状投影与格式无关的矩阵乘骨架。
//!
//! `gemm_f32`、`gemm_i8`、`gemm_i4` 是同一副三轴骨架（`m`/`k` 输入、`n` 输出）的三种
//! 权重格式：轴的角色、收缩轴长度、形状投影、代价口径、样本装配、矩阵遍历与尺寸校验完全
//! 相同，差异只在元素类型、位宽、样本取值、打包长度与点积算术。本模块通过 [`Gemm`] trait
//! 收口这部分共享事实；各格式只写一份格式契约，并按格式单态化，不在热路径上引入运行时分发。
//! 与 [`crate::gemv`] 的两轴骨架对称，只是这里多一条收缩轴。

use crate::einsum::Bits;
use crate::gemv::Element;
use crate::model::{AxisRole, AxisSpec, Buffer, Cost, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};

/// 收缩轴长度（固定档）。
pub const K: usize = 8;

/// 三轴：`m`/`k` 输入，`n` 输出。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'm',
    },
    AxisSpec {
        role: AxisRole::Input,
        label: 'k',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'n',
    },
];

/// 形状：输出 `m×n`，收缩轴 `k` 固定。
///
/// # Errors
///
/// 当前实现不会失败。
pub fn project_shape(m: usize, n: usize) -> Result<ShapeContext, Error> {
    ShapeContext::new([('m', m), ('k', K), ('n', n)])
}

/// 从形状上下文取一条轴；缺轴时按 `op` 标签报 [`Code::Invalid`]。
///
/// 三份格式对轴的判读逐字相同，只有诊断里点名的算子名不同——标签作为参数传入，避免三份
/// 平行的取轴辅助各抄一遍。
pub fn axis(shape: &ShapeContext, label: char, op: &str) -> Result<usize, Error> {
    shape
        .get(label)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `{label}`")))
}

/// 矩阵乘格式：一条 GEMM 里随精度变化的那部分。
///
/// 实现者只写"这份格式用什么元素、什么位宽、怎么取样、怎么打包、怎么点积"；代价、样本装配、
/// 矩阵遍历与尺寸校验由本模块骨架（[`cost`] / [`sample`] / [`reference()`] / [`forward`]）
/// 收口。所有关联项都是编译期常量：没有一件需要运行期再判断一次。
pub trait Gemm {
    /// 样本缓冲里的元素类型（[`Values`] 的两个闭集变体）。
    type Elem: Element;

    /// 诊断里点名的算子名：缺轴、尺寸不符时进错误文案。
    const OP: &'static str;

    /// A / B 输入元素位宽；输出恒为 f32。
    const BITS: Bits;

    /// A（`m×K`）确定性取值：同一形状永远得到同一组元素。
    fn sample_a(m: usize) -> Vec<Self::Elem>;

    /// B（`K×n`）确定性取值。
    fn sample_b(n: usize) -> Vec<Self::Elem>;

    /// A 缓冲应有的元素个数（打包后按 [`Elem`](Self::Elem) 计）。
    fn a_len(m: usize) -> usize;

    /// B 缓冲应有的元素个数。
    fn b_len(n: usize) -> usize;

    /// 输出位置 `(i, j)` 的点积：A 第 `i` 行（连续 `K` 个元素）与 B 第 `j` 列（跨步 `n`）。
    fn dot(a: &[Self::Elem], b: &[Self::Elem], n: usize, i: usize, j: usize) -> f32;
}

/// 从样本里取输入缓冲：名字固定 `a` / `b`，元素类型由格式给出。
fn take<'a, G: Gemm>(sample: &'a Sample, name: &str) -> Result<&'a [G::Elem], Error> {
    let buffer = sample
        .buffer(name)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`")))?;
    G::Elem::borrow(&buffer.values).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!(
                "缓冲 `{name}` 不是 {}：这份格式的输入元素类型",
                G::Elem::NAME
            ),
        )
    })
}

/// 标量代价：`flops` 固定 `2·m·k·n`；输入按 [`Gemm::BITS`] 计费，输出每元素 4 字节。
///
/// # Errors
///
/// 形状缺轴时返回说明（带 [`Gemm::OP`] 算子名）。
pub fn cost<G: Gemm>(shape: &ShapeContext) -> Result<Cost, Error> {
    let m = axis(shape, 'm', G::OP)?;
    let k = axis(shape, 'k', G::OP)?;
    let n = axis(shape, 'n', G::OP)?;
    let read = G::BITS.bytes_for((m * k) as u128) + G::BITS.bytes_for((k * n) as u128);
    Ok(Cost {
        flops: (2 * m * k * n) as f64,
        read: read as u64,
        write: (m * n) as u64 * 4,
    })
}

/// 确定性样本：A、B 由格式取值，Y 为 f32 零缓冲。
pub fn sample<G: Gemm>(m: usize, n: usize) -> Sample {
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
        buffers: vec![
            G::Elem::buffer("a", G::sample_a(m)),
            G::Elem::buffer("b", G::sample_b(n)),
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; m * n]),
            },
        ],
    }
}

/// 打包量化格式（每字节 `CPB` 个码）的公共解包、取样与点积助手。
///
/// int2（4 码/字节）与 int4（2 码/字节）只在码宽、掩码、零点与取样偏移上不同；这些事实作为
/// 参数传入，打包位域的字节布局、码位拆解与整数累加只写一份。int8 不打包，不走这里。
pub mod packed {
    use super::K;

    /// 每字节 `cpb` 个码对应的位宽。
    const fn bits(cpb: usize) -> usize {
        8 / cpb
    }

    /// 解包字节内第 `slot` 个码，按对称零点回退成有符号整数。
    pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: i32) -> i8 {
        let width = bits(cpb);
        let mask = (1u8 << width) - 1;
        let value = (byte >> (slot * width)) & mask;
        (value as i32 - zero) as i8
    }

    /// 确定性打包位模式：`len` 个码按 `cpb` 每字节装一行；码值取
    /// `(字节下标 · seed + offsets[c]) mod 2^width`。
    pub fn pack(len: usize, seed: usize, cpb: usize, offsets: &[u32]) -> Vec<u8> {
        let width = bits(cpb);
        let modulus = 1usize << width;
        (0..len / cpb)
            .map(|byte| {
                (0..cpb).fold(0u8, |acc, code| {
                    let value = ((byte * seed + offsets[code] as usize) % modulus) as u8;
                    acc | (value << (code * width))
                })
            })
            .collect()
    }

    /// 打包矩阵点积：A 第 `i` 行与 B 第 `j` 列，收缩轴固定 [`K`]，整数累加后转 f32。
    pub fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize, cpb: usize, zero: i32) -> f32 {
        let bytes_per_row = K / cpb;
        let mut acc: i32 = 0;
        for p in 0..K {
            let x = unpack(a[i * bytes_per_row + p / cpb], p % cpb, cpb, zero) as i32;
            let y_pos = p * n + j;
            let y = unpack(b[y_pos / cpb], y_pos % cpb, cpb, zero) as i32;
            acc += x * y;
        }
        acc as f32
    }
}

/// 矩阵遍历：外层只写一份，内层点积落在 [`Gemm::dot`]。
pub fn matmul<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = G::dot(a, b, n, i, j);
        }
    }
}

/// 标量参考：与内核算术同源，区别只在输出自带分配。
pub fn reference<G: Gemm>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let m = sample.m;
    let n = sample.n;
    let a = take::<G>(sample, "a")?;
    let b = take::<G>(sample, "b")?;
    let mut out = vec![0.0; m * n];
    matmul::<G>(a, b, m, n, &mut out);
    Ok(out)
}

/// 内核本体：先校验打包长度，再把结果写进调用方给的缓冲。
///
/// # Errors
///
/// 三项缓冲长度不符时返回说明（带 [`Gemm::OP`] 算子名）。
pub fn forward<G: Gemm>(
    a: &[G::Elem],
    b: &[G::Elem],
    m: usize,
    n: usize,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    if a.len() != G::a_len(m) || b.len() != G::b_len(n) || out.len() != m * n {
        return Err(Error::new(Code::Invalid, format!("{} 缓冲尺寸不符", G::OP)));
    }
    matmul::<G>(a, b, m, n, out);
    Ok(())
}

/// 各打包 / 不打包格式共用的对拍：内核算出的结果必须与标量参考逐位一致。
///
/// 三条格式的测试体只有算子名不同，断言与尺寸口径收口到这里，避免三份平行用例各抄一遍。
#[cfg(test)]
pub(crate) fn matches_reference<G: Gemm>() {
    let sample = sample::<G>(16, 12);
    let expected = reference::<G>(&sample).expect("参考可计算");
    let a = take::<G>(&sample, "a").expect("a");
    let b = take::<G>(&sample, "b").expect("b");
    let mut out = vec![0.0; 16 * 12];
    forward::<G>(a, b, 16, 12, &mut out).expect("尺寸符");
    assert_eq!(out, expected);
}
