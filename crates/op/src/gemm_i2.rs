//! 对称 int2 矩阵乘：`Y[m,n] = Σ_k i2(A[m,k]) · i2(B[k,n])`。
//!
//! 骨架在 [`crate::gemm`]，本文件只有"对称 int2"这一份格式契约：A、B 是打包 2 位码（每字节
//! 四个，低 2 位在前），对称零点 [`ZERO`]。点积在 [`Gemm::dot`] 内按需解包成有符号整数，
//! 再在 `i32` 上精确累加（对应设备侧 `mma` 的整数路径），最后转 f32。CPU 没有
//! mma，退化为标量整数点积；整数累加逐位精确，与设备指令无舍入分歧。

use crate::einsum::Bits;
use crate::gemm::{self, Gemm};
use crate::model::{Cost, Sample, ShapeContext};
use diagnostic::Error;

// 收缩轴、三轴契约、形状投影复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, K, project_shape};

/// 对称零点：2 位码 0..3 减去 2 得到 -2..=1。
pub const ZERO: i32 = 2;

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI2;

/// 解包一个 2 位码：`slot` 取字节内第 `slot` 个 2 位段（0 最低位），按 [`ZERO`] 回退到 -2..=1。
fn code2(byte: u8, slot: usize) -> i8 {
    let value = (byte >> (2 * slot)) & 0x03;
    (value as i32 - ZERO) as i8
}

/// 打包确定性位模式：每字节四个码（低 2 位在前、高位在后）。
fn pack(len: usize, seed: usize) -> Vec<u8> {
    (0..len / 4)
        .map(|i| {
            let c0 = ((i * seed + 1) % 4) as u8;
            let c1 = ((i * seed + 2) % 4) as u8;
            let c2 = ((i * seed + 3) % 4) as u8;
            let c3 = ((i * seed + 4) % 4) as u8;
            c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
        })
        .collect()
}

impl Gemm for GemmI2 {
    type Elem = u8;

    const OP: &'static str = "gemm_i2";

    const BITS: Bits = Bits::Two;

    fn sample_a(m: usize) -> Vec<u8> {
        pack(m * K, 37)
    }

    fn sample_b(n: usize) -> Vec<u8> {
        pack(K * n, 53)
    }

    fn a_len(m: usize) -> usize {
        m * K / 4
    }

    fn b_len(n: usize) -> usize {
        K * n / 4
    }

    fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
        // 打包位域的字节与码位位置随索引变化，无法整块取 slice；K 固定为 8，循环可被展开。
        let mut acc: i32 = 0;
        for p in 0..K {
            let x = code2(a[i * K / 4 + p / 4], p % 4) as i32;
            let y_pos = p * n + j;
            let y = code2(b[y_pos / 4], y_pos % 4) as i32;
            acc += x * y;
        }
        acc as f32
    }
}

/// 标量代价：整数乘加按 `2·m·k·n` 计；A/B 为打包 2 位码（每 4 元素 1 字节），Y 每元素 4 字节。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    gemm::cost::<GemmI2>(shape)
}

/// 确定性样本：A、B 为打包 2 位码（确定性位模式），Y 为 f32。
pub fn sample(m: usize, n: usize) -> Sample {
    gemm::sample::<GemmI2>(m, n)
}

/// 标量参考：解包后整数点积，转 f32。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    gemm::reference::<GemmI2>(sample)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) -> diagnostic::Result<()> {
    gemm::forward::<GemmI2>(a, b, m, n, out)
}

/// 对称 int2 矩阵乘的语义别名。
#[annotation::model(name = "gemm_i2", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmI2Model = GemmI2;

#[cfg(test)]
#[path = "../tests/gemm_i2_tests.rs"]
mod tests;
