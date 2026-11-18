//! 对称 int4 矩阵乘：`Y[m,n] = Σ_k i4(A[m,k]) · i4(B[k,n])`。
//!
//! 骨架在 [`crate::gemm`]，本文件只有"对称 int4"这一份格式契约：A、B 是打包 nibble（每字节
//! 2 个，低 4 位在前、高 4 位在后），对称零点 [`ZERO`]。点积在 [`Gemm::dot`] 内按需解包
//! 成有符号整数，再在 `i32` 上精确累加（对应设备侧 `mma` 的整数路径），最后转 f32。CPU 没有
//! mma，退化为标量整数点积；整数累加逐位精确，与设备指令无舍入分歧。

use crate::einsum::Bits;
use crate::gemm::{self, Gemm, packed};
use crate::model::{Cost, Sample, ShapeContext};
use diagnostic::Error;

// 收缩轴、三轴契约、形状投影复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, K, project_shape};

/// 对称零点：nibble 0..15 减去 8 得到 -8..7。
pub const ZERO: i32 = 8;

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI4;

impl Gemm for GemmI4 {
    type Elem = u8;

    const OP: &'static str = "gemm_i4";

    const BITS: Bits = Bits::Four;

    fn sample_a(m: usize) -> Vec<u8> {
        packed::pack(m * K, 37, 2, &[1, 9])
    }

    fn sample_b(n: usize) -> Vec<u8> {
        packed::pack(K * n, 53, 2, &[1, 9])
    }

    fn a_len(m: usize) -> usize {
        m * K / 2
    }

    fn b_len(n: usize) -> usize {
        K * n / 2
    }

    fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
        packed::dot(a, b, n, i, j, 2, ZERO)
    }
}

/// 标量代价：整数乘加按 `2·m·k·n` 计；A/B 为打包 nibble（每 2 元素 1 字节），Y 每元素 4 字节。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    gemm::cost::<GemmI4>(shape)
}

/// 确定性样本：A、B 为打包 nibble（确定性位模式），Y 为 f32。
pub fn sample(m: usize, n: usize) -> Sample {
    gemm::sample::<GemmI4>(m, n)
}

/// 标量参考：解包后整数点积，转 f32。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    gemm::reference::<GemmI4>(sample)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) -> diagnostic::Result<()> {
    gemm::forward::<GemmI4>(a, b, m, n, out)
}

/// 对称 int4 矩阵乘的语义别名。
#[annotation::model(name = "gemm_i4", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmI4Model = GemmI4;

#[cfg(test)]
#[path = "../tests/gemm_i4_tests.rs"]
mod tests;
