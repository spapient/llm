//! 对称 int2 矩阵乘：`Y[m,n] = Σ_k i2(A[m,k]) · i2(B[k,n])`。
//!
//! 骨架在 [`crate::gemm`]，本文件只有"对称 int2"这一份格式契约：A、B 是打包 2 位码（每字节
//! 四个，低 2 位在前），对称零点 [`ZERO`]。点积在 [`Gemm::dot`] 内按需解包成有符号整数，
//! 再在 `i32` 上精确累加（对应设备侧 `mma` 的整数路径），最后转 f32。CPU 没有
//! mma，退化为标量整数点积；整数累加逐位精确，与设备指令无舍入分歧。

use crate::einsum::Bits;
use crate::gemm::{self, Gemm, packed};
use crate::quant::I2;

// 收缩轴、三轴契约、形状投影复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, K, project_shape};

/// 对称零点：2 位码 0..3 减去 2 得到 -2..=1（取自配置表 [`I2`]）。
pub const ZERO: i32 = I2.zero;

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI2;

impl Gemm for GemmI2 {
    type Elem = u8;

    const OP: &'static str = "gemm_i2";

    const BITS: Bits = I2.bits;

    fn sample_a(m: usize) -> Vec<u8> {
        packed::pack(m * K, 37, I2.cpb, I2.offsets)
    }

    fn sample_b(n: usize) -> Vec<u8> {
        packed::pack(K * n, 53, I2.cpb, I2.offsets)
    }

    fn a_len(m: usize) -> usize {
        m * K / I2.cpb
    }

    fn b_len(n: usize) -> usize {
        K * n / I2.cpb
    }

    fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
        packed::dot(a, b, n, i, j, I2.cpb, ZERO)
    }
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
