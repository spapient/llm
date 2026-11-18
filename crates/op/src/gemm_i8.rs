//! 对称 int8 矩阵乘：`Y[m,n] = Σ_k i8(A[m,k]) · i8(B[k,n])`。
//!
//! 骨架在 [`crate::gemm`]，本文件只有"对称 int8"这一份格式契约：`u8` 承载 int8 位模式，
//! 点积先在 `i32` 上精确累加（对应 CUDA 的 `dp4a`：每 4 对 int8 乘积累加进一个 int32），再
//! 收敛成 f32。CPU 没有 dp4a，退化成标量整数点积；整数累加逐位精确，与设备指令无舍入分歧。

use crate::einsum::Bits;
use crate::gemm::{self, Gemm};
use crate::quant::I8;

// 收缩轴、三轴契约、形状投影复用 gemm 家族共享模块（dp4a 一次 4 个，K=8 即两次）。
pub use crate::gemm::{AXES, K, project_shape};

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI8;

impl Gemm for GemmI8 {
    type Elem = u8;

    const OP: &'static str = "gemm_i8";

    const BITS: Bits = I8.bits;

    fn sample_a(m: usize) -> Vec<u8> {
        (0..m * K)
            .map(|i| (((i * 37) % 16) as i8 - 8) as u8)
            .collect()
    }

    fn sample_b(n: usize) -> Vec<u8> {
        (0..K * n)
            .map(|i| (((i * 53) % 16) as i8 - 8) as u8)
            .collect()
    }

    fn a_len(m: usize) -> usize {
        m * K
    }

    fn b_len(n: usize) -> usize {
        K * n
    }

    fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
        let row = &a[i * K..][..K];
        let acc: i32 = row
            .iter()
            .enumerate()
            .map(|(p, packed)| (*packed as i8) as i32 * (b[p * n + j] as i8) as i32)
            .sum();
        acc as f32
    }

    fn codes(values: &[u8]) -> Vec<i32> {
        values.iter().map(|&byte| byte as i8 as i32).collect()
    }
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) -> diagnostic::Result<()> {
    gemm::forward::<GemmI8>(a, b, m, n, out)
}

/// 对称 int8 矩阵乘的语义别名。
#[annotation::model(name = "gemm_i8", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmI8Model = GemmI8;

#[cfg(test)]
#[path = "../tests/gemm_i8_tests.rs"]
mod tests;
