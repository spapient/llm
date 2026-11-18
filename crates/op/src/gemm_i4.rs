//! 对称 int4 矩阵乘：`Y[m,n] = Σ_k i4(A[m,k]) · i4(B[k,n])`。
//!
//! 骨架在 [`crate::gemm`]，本文件只有"对称 int4"这一份格式契约：A、B 是打包 nibble（每字节
//! 2 个，低 4 位在前、高 4 位在后），对称零点取配置表 [`I4`]。取样、打包长度与整数点积由
//! 骨架打包默认实现按 [`I4`] 一行配置读出；点积在 `i32` 上精确累加（对应设备侧 `mma` 的
//! 整数路径），最后转 f32。CPU 没有 mma，退化为标量整数点积；整数累加逐位精确，与设备指令
//! 无舍入分歧。

use crate::einsum::Bits;
use crate::gemm::{self, Gemm};
use crate::quant::I4;

// 收缩轴、三轴契约、形状投影复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, K, project_shape};

/// 标记类型：承载 `#[model]` 语义。
pub struct GemmI4;

impl Gemm for GemmI4 {
    type Elem = u8;

    const OP: &'static str = "gemm_i4";

    const BITS: Bits = I4.bits;

    fn config() -> Option<crate::quant::Row> {
        Some(I4)
    }
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
