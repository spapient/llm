//! 稠密 f32 矩阵乘：`Y[m,n] = Σ_k A[m,k] · B[k,n]`。
//!
//! 骨架在 [`crate::gemm`]，本文件只有"稠密 f32"这一份格式契约：元素类型、位宽、样本取值与
//! 点积算术。CPU 片段没有 MMA 原语，遍历退化成标量双重循环（SIMD/tile 化在 P4 后端补），
//! 收缩轴固定 [`K`]。

use crate::einsum::Bits;
use crate::gemm::{self, Gemm};
use crate::model::{Cost, Sample, ShapeContext};
use diagnostic::Error;

// 收缩轴、三轴契约、形状投影复用 gemm 家族共享模块。
pub use crate::gemm::{AXES, K, project_shape};

/// 稠密 f32 格式的标记类型：承载 `#[model]` 语义。
pub struct Dense;

impl Gemm for Dense {
    type Elem = f32;

    const OP: &'static str = "gemm_f32";

    const BITS: Bits = Bits::ThirtyTwo;

    fn sample_a(m: usize) -> Vec<f32> {
        (0..m * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect()
    }

    fn sample_b(n: usize) -> Vec<f32> {
        (0..K * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect()
    }

    fn a_len(m: usize) -> usize {
        m * K
    }

    fn b_len(n: usize) -> usize {
        K * n
    }

    fn dot(a: &[f32], b: &[f32], n: usize, i: usize, j: usize) -> f32 {
        let row = &a[i * K..][..K];
        row.iter().enumerate().map(|(p, x)| x * b[p * n + j]).sum()
    }
}

/// 标量代价：由骨架按 f32 位宽推导。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    gemm::cost::<Dense>(shape)
}

/// 确定性样本：A（`m×K`）、B（`K×n`）、Y（`m×n`）。
pub fn sample(m: usize, n: usize) -> Sample {
    gemm::sample::<Dense>(m, n)
}

/// 标量参考：三重循环，与内核同一公式。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    gemm::reference::<Dense>(sample)
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 缓冲尺寸不符时返回说明。
pub fn forward(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    gemm::forward::<Dense>(a, b, m, n, out)
}

/// 矩阵乘的具体语义别名。
#[annotation::model(name = "gemm_f32", shape = [16, 16], dtype = "f32", skeleton = "gemm")]
pub type GemmF32 = Dense;

#[cfg(test)]
mod tests;
