//! `gemm_f32`：稠密 f32 矩阵乘。骨架在 [`crate::gemm`]，本文件只有"稠密 f32"这一份格式
//! 契约：元素类型、位宽、样本取值与点积算术。CPU 片段没有 MMA 原语，遍历退化成标量双重循环。

pub use crate::gemm::{AXES, K, project_shape};

crate::gemm_variant!(
    Dense,
    GemmF32,
    "gemm_f32",
    "../tests/gemm_f32_tests.rs",
    f32,
    crate::einsum::Bits::ThirtyTwo,
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
);
