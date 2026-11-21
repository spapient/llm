//! `gemm_f32`：稠密 f32 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"稠密 f32"这一份格式
//! 契约：元素类型、位宽、样本取值与点积算术。CPU 片段没有 MMA 原语，算术走通用稠密 tile
//! （[`crate::matmul::matmul_tiled`]），与标量双重循环逐位一致。

pub use crate::gemm::{AXES, K, project_shape};

crate::gemm_variant!(
    Dense,
    GemmF32,
    "gemm_f32",
    gemm_dense_tests,
    true,
    f32,
    crate::einsum::Bits::ThirtyTwo,
    fn sample_a(m: usize) -> Vec<f32> {
        (0..m * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect()
    },
    fn sample_b(n: usize) -> Vec<f32> {
        (0..K * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect()
    },
    fn a_len(m: usize) -> usize {
        m * K
    },
    fn b_len(n: usize) -> usize {
        K * n
    },
    fn dot(a: &[f32], b: &[f32], n: usize, i: usize, j: usize) -> f32 {
        let row = &a[i * K..][..K];
        row.iter().enumerate().map(|(p, x)| x * b[p * n + j]).sum()
    },
    fn mul(a: &[f32], b: &[f32], m: usize, n: usize, out: &mut [f32]) {
        // 稠密 f32 不分块：`K = 8` 时一行 A 的 8 个权重就在寄存器里，分块（64 列块或
        // `matmul_tiled` 的三级块）只会多出累加器驻留与写回，实测反而慢 2~3 倍
        // （`gemm_f32` 基准的三档对照）。顺序累加同时保证与标量参考逐位相等。
        crate::gemm::matmul::<Self>(a, b, m, n, out)
    },
);
