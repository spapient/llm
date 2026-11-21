//! `gemm_i8`：对称 int8 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int8"这一份格式
//! 契约：`u8` 承载 int8 位模式，点积先在 `i32` 上精确累加（对应 CUDA `dp4a`：每 4 对 int8
//! 乘积累加进一个 int32），再收敛成 f32。CPU 没有 dp4a，退化为标量整数点积。

pub use crate::gemm::{AXES, K, project_shape};

use crate::quant::I8;

crate::gemm_variant!(
    GemmI8,
    GemmI8Model,
    "gemm_i8",
    gemm_packed_tests,
    false,
    u8,
    I8.bits,
    fn sample_a(m: usize) -> Vec<u8> {
        (0..m * K)
            .map(|i| (((i * 37) % 16) as i8 - 8) as u8)
            .collect()
    },
    fn sample_b(n: usize) -> Vec<u8> {
        (0..K * n)
            .map(|i| (((i * 53) % 16) as i8 - 8) as u8)
            .collect()
    },
    fn a_len(m: usize) -> usize {
        m * K
    },
    fn b_len(n: usize) -> usize {
        K * n
    },
    fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
        let row = &a[i * K..][..K];
        let acc: i32 = row
            .iter()
            .enumerate()
            .map(|(p, packed)| (*packed as i8) as i32 * (b[p * n + j] as i8) as i32)
            .sum();
        acc as f32
    },
    fn codes(values: &[u8]) -> Vec<i32> {
        crate::gemm::signed_codes(values)
    },
    fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
        let codes_a = crate::gemm::signed_codes(a);
        let codes_b = crate::gemm::signed_codes(b);
        crate::gemm::accumulate_tiled(&codes_a, &codes_b, m, n, out)
    },
);
