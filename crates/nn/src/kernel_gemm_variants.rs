//! GEMM 精度变体：int2 / int4 / int8 / f32 四档格式契约。
//!
//! 格式无关的轴契约、形状投影与矩阵乘骨架在 [`crate::gemm`]；本文件每档只声明"这一份格式"的
//! 元素类型、位宽、样本取值与点积算术，经 [`crate::gemm_variant!`] 一行装配出完整公开面。

pub mod i2 {
    //! `gemm_i2`：对称 int2 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int2"这一份格式
    //! 契约：A、B 是打包 2 位码（每字节 4 个，低 2 位在前），对称零点取配置表 [`I2`]；取样、
    //! 打包长度与整数点积由骨架按 [`I2`] 一行配置读出，点积在 `i32` 上精确累加后转 f32。
    use crate::gemm as skel;
    pub use crate::gemm::{AXES, K, project_shape};

    use crate::quant::I2;

    crate::gemm_variant!(
        GemmI2,
        GemmI2Model,
        "gemm_i2",
        gemm_packed_tests,
        false,
        u8,
        I2.bits,
        fn config() -> Option<crate::quant::Row> {
            Some(I2)
        },
        fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
            crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::Conf::from(&I2), out)
        },
    );
}

pub mod i4 {
    //! `gemm_i4`：对称 int4 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int4"这一份格式
    //! 契约：A、B 是打包 nibble（每字节 2 个，低 4 位在前、高 4 位在后），对称零点取配置表
    //! [`I4`]；取样、打包长度与整数点积由骨架按 [`I4`] 一行配置读出，点积在 `i32` 上精确累加。
    use crate::gemm as skel;
    pub use crate::gemm::{AXES, K, project_shape};

    use crate::quant::I4;

    crate::gemm_variant!(
        GemmI4,
        GemmI4Model,
        "gemm_i4",
        gemm_packed_tests,
        false,
        u8,
        I4.bits,
        fn config() -> Option<crate::quant::Row> {
            Some(I4)
        },
        fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
            crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::Conf::from(&I4), out)
        },
    );
}

pub mod i8 {
    //! `gemm_i8`：对称 int8 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int8"这一份格式
    //! 契约：`u8` 承载 int8 位模式，点积先在 `i32` 上精确累加（对应 CUDA `dp4a`：每 4 对 int8
    //! 乘积累加进一个 int32），再收敛成 f32。CPU 没有 dp4a，退化为标量整数点积。
    use crate::gemm as skel;
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
}

pub mod f32 {
    //! `gemm_f32`：稠密 f32 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"稠密 f32"这一份格式
    //! 契约：元素类型、位宽、样本取值与点积算术。CPU 片段没有 MMA 原语，算术走通用稠密 tile
    //! （[`crate::matmul::matmul_tiled`]），与标量双重循环逐位一致。
    use crate::gemm as skel;
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
}
