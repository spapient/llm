//! `gemm_i2`：对称 int2 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int2"这一份格式
//! 契约：A、B 是打包 2 位码（每字节 4 个，低 2 位在前），对称零点取配置表 [`I2`]；取样、
//! 打包长度与整数点积由骨架按 [`I2`] 一行配置读出，点积在 `i32` 上精确累加后转 f32。

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
        crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::packed::Conf::from(&I2), out)
    },
);
