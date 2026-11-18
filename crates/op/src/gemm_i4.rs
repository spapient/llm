//! `gemm_i4`：对称 int4 矩阵乘。骨架在 [`crate::gemm`]，本文件只有"对称 int4"这一份格式
//! 契约：A、B 是打包 nibble（每字节 2 个，低 4 位在前、高 4 位在后），对称零点取配置表
//! [`I4`]；取样、打包长度与整数点积由骨架按 [`I4`] 一行配置读出，点积在 `i32` 上精确累加。

pub use crate::gemm::{AXES, K, project_shape};

use crate::quant::I4;

crate::gemm_variant!(
    GemmI4,
    GemmI4Model,
    "gemm_i4",
    "../tests/gemm_i4_tests.rs",
    u8,
    I4.bits,
    fn config() -> Option<crate::quant::Row> { Some(I4) }
);
