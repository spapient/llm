//! 设备侧分块 lowering：共享内存 GEMM、张量核 GEMM 与分块注意力。
//!
//! 几何助手（`geometry`）、张量核 GEMM（`tensor`）、标量 / simdgroup / 寄存器 / int8 分块 GEMM
//! （`scalar`）、注意力两路（`attention`）与 half 转置前置（`transpose`）按职责拆到子模块，
//! 这里只保留模块声明与入口符号重导出。

pub(crate) mod attention;
pub(crate) mod geometry;
pub(crate) mod scalar;
pub(crate) mod tensor;
pub(crate) mod transpose;

pub use attention::{metal_flash_attention, metal_tiled_attention};
pub use scalar::{
    metal_int8_gemm, metal_int8_gemm_program, metal_simdgroup_gemm, metal_tiled_gemm,
};
pub use tensor::{metal_tensor_gemm, tensor_geometry};
pub use transpose::metal_transpose_half;

#[cfg(test)]
#[path = "../tests/tiled.rs"]
mod tests;
