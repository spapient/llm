//! 经典卷积与池化：NHWC 布局的 `Conv1d`/`Conv2d`、`MaxPool2d`/`AvgPool2d`。
//!
//! 卷积不写双重循环：先 `im2col` 把每个输出位置展开成一行，再交给 [`crate::fast`] 的
//! GEMM 微核。语义与逐点直算逐位对齐 f32 容差，性能走 SIMD 通道（i8 量化路径见
//! [`Conv2d::forward_i8`]，复用 `fast::gemm_i8` 的多槽池）。
//!
//! **拥有**：本模块的四个层结构。**不拥有**：GEMM 微核（`fast`）、张量元数据（`tensor`）。

/// 卷积的公共几何：核、步长、补零。通道数不在这里——1d 与 2d 共用同一套核参数。
mod geom;

mod interp;

mod pool;

mod support;
pub use geom::{Conv1d, Conv2d, ConvGeom};
pub use interp::{Interpolate, Sampler};
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d};

#[cfg(test)]
#[path = "../tests/conv_tests.rs"]
mod tests;
