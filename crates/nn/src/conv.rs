//! 神经算子：卷积 / 反卷积（`geom` / `pool` / `interp` / `deconv`）、注意力（`attention`）、
//! 池化与范数（`classic`）、扫描骨架（`scan`）。

pub mod attention;
pub mod classic;
pub mod deconv;
pub mod scan;

/// 卷积的公共几何：核、步长、补零。通道数不在这里——1d 与 2d 共用同一套核参数。
mod geom;
mod interp;
mod pool;
mod support;

pub use attention::decode_attention;
pub use classic::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d,
};
pub use geom::{Conv1d, Conv2d, ConvGeom};
pub use interp::{Interpolate, Sampler};
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use scan::f32 as scan_f32;

#[cfg(test)]
#[path = "../tests/conv/conv_tests.rs"]
mod conv_tests;

#[cfg(test)]
#[path = "../tests/conv/prop_numeric.rs"]
mod prop_numeric;
