//! 神经算子：卷积 / 反卷积（`geom` / `pool` / `interp` / `deconv`）、注意力（`attention`）、
//! 池化与范数（`classic`）、扫描骨架（`scan`）。

#[path = "conv_attention.rs"]
pub mod attention;
#[path = "conv_classic.rs"]
pub mod classic;
#[path = "conv_deconv.rs"]
pub mod deconv;
#[path = "conv_geom.rs"]
mod geom;
#[path = "conv_interp.rs"]
mod interp;
#[path = "conv_pool.rs"]
mod pool;
#[path = "conv_scan.rs"]
pub mod scan;
#[path = "conv_support.rs"]
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
#[path = "../tests/conv_tests.rs"]
mod conv_tests;

#[cfg(test)]
#[path = "../tests/conv_prop_tests.rs"]
mod conv_prop_tests;
