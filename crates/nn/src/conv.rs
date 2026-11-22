//! 神经算子：卷积 / 反卷积（`conv` / `deconv`）、注意力（`attention`）、池化与范数（`classic`）、
//! 扫描骨架（`scan`）。

pub mod attention;
pub mod classic;
pub mod conv;
pub mod deconv;
pub mod scan;

pub use attention::decode_attention;
pub use classic::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d,
};
pub use crate::Scalar;
pub use scan::f32 as scan_f32;


#[cfg(test)]
#[path = "../../tests/conv/prop_numeric.rs"]
mod prop_numeric;
