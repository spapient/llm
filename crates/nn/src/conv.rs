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
/// 卷积族的几何与窗口算子实现。
///
/// 原先这些类型在 crate 根，内部写 `crate::conv::Conv2d` 指的是实现模块本身；并入 `nn` 后
/// `crate::conv` 变成外层模块，故在这里把实现模块的公开面平铺一层，旧路径照旧可解析。
pub use conv::{
    AdaptiveAvgPool2d, AvgPool2d, Conv1d, Conv2d, ConvGeom, Interpolate, MaxPool1d, MaxPool2d,
    Sampler,
};
pub use crate::Scalar;
pub use scan::f32 as scan_f32;


#[cfg(test)]
#[path = "../../tests/conv/prop_numeric.rs"]
mod prop_numeric;
