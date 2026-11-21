//! 神经算子：卷积 / 反卷积（`conv` / `deconv`）、注意力（`attention`）、池化与范数（`classic`）、
//! 扫描骨架（`scan`）。

#![deny(unsafe_code)]

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
pub use llm_gemm::Scalar;
pub use scan::f32 as scan_f32;

// 契约层在 `spec`：本 crate 的骨架宏与语义表按 crate 根解析契约路径，这里搬回来，
// 于是源码里写的是 `crate::model::…` / `crate::operator_here!` 而不是到处写依赖 crate 名。
pub use llm_spec::same_bytes;
pub(crate) use llm_spec::tpl::guard;
pub use llm_spec::{einsum, model, operator, sample, sch, sig, tpl};
pub use llm_spec::{formula_cell, operator_here, operator_via};
pub use model::quant;

#[cfg(test)]
#[path = "../tests/prop_numeric.rs"]
mod prop_numeric;
