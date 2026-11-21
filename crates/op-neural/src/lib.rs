//! 神经算子：卷积 / 反卷积（`conv` / `deconv`）、注意力（`attention`）、池化与范数（`classic`）、
//! 扫描骨架（`scan`）。
//!
//! 与 `op-kernel` 的分界不是"精度"而是**形状自由度**：内核算的是「一行乘以一列」，几何是固定的；
//! 这里算的是「窗口滑过张量」——同一份算术要在 padding / stride / dilation / 分组下都成立，
//! 于是每个模块先求出位置映射（[`conv::geom`]），再把映射交给内核的稠密路径
//! （[`llm_op_kernel::fast`]），自己不写第二份乘加循环。
//!
//! 依赖是单向的：本 crate 依赖 `op-kernel` 与 `op-spec`，反之不成立；注册表在 `op`，
//! 于是"神经算子是否与声明一致"由 `op` 的跨层测试核对，而本 crate 不必知道算子名。

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
pub use llm_op_kernel::Scalar;
pub use scan::f32 as scan_f32;

// 契约层在 `op-spec`：本 crate 的骨架宏与语义表按 crate 根解析契约路径，这里搬回来，
// 于是源码里写的是 `crate::model::…` / `crate::operator_here!` 而不是到处写依赖 crate 名。
pub use llm_op_spec::same_bytes;
pub use llm_op_spec::{formula_cell, operator_here, operator_via};
pub use llm_op_spec::{einsum, model, operator, sample, sch, sig, tpl};
pub(crate) use llm_op_spec::tpl::guard;
pub use model::quant;
