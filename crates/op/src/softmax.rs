//! f32 小算子四合一：softmax / rmsnorm / embed / rope 的语义、样本与参考。
//!
//! 四个模块各自保留原有的 `#[annotation::model]` 语义常量与测试；公开路径由 lib.rs 的
//! re-export 保持为 `op::softmax_f32`、`op::rmsnorm_f32`、`op::embed_f32`、`op::rope_f32`。
pub mod softmax_f32;

pub mod rmsnorm_f32;

pub mod embed_f32;

pub mod rope_f32;
