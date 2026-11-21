//! f32 小算子的四份实现：softmax / rmsnorm / embed / rope。
//!
//! 位宽限定词交给目录（`f32`），文件名只留概念；公开路径仍由 `lib.rs` 的 re-export 保持为
//! `op::softmax_f32`、`op::rmsnorm_f32`、`op::embed_f32`、`op::rope_f32`。

pub mod embed;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;
