//! f32 小算子四合一：softmax / rmsnorm / embed / rope 的语义、样本与参考。
//!
//! 实现在 [`f32`] 下按概念分文件；公开路径由 lib.rs 的 re-export 保持为 `op::softmax_f32`、
//! `op::rmsnorm_f32`、`op::embed_f32`、`op::rope_f32`。
pub mod f32;
