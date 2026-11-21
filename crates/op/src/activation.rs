//! 单行激活的四个小算子：softmax / rmsnorm / embed / rope 的语义、样本与参考。
//!
//! 四者都是对单行激活的逐元素/归约变换，形状与校验逻辑同构，所以放在一个文件里而不是
//! 四个目录：每个模块只管自己的算术，公开路径仍由 `lib.rs` 的 re-export 保持为
//! `op::softmax_f32`、`op::rmsnorm_f32`、`op::embed_f32`、`op::rope_f32`。

pub mod embed;

pub mod rmsnorm;

pub mod rope;

pub mod softmax;
