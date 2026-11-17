//! Flash decode 注意力的语义别名：形状、样本、代价与朴素 [`crate::decode_attention`] 共用，
//! 只把标量参考换成 online softmax 版本（`decode_attention::decode_flash_reference`）。
//!
//! 单独成模块是因为 `#[annotation::model]` 在每个模块里固定生成一个 `MODEL`；
//! 两个算子各自一个 `MODEL`，注册表分别指向它们。

pub use crate::decode_attention::{AXES, cost, project_shape, sample};

/// 标记类型：承载 flash decode 注意力的 `#[model]` 语义。
pub struct DecodeFlashAttention;

/// Flash decode 注意力语义别名；codegen 按注册名选 `decode_attention::decode_flash_reference`。
#[annotation::model(name = "decode_flash_attention_f32", shape = [1, 8], dtype = "f32", skeleton = "attention")]
pub type DecodeFlashAttentionF32 = DecodeFlashAttention;
