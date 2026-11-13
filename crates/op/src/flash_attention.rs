//! Flash 注意力的语义别名：形状、样本、代价与朴素 [`crate::attention`] 完全共用，
//! 只把标量参考换成 online softmax 版本（`attention::flash_reference`）。
//!
//! 单独成模块是因为 `#[annotation::model]` 在每个模块里固定生成一个 `MODEL`；
//! 两个算子各自一个 `MODEL`，注册表分别指向它们。

/// 标记类型：承载 flash 注意力的 `#[model]` 语义。
pub struct FlashAttention;

/// Flash 注意力语义别名；codegen 按注册名选 `attention::flash_reference`。
#[annotation::model(name = "flash_attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention")]
pub type FlashAttentionF32 = FlashAttention;
