//! decode 注意力的两个面：内核算术（[`kernel`]）与语义表（[`semantics`]）。
//!
//! 内核把 query 行数与 K/V 历史行数解耦之后的校验、逐行打分、online softmax 与按头/按批变体
//! 集中在一处；语义侧只补形状上下文、确定性样本与代价。公开路径仍由 [`crate::attention`]
//! 与 `lib.rs` 的 re-export 保持为 `attention::decode_forward*` 与 `op::decode_attention`。

mod kernel;
pub mod semantics;

pub use kernel::{
    decode_flash_forward, decode_flash_forward_batch, decode_flash_forward_batch_var,
    decode_flash_forward_heads, decode_forward, decode_forward_batch, decode_forward_batch_var,
    decode_forward_heads,
};

pub(crate) use kernel::check_decode_len;
pub(super) use kernel::decode;
