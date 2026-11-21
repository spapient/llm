//! 职责：外部权重格式的只读适配层——把别人的文件格式读成「名字 + 类型 + 形状 + 字节区间」，
//! 不参与自有容器、量化方案与证书判定。
//!
//! 所有文件平铺在 `src/` 一层，文件名即格式名或唯一机制名：`json`（头部与账本文本）、
//! `zip`（归档）、`pickle`（栈式对象流）、`pickle_tensor`（pickle 值 → torch 张量）、
//! `torch`（torch.save 检查点）、`gguf`（GGUF 文件）、`gguf_quant`（GGUF 分块反量化）、
//! `cursor`（定界小端读取，上面五者共用）。

#![deny(unsafe_code)]

pub(crate) mod cursor;
pub mod gguf;
pub(crate) mod gguf_quant;
pub mod json;
pub(crate) mod pickle;
pub(crate) mod pickle_tensor;
pub mod torch;
pub(crate) mod zip;

#[cfg(test)]
#[path = "../tests/sample.rs"]
mod sample;
