//! 职责：外部权重格式的只读适配层——把别人的文件格式读成「名字 + 类型 + 形状 + 字节区间」，
//! 不参与自有容器、量化方案与证书判定。
//!
//! 文件名即格式名或唯一机制名，限定词下沉为同名目录（`gguf/quant`、`pickle/tensor`）：
//! `json`、`zip`、`pickle`、`torch`、`gguf`，共享机制 `cursor`；测试夹具名同样以
//! 职责命名（`torch_sample` 只负责本机 torch `.bin` 样本）。

#![deny(unsafe_code)]

pub(crate) mod cursor;
pub mod gguf;
pub mod json;
pub(crate) mod pickle;
pub mod torch;
pub(crate) mod zip;

#[cfg(test)]
#[path = "../tests/torch_sample.rs"]
mod torch_sample;
