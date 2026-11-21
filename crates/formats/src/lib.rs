//! 外部权重格式的只读适配层：把别人的文件格式读成"名字 + 类型 + 形状 + 字节区间"，不参与
//! 自有容器、量化方案与证书判定。

#![deny(unsafe_code)]

pub mod bin;
pub mod gguf;
pub mod json;
pub(crate) mod gguf_dequant;
pub(crate) mod gguf_reader;
pub(crate) mod json_number;
pub(crate) mod json_reader;
pub(crate) mod pickle;
pub(crate) mod pickle_stack;
pub(crate) mod pickle_storage;
pub(crate) mod zip;

#[cfg(test)]
#[path = "../tests/testutil.rs"]
mod testutil;
