//! 外部权重格式的只读适配层：把别人的文件格式读成"名字 + 类型 + 形状 + 字节区间"，不参与
//! 自有容器、量化方案与证书判定。
//!
//! **拥有**：torch `.bin` 的 ZIP_STORED 解归档与 pickle 栈式虚拟机（[`bin`]、`pickle`、`zip`）、
//! GGUF 的种类表与数据区解析（[`gguf`]）、以及只读 JSON（[`json`]）——它是 safetensors 头部与
//! 沉降账本共用的词法层。
//!
//! **不拥有**：条目唯一性与区间不重叠（`weights::Header`）、精度闭集（`dtype`）、张量元数据
//! （`tensor`）、任何写路径。解出的四要素交回上层，由上层用同一套判据收口。
//!
//! **依赖**：`diagnostic`、`dtype`、`wire`（对齐落位与 `Invalid` 构造点）。**被依赖**：
//! `weights`（重导出本层，下游 crate 的路径不变）。
//!
//! 三个适配器都只读、零依赖、纯安全 Rust：未知 opcode、不可识别的可调用对象、越界区间一律
//! 显式 [`diagnostic::Code::Unsupported`] 或 [`diagnostic::Code::Invalid`]，不执行任意代码、
//! 不猜结果。

#![deny(unsafe_code)]

pub mod bin;
pub mod gguf;
pub mod json;
pub(crate) mod pickle;
pub(crate) mod zip;

#[cfg(test)]
#[path = "../tests/testutil.rs"]
mod testutil;
