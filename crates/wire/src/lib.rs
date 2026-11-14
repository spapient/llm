//! 与领域无关的字节码闭集：CRC-32、LEB128 变长整数，以及统一字节输出接收器。
//!
//! 这里只放"任何线格式都能复用"的基础件，不认识张量、权重条目或容器布局。上层（`weights`
//! 的 LLPK 容器、`train` 的检查点）只通过这套闭集写字节，长度算术、规范性拒绝与 CRC 覆盖
//! 各只有一处实现。
//!
//! **依赖**：`diagnostic`（错误类型）。**不拥有**：任何精度闭集（`dtype`）、分配（`mem`）、
//! 张量元数据（`tensor`）。

#![deny(unsafe_code)]

pub mod crc;
pub mod leb;
pub mod sink;
