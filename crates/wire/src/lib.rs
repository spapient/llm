//! 与领域无关的字节码闭集：CRC-32、LEB128 变长整数、统一字节输出接收器，以及三条被各线格式
//! 共用的基础件——对齐落位、`Invalid` 错误的唯一构造点、`Option` → `Result` 的短路。
//!
//! 这里只放"任何线格式都能复用"的基础件，不认识张量、权重条目或容器布局。上层（`weights`
//! 的 `.u` 容器、`train` 的检查点、`formats` 的外部格式适配器）只通过这套闭集写字节，长度
//! 算术、规范性拒绝、对齐判据与 CRC 覆盖各只有一处实现。
//!
//! **依赖**：`diagnostic`（错误类型）。**不拥有**：任何精度闭集（`dtype`）、分配（`storage`）、
//! 张量元数据（`tensor`）。

#![deny(unsafe_code)]

pub mod crc;
pub mod leb;
pub mod sink;

pub mod fault;

pub use fault::{bad, or_bad};

/// 向上取整到 `align` 的倍数；`align` 非 2 的幂或结果溢出时返回 `None`。
///
/// 容器帧与 GGUF 数据区都要按对齐落位，判据（只允许 2 的幂）与溢出路径各只有这一处。
#[must_use]
pub fn align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    value.div_ceil(align).checked_mul(align)
}
