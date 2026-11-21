//! 与领域无关的字节码闭集：CRC-32、LEB128 变长整数、统一字节输出接收器，以及三条被各线格式
//! 共用的基础件——对齐落位、`Invalid` 错误的唯一构造点、`Option` → `Result` 的短路。

#![deny(unsafe_code)]

pub mod crc;
pub mod fault;
pub mod leb;
pub mod sink;

pub use fault::{bad, or_bad};

/// 向上取整到 `align` 的倍数；`align` 非 2 的幂或结果溢出时返回 `None`。
#[must_use]
pub fn align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    value.div_ceil(align).checked_mul(align)
}
