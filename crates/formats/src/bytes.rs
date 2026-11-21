//! 职责：`&[u8]` 上的定界小端读取——「偏移 + 宽度」判据的唯一出口。
//!
//! zip、gguf 与 pickle 原本各写一遍越界检查和小端拼装；这里收成一处，
//! 越界一律报错，绝不退化成「空的成功切片」。

use diagnostic::Result;

use wire::bad;

/// `bytes[at..at + len]`：偏移溢出与越界都显式失败。
pub(crate) fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    let stop = at.checked_add(len).ok_or_else(|| bad("字段偏移溢出"))?;
    bytes.get(at..stop).ok_or_else(|| bad("字段越界"))
}

/// 小端定宽整数：宽度由类型给出，越界由 [`slice`] 统一拒。
pub(crate) fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    let raw = slice(bytes, at, 2)?;
    Ok(u16::from_le_bytes([raw[0], raw[1]]))
}

pub(crate) fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    let raw = slice(bytes, at, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

pub(crate) fn u64_at(bytes: &[u8], at: usize) -> Result<u64> {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(slice(bytes, at, 8)?);
    Ok(u64::from_le_bytes(raw))
}
