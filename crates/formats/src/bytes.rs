//! 职责：`&[u8]` 上的定界小端读取——「偏移 + 宽度」判据的唯一出口。
//!
//! zip、gguf 与 pickle 原本各写一遍越界检查和小端拼装；这里收成两层：
//! 随机访问的 [`slice`]/[`u16_at`]/[`u32_at`]/[`u64_at`]，顺序消费的 [`Cursor`]。

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

/// 顺序游标：每次读取都过 [`slice`]，位置只在读取成功后前进；越界不会退化成「空的成功切片」。
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// 已消费的字节数（调用方用它把「流内偏移」换算成「文件内偏移」）。
    pub(crate) const fn pos(&self) -> usize {
        self.pos
    }

    /// 全部字节数（只用于「谎报计数不得超过流规模」这类粗查）。
    pub(crate) fn total(&self) -> usize {
        self.bytes.len()
    }

    /// 尚未消费的字节数（同上，用于「计数不得超过剩余字节」）。
    pub(crate) fn rest(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// 未消费部分（只读视图）：用于「扫到分隔符为止」这类前视。
    pub(crate) fn tail(&self) -> &'a [u8] {
        self.bytes.get(self.pos..).unwrap_or_default()
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let out = slice(self.bytes, self.pos, len)?;
        self.pos += len;
        Ok(out)
    }

    pub(crate) fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// 定宽 4 字节小端整数（ZIP/GGUF/pickle 的「字」）。
    pub(crate) fn word(&mut self) -> Result<u32> {
        let at = self.pos;
        self.take(4)?;
        u32_at(self.bytes, at)
    }

    /// 定宽 8 字节小端整数（GGUF 的计数与偏移）。
    pub(crate) fn wide(&mut self) -> Result<u64> {
        let at = self.pos;
        self.take(8)?;
        u64_at(self.bytes, at)
    }

    /// 变宽小端整数：宽度由调用方（opcode 或长度前缀）给出。
    pub(crate) fn uint(&mut self, len: usize) -> Result<u64> {
        Ok(self
            .take(len)?
            .iter()
            .rev()
            .fold(0u64, |acc, &byte| (acc << 8) | u64::from(byte)))
    }

    /// 宽度已知的 UTF-8 文本。
    pub(crate) fn text(&mut self, len: usize) -> Result<String> {
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| bad("字符串不是 UTF-8"))
    }
}
