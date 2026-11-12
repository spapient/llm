//! LEB128（小端基 128）变长整数：写端只有一种写法，读端拒绝一切非规范编码。
//!
//! 非规范的两种形态都被拒：用前导 `0x80` 把同一个数拉长（例如 `0x80 0x00` 表示 0），以及
//! 超过 `u64` 容量的序列。`u64` 最多 10 个字节，且第 10 字节只允许最低位（前 9 字节已占 63 位）。

use diagnostic::{Code, Error, Result};

/// 把 `value` 按 LEB128 追加进 `buf`。
pub fn write_u64(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            return;
        }
    }
}

/// 把 `u32` 按 LEB128 追加进 `buf`。
pub fn write_u32(buf: &mut Vec<u8>, value: u32) {
    write_u64(buf, u64::from(value));
}

/// 只读游标：在一段字节上顺序取数，所有越界与非规范都返回错误而不是 panic。
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// 从字节起点构造。
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// 当前位置。
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 读一个原生 `u8`（不是 LEB）。
    pub fn read_u8(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.pos).ok_or_else(|| end("u8"))?;
        self.pos += 1;
        Ok(byte)
    }

    /// 读一个 LEB128 `u64`，并做规范性校验。
    pub fn read_u64(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.read_u8().map_err(|error| error.with("LEB128 整数"))?;
            let group = u64::from(byte & 0x7F);
            if shift >= 64 {
                return Err(bad("LEB128 超过 10 个字节"));
            }
            if shift == 63 {
                // 第 10 字节只有 1 个有效位，且不允许再带 continuation。
                if byte & 0x80 != 0 || group > 1 {
                    return Err(bad("LEB128 超出 u64 容量"));
                }
            }
            value |= group << shift;
            if byte & 0x80 == 0 {
                // 最后一个字节的有效位全 0，却不是唯一的首字节：这是拉长的非规范编码。
                if group == 0 && shift != 0 {
                    return Err(bad("LEB128 含多余的前导零字节"));
                }
                return Ok(value);
            }
            shift += 7;
        }
    }

    /// 读一个 LEB128 `u32`。
    pub fn read_u32(&mut self) -> Result<u32> {
        let value = self.read_u64()?;
        u32::try_from(value).map_err(|_| bad("LEB128 整数超出 u32"))
    }

    /// 取出接下来的 `n` 个字节（借用，不复制）。
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let stop = self.pos.checked_add(n).ok_or_else(|| bad("偏移相加溢出"))?;
        let slice = self
            .bytes
            .get(self.pos..stop)
            .ok_or_else(|| end(&format!("{n} 字节")))?;
        self.pos = stop;
        Ok(slice)
    }
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("pack::leb")
}

fn end(what: &str) -> Error {
    Error::new(Code::Invalid, format!("字节不足，读不出 {what}")).with("pack::leb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_round_trip_across_boundaries() {
        for value in [0u64, 1, 127, 128, 16_383, 16_384, u32::MAX as u64, u64::MAX] {
            let mut buf = Vec::new();
            write_u64(&mut buf, value);
            assert_eq!(Cursor::new(&buf).read_u64().expect("规范"), value);
        }
    }

    #[test]
    fn a_padded_zero_is_rejected() {
        let error = Cursor::new(&[0x80, 0x00]).read_u64().expect_err("非规范");
        assert_eq!(error.code(), Code::Invalid);
    }

    #[test]
    fn an_overflowing_sequence_is_rejected() {
        let mut seq = vec![0x80u8; 9];
        seq.push(0x02); // 第 10 字节用了第 2 位，超出 u64
        assert!(Cursor::new(&seq).read_u64().is_err());
        assert!(Cursor::new(&[0x80u8; 11]).read_u64().is_err());
    }

    #[test]
    fn taking_past_the_end_is_an_error() {
        let mut cursor = Cursor::new(&[1, 2, 3]);
        assert_eq!(cursor.take(2).expect("可取"), &[1, 2]);
        assert_eq!(cursor.pos(), 2);
        assert!(cursor.take(2).is_err());
    }
}
