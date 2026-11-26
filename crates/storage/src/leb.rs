//! LEB128（小端基 128）变长整数：写端只有一种写法，读端拒绝一切非规范编码。

use diagnostic::{Code, Error, Result};

use crate::sink::Sink;

/// `u64` 的 LEB128 最多字节数（每字节 7 位）。
const MAX_BYTES: usize = 10;

/// 可按 LEB128 编码的无符号整数族：归约到 `u64` 这一步只发生在本 trait 内。
pub trait Uint: Copy {
    /// 归约到 LEB128 的内部宽度 `u64`。
    fn as_u64(self) -> u64;
}

macro_rules! impl_uint {
    ($($t:ty),*) => {
        $(
            impl Uint for $t {
                #[inline]
                fn as_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    };
}

impl_uint!(u8, u16, u32, u64, usize);

/// 把 `value` 按 LEB128 追加进 `out`。
#[inline]
pub fn write<N: Uint>(out: &mut impl Sink, value: N) {
    let mut value = value.as_u64();
    let mut buf = [0u8; MAX_BYTES];
    let mut len = 0;
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf[len] = byte;
        len += 1;
        if value == 0 {
            break;
        }
    }
    out.put(&buf[..len]);
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
    #[inline]
    pub fn read_u8(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.pos).ok_or_else(|| end("u8"))?;
        self.pos += 1;
        Ok(byte)
    }

    /// 规范 LEB128 累积到 `u64`：宽度校验引擎。每组依赖上一组的移位，串行不可并行。
    fn raw(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.read_u8().map_err(|error| error.with("LEB128 整数"))?;
            let group = checked_group(byte, shift)?;
            value |= group << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
    }

    /// 读一个规范 LEB128 整数，归约到目标无符号类型；超出目标宽度即报错。
    #[inline]
    pub fn read<N: TryFrom<u64>>(&mut self) -> Result<N> {
        let value = self.raw()?;
        N::try_from(value).map_err(|_| bad("LEB128 整数超出目标宽度"))
    }

    /// 从当前位置到末尾的全部剩余字节（不复制、不移动位置）。
    #[must_use]
    pub fn remaining(&self) -> &'a [u8] {
        self.bytes.get(self.pos..).unwrap_or(&[])
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

    /// 全部字节数（粗查「谎报计数不得超过流规模」）。
    pub fn total(&self) -> usize {
        self.bytes.len()
    }

    /// 定宽 `N` 字节小端原始数组：先按 [`Self::take`] 定界，再一次性收窄，不经 `u64` 往返。
    fn le<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut raw = [0u8; N];
        raw.copy_from_slice(self.take(N)?);
        Ok(raw)
    }

    /// 定宽 4 字节小端整数（ZIP/GGUF/pickle 的「字」）。
    pub fn word(&mut self) -> Result<u32> {
        self.le().map(u32::from_le_bytes)
    }

    /// 定宽 8 字节小端整数（GGUF 的计数与偏移）。
    pub fn wide(&mut self) -> Result<u64> {
        self.le().map(u64::from_le_bytes)
    }

    /// 变宽小端整数：宽度由调用方（opcode 或长度前缀）给出。
    pub fn uint(&mut self, len: usize) -> Result<u64> {
        Ok(self
            .take(len)?
            .iter()
            .rev()
            .fold(0u64, |acc, &byte| (acc << 8) | u64::from(byte)))
    }

    /// 宽度已知的 UTF-8 文本。
    pub fn text(&mut self, len: usize) -> Result<String> {
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| bad("字符串不是 UTF-8"))
    }
}

/// `bytes[at..at + len]`：偏移溢出与越界都显式失败（随机访问的「偏移 + 宽度」判据）。
pub fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    let stop = at.checked_add(len).ok_or_else(|| bad("字段偏移溢出"))?;
    bytes.get(at..stop).ok_or_else(|| bad("字段越界"))
}

/// 小端定宽字节：宽度由 const 泛型给出，越界由 [`slice`] 统一拒。
fn le_at<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N]> {
    let mut raw = [0u8; N];
    raw.copy_from_slice(slice(bytes, at, N)?);
    Ok(raw)
}

/// 小端定宽整数：三个入口只是 [`le_at`] 的一次 `from_le_bytes` 收窄。
pub fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    le_at(bytes, at).map(u16::from_le_bytes)
}

/// 偏移 `at` 处的 4 字节小端整数。
pub fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    le_at(bytes, at).map(u32::from_le_bytes)
}

/// 偏移 `at` 处的 8 字节小端整数。
pub fn u64_at(bytes: &[u8], at: usize) -> Result<u64> {
    le_at(bytes, at).map(u64::from_le_bytes)
}

/// 一个 7 位组的规范性：第 10 个字节（`shift == 63`）只允许 `0x00`/`0x01`，
/// 终止字节的有效位全 0 却又不是首字节时属于拉长的非规范编码。
/// 只判这一组能不能收，不判整串够不够长——长度由 [`Cursor::read_u8`] 报。
#[inline]
fn checked_group(byte: u8, shift: u32) -> Result<u64> {
    if shift == 63 && byte > 1 {
        return Err(bad("LEB128 超出 u64 容量"));
    }
    if byte == 0 && shift != 0 {
        return Err(bad("LEB128 含多余的前导零字节"));
    }
    Ok(u64::from(byte & 0x7F))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("storage::leb")
}

fn end(what: &str) -> Error {
    Error::new(Code::Invalid, format!("字节不足，读不出 {what}")).with("storage::leb")
}

#[cfg(test)]
mod tests {
    use super::*;

    testkit::test_cases! {
    fn integers_round_trip_across_boundaries() {
        for value in [
            0u64,
            1,
            127,
            128,
            16_383,
            16_384,
            u64::from(u32::MAX),
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            write(&mut buf, value);
            assert_eq!(Cursor::new(&buf).read::<u64>().expect("规范"), value);
        }
    }

    fn every_unsigned_width_encodes() {
        let mut buf = Vec::new();
        write(&mut buf, 7u8);
        write(&mut buf, 300u16);
        write(&mut buf, 40_000u32);
        write(&mut buf, 9_000_000_000u64);
        write(&mut buf, 5usize);
        let mut cur = Cursor::new(&buf);
        assert_eq!(cur.read::<u64>().expect("u8"), 7);
        assert_eq!(cur.read::<u64>().expect("u16"), 300);
        assert_eq!(cur.read::<u32>().expect("u32"), 40_000);
        assert_eq!(cur.read::<u64>().expect("u64"), 9_000_000_000);
        assert_eq!(cur.read::<u64>().expect("usize"), 5);
    }

    fn a_padded_zero_is_rejected() {
        let error = Cursor::new(&[0x80, 0x00])
            .read::<u64>()
            .expect_err("非规范");
        assert_eq!(error.code(), Code::Invalid);
    }

    fn an_overflowing_sequence_is_rejected() {
        let mut seq = vec![0x80u8; 9];
        seq.push(0x02); // 第 10 字节用了第 2 位，超出 u64
        assert!(Cursor::new(&seq).read::<u64>().is_err());
        assert!(Cursor::new(&[0x80u8; 11]).read::<u64>().is_err());
    }

    fn taking_past_the_end_is_an_error() {
        let mut cursor = Cursor::new(&[1, 2, 3]);
        assert_eq!(cursor.take(2).expect("可取"), &[1, 2]);
        assert_eq!(cursor.pos(), 2);
        assert!(cursor.take(2).is_err());
    }

    /// 属性骨架：随机 `u64` 编码后必须规范解码回原值。
    ///
    /// 用例串行生成——下一个输入依赖上一次 PRNG 状态，无法预生成批量；不变量是往返一致与
    /// 字节数上界，边界值（0、127、128、u32::MAX、u64::MAX）由 `integers_round_trip_across_boundaries`
    /// 单独锁定，这里只覆盖中间分布。
    fn random_u64_values_round_trip_through_canonical_encoding() {
        let mut rng = testkit::Rng::new(0x1E_B12E);
        for _ in 0..64 {
            let value = (u64::from(rng.below(u32::MAX)) << 32) | u64::from(rng.below(u32::MAX));
            let mut buf = Vec::new();
            write(&mut buf, value);
            assert!(buf.len() <= 10, "u64 的 LEB128 最多 10 字节");
            let decoded: u64 = Cursor::new(&buf).read().expect("随机值必为规范编码");
            assert_eq!(decoded, value);
        }
    }

    fn fixed_width_le_reads_and_random_access() {
        // 顺序：word(4) + wide(8) + uint(2) + text(3)，小端拼装与位置纪律。
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x0102_0304u32.to_le_bytes());
        bytes.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        bytes.extend_from_slice(&0xABCDu16.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        let mut cur = Cursor::new(&bytes);
        assert_eq!(cur.total(), bytes.len());
        assert_eq!(cur.word().unwrap(), 0x0102_0304);
        assert_eq!(cur.wide().unwrap(), 0x0102_0304_0506_0708);
        assert_eq!(cur.uint(2).unwrap(), 0xABCD);
        assert_eq!(cur.text(3).unwrap(), "abc");
        assert!(cur.remaining().is_empty());
        assert!(Cursor::new(&bytes).text(99).is_err(), "越界文本必须失败");

        // 随机访问：偏移 + 宽度判据，溢出与越界都显式失败。
        let raw = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(u16_at(&raw, 0).unwrap(), 0x2211);
        assert_eq!(u32_at(&raw, 0).unwrap(), 0x4433_2211);
        assert_eq!(u64_at(&raw, 0).unwrap(), 0x8877_6655_4433_2211);
        assert_eq!(slice(&raw, 2, 3).unwrap(), &[0x33, 0x44, 0x55]);
        assert!(slice(&raw, usize::MAX, 4).is_err(), "偏移溢出必须失败");
        assert!(u32_at(&raw, 6).is_err(), "越界必须失败");
    }
    }
}
