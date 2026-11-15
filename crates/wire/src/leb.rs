//! LEB128（小端基 128）变长整数：写端只有一种写法，读端拒绝一切非规范编码。
//!
//! 非规范的两种形态都被拒：用前导 `0x80` 把同一个数拉长（例如 `0x80 0x00` 表示 0），以及
//! 超过 `u64` 容量的序列。`u64` 最多 10 个字节，且第 10 字节只允许最低位（前 9 字节已占 63 位）。

use diagnostic::{Code, Error, Result};

use crate::sink::Sink;

/// `u64` 的 LEB128 最多字节数（每字节 7 位）。
const MAX_BYTES: usize = 10;

/// 可按 LEB128 编码的无符号整数族：归约到 `u64` 这一步只发生在本 trait 内。
///
/// 只开放五个原生无符号类型，把可接受的类型闭集写进类型系统；不像裸 `Into<u64>` 那样
/// 允许不相关的外部类型意外参与编码，也避免在每个调用点散落 `as u64` 强转。
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
///
/// 各 7 位组先落栈上定长缓冲，**一次** [`Sink::put`] 写出，避免逐字节回调下游。
/// 分组次数由剩余值决定、且每一组都依赖上一次右移，无法跨组并行；这里保留串行循环，
/// 批处理体现在收尾的单次 `put`，而不是逐字节回调。
#[inline]
pub fn write<N: Uint>(out: &mut impl Sink, value: N) {
    let mut value = value.as_u64();
    let mut buffer = [0u8; MAX_BYTES];
    let mut len = 0;
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buffer[len] = byte;
        len += 1;
        if value == 0 {
            break;
        }
    }
    out.put(&buffer[..len]);
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
    ///
    /// 目标类型由绑定推断（`let n: u32 = cur.read()?;`）或 turbofish
    /// （`cur.read::<u64>()`）给出，不再提供按宽度拆开的包装方法。
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
    Error::new(Code::Invalid, message).with("wire::leb")
}

fn end(what: &str) -> Error {
    Error::new(Code::Invalid, format!("字节不足，读不出 {what}")).with("wire::leb")
}

#[cfg(test)]
mod tests;
