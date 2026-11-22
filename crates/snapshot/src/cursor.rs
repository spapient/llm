//! 快照侧的读取游标与错误词汇：在 `wire::Cursor` 之上保留字段上下文与截断/非规范判据。

use std::ops::Range;

use diagnostic::{Code, Error, Result};
use wire::leb::Cursor as WireCursor;

/// 解码失败的上下文：错误只由这一层产生，排查时不必再猜是哪一段。
pub(crate) const CTX: &str = "snapshot::Snapshot::decode";

/// 取一段字节；不足时按起始偏移报截断。
pub(crate) fn need(bytes: &[u8], span: Range<usize>, offset: usize) -> Result<&[u8]> {
    match bytes.get(span) {
        Some(slice) => Ok(slice),
        None => Err(truncated(offset)),
    }
}

/// 检查点在给定偏移处被截断。
pub(crate) fn truncated(offset: usize) -> Error {
    Error::new(Code::Invalid, format!("检查点在偏移 {offset} 处截断")).with(CTX)
}

/// 一条解码期非法输入；上下文固定在快照层。
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(CTX)
}

/// 检查点与手头的权重表不一致：只由 [`super::Snapshot::check_weights`] 产生。
pub(crate) fn mismatch(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("snapshot::Snapshot::check_weights")
}

/// 第 `idx` 个条目的错误帧：一个几千条目的检查点里，"第 3 个条目"比"某个偏移"有用得多。
pub(crate) fn frame(idx: usize) -> impl Fn(Error) -> Error {
    move |error| error.with(format!("第 {index} 个权重条目"))
}

/// 第 `idx` 个条目的非法输入：`frame` 与 `invalid` 的现成组合，省掉调用处的双层括号。
pub(crate) fn invalid_at(idx: usize, message: impl Into<String>) -> Error {
    frame(idx)(invalid(message))
}

/// 逐字节读取器：在 `wire::Cursor` 之上保留快照侧的字段上下文与错误词汇。
pub(crate) struct Cursor<'a> {
    inner: WireCursor<'a>,
    /// 当前消费偏移；测试用它给名字块切片区间定位。
    pub(crate) offset: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], offset: usize) -> Result<Self> {
        let mut inner = WireCursor::new(bytes);
        inner.take(offset).map_err(|_| truncated(offset))?;
        Ok(Self { inner, offset })
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let slice = self
            .inner
            .take(len)
            .map_err(|_| truncated(self.inner.pos()))?;
        self.offset = self.inner.pos();
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        let byte = self
            .inner
            .read_u8()
            .map_err(|_| truncated(self.inner.pos()))?;
        self.offset = self.inner.pos();
        Ok(byte)
    }

    /// 读一个 LEB128 变长整数；非规范写法（如 `0x87 0x00`）或超出 `u64` 宽度都拒绝。
    pub(crate) fn varint(&mut self) -> Result<u64> {
        let value = self.inner.read::<u64>().map_err(|error| {
            if error.message().starts_with("字节不足") {
                truncated(self.inner.pos())
            } else {
                invalid("检查点里的整数非规范写法（含前导零字节）或超出 u64 上限")
            }
        })?;
        self.offset = self.inner.pos();
        Ok(value)
    }

    /// 读一个不能超过 `usize` 的计数；`what` 是字段名，用来把错误指到具体位置。
    pub(crate) fn count(&mut self, what: &str) -> Result<usize> {
        let value = self.varint()?;
        usize::try_from(value)
            .map_err(|_| invalid(format!("检查点的{what} {value} 超出 usize 上限")))
    }

    pub(crate) fn remaining(&self) -> usize {
        self.inner.remaining().len()
    }
}
