//! 字节输出闭集：同一份编码逻辑可写到计数、`Vec`、固定缓冲或流式文件。

use std::io;

use crate::crc;

/// 统一字节输出。
pub trait Sink {
    /// 追加一段字节。
    fn put(&mut self, bytes: &[u8]);
}

/// 只做加法：算长度用，与写出共用同一编码路径。
#[derive(Default)]
pub struct Count(pub usize);

impl Sink for Count {
    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        self.0 += bytes.len();
    }
}

/// `Vec<u8>` 本身就是一个 Sink，便于写函数统一泛型化。
impl Sink for Vec<u8> {
    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

/// 写进调用方固定缓冲，放不下就置 `overflow`，不越界、不截断。
pub struct SliceSink<'a> {
    /// 目标缓冲。
    out: &'a mut [u8],
    /// 已写位置。
    at: usize,
    /// 是否发生过放不下。
    overflow: bool,
}

impl<'a> SliceSink<'a> {
    /// 绑定一块缓冲。
    pub fn new(out: &'a mut [u8]) -> Self {
        Self {
            out,
            at: 0,
            overflow: false,
        }
    }

    /// 已写字节数。
    pub fn written(&self) -> usize {
        self.at
    }

    /// 是否有过溢出。
    pub fn overflow(&self) -> bool {
        self.overflow
    }
}

impl Sink for SliceSink<'_> {
    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        let Some(end) = self.at.checked_add(bytes.len()) else {
            self.overflow = true;
            return;
        };
        let Some(slot) = self.out.get_mut(self.at..end) else {
            self.overflow = true;
            return;
        };
        slot.copy_from_slice(bytes);
        self.at = end;
    }
}

/// 直接写进 IO 后端：整份容器从不进内存；写失败只记第一次，之后追加为空操作。
pub struct FileSink<W> {
    /// 底层写入器（通常是 `BufWriter<File>`）。
    inner: W,
    /// 已写字节数。
    written: usize,
    /// 第一次写错误（若有）。
    error: Option<io::Error>,
}

impl<W: io::Write> FileSink<W> {
    /// 绑定 IO 后端。
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            written: 0,
            error: None,
        }
    }

    /// 已写字节数。
    pub fn written(&self) -> usize {
        self.written
    }

    /// 取出第一次写错误。
    pub fn error(&self) -> Option<&io::Error> {
        self.error.as_ref()
    }

    /// 取走第一次写错误（消费所有权；`io::Error` 不可克隆）。
    pub fn take_error(&mut self) -> Option<io::Error> {
        self.error.take()
    }

    /// 拆解出底层写入器。
    pub fn into_inner(self) -> io::Result<W> {
        // 显式 flush，让缓冲内残留字节在拆出前落地。
        let mut inner = self.inner;
        inner.flush()?;
        Ok(inner)
    }
}

impl<W: io::Write> Sink for FileSink<W> {
    fn put(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        match self.inner.write_all(bytes) {
            Ok(()) => self.written += bytes.len(),
            Err(error) => self.error = Some(error),
        }
    }
}

/// 透传且精确计数：需要当前写入位置（做对齐）时查询，长度始终准确。
pub struct Measured<'a, S> {
    /// 下游。
    inner: &'a mut S,
    /// 已写字节数。
    len: usize,
}

impl<'a, S: Sink> Measured<'a, S> {
    /// 包装下游。
    pub fn new(inner: &'a mut S) -> Self {
        Self { inner, len: 0 }
    }

    /// 已写字节数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否还没写过任何字节。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<S: Sink> Sink for Measured<'_, S> {
    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        self.len += bytes.len();
        self.inner.put(bytes);
    }
}

/// 一边写一边更新 CRC：经过它的每个字节都进入校验，范围不可能漏字段。
pub struct Sealed<'a, S> {
    /// 被包装的下游。
    inner: &'a mut S,
    /// 当前 CRC 状态（内部表示，取结果时取反）。
    state: u32,
}

impl<'a, S: Sink> Sealed<'a, S> {
    /// 包装下游，从 CRC 初始内部状态开始。
    pub fn new(inner: &'a mut S) -> Self {
        Self { inner, state: !0 }
    }

    /// 当前 CRC-32 结果。
    pub fn checksum(&self) -> u32 {
        !self.state
    }
}

impl<S: Sink> Sink for Sealed<'_, S> {
    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        self.state = crc::update(self.state, bytes);
        self.inner.put(bytes);
    }
}

#[cfg(test)]
#[path = "../tests/sink_tests.rs"]
mod tests;
