//! 增量解码：[`Decoder`] 把"从连接上读到的字节"变成一帧一帧。

use diagnostic::{Code, Error, Result};

use crate::frame::{self, Frame, Scan};
use crate::protocol::HEADER_LEN;

/// 增量解码器。
#[derive(Debug, Default)]
pub struct Decoder {
    buffer: Vec<u8>,
    /// 已交付前缀的终点：`buffer[..start]` 是废料，下次 `push` 时再压缩。
    start: usize,
    poisoned: Option<Error>,
}

impl Decoder {
    /// 空解码器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 按预期先分配缓冲：连接上未处理的字节大概有多少。
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(bytes),
            ..Self::default()
        }
    }

    /// 追加刚读到的字节。
    pub fn push(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(error) = &self.poisoned {
            return Err(error.clone().with("proto::Decoder::push"));
        }
        self.compact();
        self.buffer.extend_from_slice(bytes);
        let available = &self.buffer[self.start..];
        if available.len() >= HEADER_LEN
            && let Err(error) = frame::header(available)
        {
            let error = error.with("proto::Decoder::push");
            self.poisoned = Some(error.clone());
            return Err(error);
        }
        Ok(())
    }

    /// 取下一帧；一帧还没到齐时返回 `None`。
    pub fn next_frame(&mut self) -> Result<Option<Frame<'_>>> {
        if let Some(error) = &self.poisoned {
            return Err(error.clone().with("proto::Decoder::next_frame"));
        }
        match frame::scan(&self.buffer[self.start..]) {
            Ok(Scan::Short | Scan::Partial { .. }) => Ok(None),
            Ok(Scan::Frame {
                header,
                payload,
                total,
            }) => {
                self.start += total;
                Ok(Some(Frame::new(header.version, header.kind, payload)))
            }
            Err(error) => {
                let error = error.with("proto::Decoder::next_frame");
                self.poisoned = Some(error.clone());
                Err(error)
            }
        }
    }

    /// 还没被取走的字节数（含半帧）。
    pub fn buffered(&self) -> usize {
        self.buffer.len() - self.start
    }

    /// 连接关闭时调用：只有正好停在帧边界才是正常关闭。
    pub fn finish(self) -> Result<()> {
        if let Some(error) = self.poisoned {
            return Err(error);
        }
        let left = self.buffer.len() - self.start;
        if left == 0 {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!("连接在帧中间关闭：还剩 {left} 字节不构成一帧"),
        )
        .with("proto::Decoder::finish"))
    }

    /// 丢掉缓冲与"这条流已坏"的判定，供复用同一个对象解下一条连接。
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.start = 0;
        self.poisoned = None;
    }

    /// 把已交付的前缀挪掉。只在没有借出去的帧时调用（即 [`Decoder::push`]）。
    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }
        if self.start == self.buffer.len() {
            self.buffer.clear();
            self.start = 0;
            return;
        }
        // 拷一次的代价是"留下的字节数"，所以只在挪掉的不少于留下的时候才做：每次 push 至多
        // 一次，摊还下来仍是每字节 O(1)，而不是随帧数增长的二次搬运。
        if self.start * 2 >= self.buffer.len() {
            self.buffer.copy_within(self.start.., 0);
            self.buffer.truncate(self.buffer.len() - self.start);
            self.start = 0;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/frame_decoder_tests.rs"]
mod tests;
