//! 增量解码：[`Decoder`] 把"从连接上读到的字节"变成一帧一帧。
//!
//! 连接上的字节本来就分批到：一帧可能只到了半个帧头，一个缓冲里也可能躺着好几帧。三种结局
//! 必须能分开，否则调用方只能在"等"与"断"之间猜——[`Decoder::next_frame`] 因此是
//! `Result<Option<Frame>>`：`None` 是"还没到齐"，`Err` 是"这条流坏了"。

use diagnostic::{Code, Error, Result};

use crate::proto::HEADER_LEN;
use crate::proto::frame::{self, Frame, Scan};

/// 增量解码器。
///
/// 内存有上界：本对象持有的字节数不超过"一次 `push` 的量 + 一帧"（一帧最多
/// [`HEADER_LEN`] + [`crate::proto::MAX_PAYLOAD`] + [`crate::proto::TRAILER_LEN`] 字节），超限的
/// 声明在帧头到达时就被拒绝，不会替对端攒载荷。
///
/// 失败是一次性的：帧头坏掉之后没有可靠的重同步点（长度字段本身可能正是坏掉的那部分），所以
/// 第一次失败之后每次调用都返回同一个错误——"再喂点字节说不定又对上了"不成立。要复用这个对象
/// 解下一条连接，显式调用 [`Decoder::reset`]。
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
    ///
    /// 帧头一到就先判一次：坏魔数、不认识的版本、超限的载荷声明都不该等到"攒够载荷"才被发现
    /// ——那正是拒绝服务最容易换到东西的位置。
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
    ///
    /// 交付的 [`Frame`] 借用本对象的缓冲，不复制载荷：一帧处理完再取下一帧，中间不要 `push`。
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
    ///
    /// "读完了"与"读到一半断了"是两件事：前者最后一条消息完整到达，后者少了一截，把它当成前者
    /// 就是这个协议能犯的最安静的错。
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
mod tests;
