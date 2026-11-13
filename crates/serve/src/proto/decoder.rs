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
/// [`HEADER_LEN`] + [`crate::proto::MAX_PAYLOAD`] + 4 字节），超限的声明在帧头到达时就被拒绝，不会替对端攒
/// 载荷。
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
mod tests {
    use super::*;
    use crate::proto::{Envelope, Kind, Version};

    /// 一条混着四种类型、两种版本、若干载荷长度的流。
    fn stream() -> (Vec<u8>, Vec<Envelope>) {
        let mut bytes = Vec::new();
        let mut frames = Vec::new();
        for (index, kind) in Kind::ALL.into_iter().enumerate() {
            let version = if index % 2 == 0 {
                Version::FIRST
            } else {
                Version::current()
            };
            let envelope = Envelope::new(version, kind, vec![index as u8; index * 3]);
            bytes.extend_from_slice(&envelope.frame().expect("编码"));
            frames.push(envelope);
        }
        (bytes, frames)
    }

    /// 把当前能取出来的帧全部取走（复制成拥有的信封，缓冲的借用随之结束）。
    fn drain(decoder: &mut Decoder) -> Vec<Envelope> {
        let mut out = Vec::new();
        while let Some(frame) = decoder.next_frame().expect("解析") {
            out.push(frame.envelope());
        }
        out
    }

    #[test]
    fn a_stream_read_one_byte_at_a_time_delivers_every_frame() {
        let (bytes, frames) = stream();
        let mut decoder = Decoder::new();
        let mut got = Vec::new();
        for byte in &bytes {
            decoder.push(std::slice::from_ref(byte)).expect("追加");
            got.extend(drain(&mut decoder));
        }
        assert_eq!(got, frames);
        assert_eq!(decoder.buffered(), 0, "每帧都被取走了，不该留下字节");
        decoder.finish().expect("正好停在帧边界");
    }

    #[test]
    fn frames_arriving_together_are_all_delivered() {
        let (bytes, frames) = stream();
        let mut decoder = Decoder::new();
        decoder.push(&bytes).expect("一次全给");
        assert_eq!(drain(&mut decoder), frames);
        decoder.finish().expect("正好停在帧边界");
    }

    /// 每一个切法都必须得到同一条帧序列：`start` / 压缩 / 半帧这些状态机错误只在某个切法下出现。
    #[test]
    fn every_way_of_splitting_a_stream_delivers_the_same_frames() {
        let (bytes, frames) = stream();
        let mut generator = 0x2545_f491_4f6c_dd1du64;
        let mut roll = move || {
            generator ^= generator << 13;
            generator ^= generator >> 7;
            generator ^= generator << 17;
            generator
        };
        for _ in 0..64 {
            let mut decoder = Decoder::new();
            let mut got = Vec::new();
            let mut at = 0;
            while at < bytes.len() {
                let end = (at + 1 + (roll() as usize) % 7).min(bytes.len());
                decoder.push(&bytes[at..end]).expect("追加");
                at = end;
                got.extend(drain(&mut decoder));
            }
            decoder.finish().expect("正好停在帧边界");
            assert_eq!(got, frames);
        }
    }

    #[test]
    fn a_partial_frame_is_not_an_error_but_finish_is() {
        let (bytes, frames) = stream();
        let mut decoder = Decoder::new();
        decoder
            .push(&bytes[..bytes.len() - 1])
            .expect("半帧不是错误");
        // 前面三帧是完整的，必须照常交付。
        assert_eq!(drain(&mut decoder), frames[..frames.len() - 1]);
        assert!(decoder.next_frame().expect("半帧不是错误").is_none());
        assert!(decoder.buffered() > 0, "最后那一帧还差一个字节");
        let error = decoder.finish().expect_err("断在帧中间");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("帧中间"), "{error}");
    }

    #[test]
    fn a_broken_header_poisons_the_stream() {
        let mut bytes = Envelope::new(Version::current(), Kind::Chunk, b"ok".to_vec())
            .frame()
            .expect("编码");
        bytes[0] = 0xff;
        let mut decoder = Decoder::new();
        assert_eq!(
            decoder.push(&bytes).expect_err("魔数不符").code(),
            Code::Invalid
        );

        // 再喂一帧好帧也不复活：帧头坏了就没有可靠的重同步点。
        let good = Envelope::new(Version::current(), Kind::Done, Vec::new())
            .frame()
            .expect("编码");
        assert!(decoder.push(&good).is_err(), "坏流不再接受字节");
        assert!(decoder.next_frame().is_err(), "坏了就不再交付帧");

        decoder.reset();
        decoder.push(&good).expect("reset 之后是另一条连接");
        assert_eq!(drain(&mut decoder).len(), 1);
    }

    #[test]
    fn an_oversized_declaration_is_rejected_before_the_payload_arrives() {
        let mut header = Envelope::new(Version::current(), Kind::Chunk, Vec::new())
            .frame()
            .expect("编码");
        header[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut decoder = Decoder::new();
        // 只给帧头：正常情况这里应该安心等载荷，但声明已经超限，必须此刻拒绝。
        assert_eq!(
            decoder
                .push(&header[..HEADER_LEN])
                .expect_err("超限")
                .code(),
            Code::Invalid
        );
        assert!(decoder.next_frame().is_err());
    }

    #[test]
    fn a_corrupted_payload_is_caught_by_the_checksum() {
        let mut bytes = Envelope::new(Version::current(), Kind::Chunk, vec![1, 2, 3])
            .frame()
            .expect("编码");
        bytes[HEADER_LEN + 1] ^= 0x01;
        let mut decoder = Decoder::new();
        // 帧头本身是好的，所以问题要到"整帧到齐"才暴露。
        decoder.push(&bytes).expect("帧头合法");
        let error = decoder.next_frame().expect_err("校验和不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("校验和"), "{error}");
        assert!(decoder.next_frame().is_err(), "坏帧之后不再重同步");
    }

    /// 一帧一帧地反复走同一条缓冲：已经交付的字节必须被丢掉，占用不能随帧数增长。
    #[test]
    fn draining_leaves_no_bytes_behind() {
        let (bytes, frames) = stream();
        let mut decoder = Decoder::with_capacity(bytes.len());
        for _ in 0..100 {
            decoder.push(&bytes).expect("追加");
            assert_eq!(drain(&mut decoder), frames);
            assert_eq!(decoder.buffered(), 0);
        }
        decoder.finish().expect("正好停在帧边界");
    }
}
