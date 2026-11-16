//! 线协议的帧格式：[`Version`]、[`Kind`]、[`Envelope`] 与增量解码器 [`Decoder`]。
//!
//! **拥有**：`Envelope`、`Frame`、`Decoder`、`Kind`、`Version`、`MAGIC`、`HEADER_LEN`、
//! `TRAILER_LEN`、`MAX_PAYLOAD`。**不拥有**：请求/响应语义（归 [`crate::Endpoint`]）、载荷的
//! 序列化格式（由端点决定，现在只有字节块）。**依赖**：`diagnostic`。**被依赖**：仅本 crate 的
//! [`crate::Response::frame`] 与 [`crate::Endpoint`]——它是"端点契约的线格式"，不是独立成员。
//!
//! # 帧
//!
//! 帧头固定 10 字节，全部小端；第 2 版起帧尾再带 4 字节 CRC-32：
//!
//! ```text
//! magic(u16) | version(u16) | kind(u16) | len(u32) | payload[len] | crc32(u32)   ← v2
//! magic(u16) | version(u16) | kind(u16) | len(u32) | payload[len]               ← v1
//! ```
//!
//! 三件事都由这一个布局解决：
//!
//! - **固定长度 + 单条魔数**让"读到半截连接"与"读错协议"可以区分开——这是服务端唯一无法靠
//!   重试修复的两类错误；
//! - **载荷上限 [`MAX_PAYLOAD`]** 是协议常量而不是配置项：对端声明 4 GiB 时，接收方要么用
//!   [`HEADER_LEN`] 个字节就拒绝，要么替对端分配 4 GiB；
//! - **校验尾**让"长度正确、内容坏了"在解码期失败，而不是在若干层之后表现成一串合法却错误的数。
//!
//! 版本只决定布局，不决定语义；收到的版本在往返里保留（[`Envelope::version`]），所以第 1 版的
//! 对端不会被本端"顺手升级"。
//!
//! # 例子
//!
//! ```
//! use llm_serve::proto::{Envelope, Kind, Version};
//!
//! let envelope = Envelope::new(Version::current(), Kind::Chunk, b"hi".to_vec());
//! let frame = envelope.frame().expect("帧合法");
//! assert_eq!(
//!     frame.len(),
//!     llm_serve::proto::HEADER_LEN + 2 + llm_serve::proto::TRAILER_LEN
//! );
//! assert_eq!(Envelope::parse(&frame).expect("解析"), envelope);
//! ```
//!
//! 连接上的字节是分批到的，一帧可能只到了半个、一个缓冲里也可能躺着好几帧：
//!
//! ```
//! use llm_serve::proto::{Decoder, Envelope, Kind, Version};
//!
//! let mut wire = Vec::new();
//! for kind in [Kind::Request, Kind::Chunk, Kind::Done] {
//!     let envelope = Envelope::new(Version::current(), kind, b"x".to_vec());
//!     wire.extend_from_slice(&envelope.frame().expect("编码"));
//! }
//!
//! let mut decoder = Decoder::new();
//! let mut kinds = Vec::new();
//! for byte in &wire {
//!     decoder.push(std::slice::from_ref(byte)).expect("追加");
//!     while let Some(frame) = decoder.next_frame().expect("解析") {
//!         kinds.push(frame.kind());
//!     }
//! }
//! assert_eq!(kinds, [Kind::Request, Kind::Chunk, Kind::Done]);
//! decoder.finish().expect("正好停在帧边界");
//! ```

#![deny(unsafe_code)]

use core::fmt;

pub use crate::decoder::Decoder;
pub use crate::frame::{Envelope, Frame};

/// 魔数：`0x4C4C`（"LL"），用来在第一时间拒绝"连错端口"。
pub const MAGIC: u16 = 0x4c4c;

/// 帧头长度：魔数 2 + 版本 2 + 类型 2 + 载荷长度 4。
pub const HEADER_LEN: usize = 10;

/// 校验尾长度：第 2 版起，每帧末尾 4 字节 CRC-32，覆盖帧头与载荷。
pub const TRAILER_LEN: usize = 4;

/// 单帧载荷上限：1 MiB。
///
/// 这是协议常量而非配置项：接收方的上限必须与发送方一致，否则"本端写得出、对端收不下"就成了
/// 一个只在特定连接上出现的失败。更大的载荷要切成多帧——一帧是一段字节块，不是一条完整请求。
pub const MAX_PAYLOAD: usize = 1 << 20;

/// 协议版本。
///
/// 版本决定一帧的**布局**，不决定语义：第 1 版没有校验尾，第 2 版起每帧末尾 4 字节 CRC-32。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(u16);

impl Version {
    /// 第 1 版：10 字节帧头 + 载荷。
    pub const FIRST: Self = Self(1);

    /// 当前版本：写入时用这个值，解析时以它为上界。
    pub const CURRENT: Self = Self(2);

    /// 校验尾从这一版开始存在。
    const CHECKSUM_FROM: u16 = 2;

    /// 由编号构造。编号是否被支持由 [`Version::is_supported`] 判定——构造不出来不代表能说。
    pub fn new(value: u16) -> Self {
        Self(value)
    }

    /// 当前版本：写入时用这个值，解析时以它为上界。
    pub fn current() -> Self {
        Self::CURRENT
    }

    /// 编号。
    pub fn get(&self) -> u16 {
        self.0
    }

    /// 本端是否支持这个版本（`FIRST..=CURRENT`）。
    pub fn is_supported(self) -> bool {
        (Self::FIRST.0..=Self::CURRENT.0).contains(&self.0)
    }

    /// 这一版的帧末尾有没有 4 字节 CRC-32。
    pub(crate) fn has_checksum(self) -> bool {
        self.0 >= Self::CHECKSUM_FROM
    }
}

impl fmt::Display for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "v{}", self.0)
    }
}

code_name_set! {
    /// 帧类型闭集。
    ///
    /// 加一个变体只改表内一行：类型是日志与对端行为的对外契约。[`Kind::from_code`] 由 `ALL`
    /// 推出，不再单独维护一张解码表——那样漏一处只有发送端能编、接收端却认不出，正是跨版本
    /// 互通最难查的一类问题。
    pub enum Kind {
        /// 一个请求的开始。
        Request => 1, "request",
        /// 一个流式数据块。
        Chunk => 2, "chunk",
        /// 一个错误响应。
        Error => 3, "error",
        /// 流的结束。
        Done => 4, "done",
    }
}

#[cfg(test)]
#[path = "../tests/protocol_tests.rs"]
mod tests;
