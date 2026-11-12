//! 线协议的帧格式：[`Version`]、[`Kind`]、[`Envelope`]。
//!
//! **拥有**：`Envelope`、`Kind`、`Version`、`MAGIC`、`HEADER_LEN`。**不拥有**：请求/响应语义
//! （归 `serve`）、载荷的序列化格式（由 `serve` 的端点决定，现在只有字节块）。
//! **依赖**：`diagnostic`。**被依赖**：`serve`。
//!
//! 帧头固定 10 字节，全部小端：`magic(u16) | version(u16) | kind(u16) | len(u32)`。
//! 固定长度 + 单条魔数让"读到半截连接"与"读错协议"这两种故障可以区分开——这是服务端
//! 唯一无法靠重试修复的两类错误。
//!
//! # 例子
//!
//! ```
//! use llm_proto::{Envelope, Kind, Version};
//!
//! let envelope = Envelope::new(Version::current(), Kind::Chunk, b"hi".to_vec());
//! let frame = envelope.frame().expect("帧合法");
//! assert_eq!(frame.len(), llm_proto::HEADER_LEN + 2);
//! assert_eq!(Envelope::parse(&frame).expect("解析"), envelope);
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};

/// 魔数：`0x4C4C`（"LL"），用来在第一时间拒绝"连错端口"。
pub const MAGIC: u16 = 0x4c4c;

/// 帧头长度：魔数 2 + 版本 2 + 类型 2 + 载荷长度 4。
pub const HEADER_LEN: usize = 10;

/// 协议版本。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(u16);

impl Version {
    /// 由编号构造。
    pub fn new(value: u16) -> Self {
        Self(value)
    }

    /// 当前版本：写入时用这个值，解析时以它为上界。
    pub fn current() -> Self {
        Self(1)
    }

    /// 编号。
    pub fn get(&self) -> u16 {
        self.0
    }
}

/// 帧类型闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// 一个请求的开始。
    Request,
    /// 一个流式数据块。
    Chunk,
    /// 一个错误响应。
    Error,
    /// 流的结束。
    Done,
}

impl Kind {
    /// 全部类型，供遍历。
    pub const ALL: [Self; 4] = [Self::Request, Self::Chunk, Self::Error, Self::Done];

    /// 线上编码。
    pub fn code(self) -> u16 {
        match self {
            Self::Request => 1,
            Self::Chunk => 2,
            Self::Error => 3,
            Self::Done => 4,
        }
    }

    /// 由编码还原；未知编码返回 `None`。
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Request),
            2 => Some(Self::Chunk),
            3 => Some(Self::Error),
            4 => Some(Self::Done),
            _ => None,
        }
    }
}

/// 一帧：版本 + 类型 + 原始载荷字节。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Envelope {
    version: Version,
    kind: Kind,
    payload: Vec<u8>,
}

impl Envelope {
    /// 由版本、类型与载荷构造。
    pub fn new(version: Version, kind: Kind, payload: Vec<u8>) -> Self {
        Self {
            version,
            kind,
            payload,
        }
    }

    /// 版本。
    pub fn version(&self) -> Version {
        self.version
    }

    /// 类型。
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// 载荷。
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// 编码成可以写进连接的字节串。
    pub fn frame(&self) -> Result<Vec<u8>> {
        let len = u32::try_from(self.payload.len()).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("载荷 {} 字节超出 u32", self.payload.len()),
            )
            .with("proto::Envelope::frame")
        })?;
        let mut frame = Vec::with_capacity(HEADER_LEN + self.payload.len());
        frame.extend_from_slice(&MAGIC.to_le_bytes());
        frame.extend_from_slice(&self.version.get().to_le_bytes());
        frame.extend_from_slice(&self.kind.code().to_le_bytes());
        frame.extend_from_slice(&len.to_le_bytes());
        frame.extend_from_slice(&self.payload);
        Ok(frame)
    }

    /// 解析一帧；短于帧头、魔数不符、版本过新、类型未知、载荷比声明的短都报错。
    ///
    /// 多出来的字节不动：流式连接上一个缓冲里可能躺着好几帧，调用方按 [`HEADER_LEN`] 加
    /// [`Envelope::payload`] 的长度切下一帧，本函数只保证"头 + 声明的载荷"这一段可用。
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::new(
                Code::Invalid,
                format!("帧长 {} 字节，短于帧头 {HEADER_LEN} 字节", bytes.len()),
            )
            .with("proto::Envelope::parse"));
        }
        let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
        if magic != MAGIC {
            return Err(Error::new(
                Code::Invalid,
                format!("魔数不匹配：期望 {MAGIC:#06x}，得到 {magic:#06x}"),
            )
            .with("proto::Envelope::parse"));
        }
        let version = Version::new(u16::from_le_bytes([bytes[2], bytes[3]]));
        if version > Version::current() {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "协议版本 {} 高于本端支持的 {}",
                    version.get(),
                    Version::current().get()
                ),
            )
            .with("proto::Envelope::parse"));
        }
        let code = u16::from_le_bytes([bytes[4], bytes[5]]);
        let kind = Kind::from_code(code).ok_or_else(|| {
            Error::new(Code::Invalid, format!("未知帧类型 {code}")).with("proto::Envelope::parse")
        })?;
        let len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        // 先切掉帧头再比长度：`len` 是别人发来的数，可以是 `0xFFFF_FFFF`，而 32 位平台上
        // `HEADER_LEN + len` 会溢出——那会变成 debug 构建里的 panic，也就是一个可以用 10 字节
        // 打出来的拒绝服务。
        let rest = &bytes[HEADER_LEN..];
        if rest.len() < len {
            return Err(Error::new(
                Code::Invalid,
                format!("载荷声明 {len} 字节，实际只有 {} 字节", rest.len()),
            )
            .with("proto::Envelope::parse"));
        }
        let payload = &rest[..len];
        Ok(Self {
            version,
            kind,
            payload: payload.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> Envelope {
        Envelope::new(Version::current(), Kind::Chunk, b"payload".to_vec())
    }

    #[test]
    fn framing_round_trips() {
        let frame = envelope().frame().expect("编码");
        assert_eq!(frame.len(), HEADER_LEN + 7);
        assert_eq!(Envelope::parse(&frame).expect("解析"), envelope());
    }

    #[test]
    fn an_empty_payload_is_legal() {
        let empty = Envelope::new(Version::current(), Kind::Done, Vec::new());
        let frame = empty.frame().expect("编码");
        assert_eq!(frame.len(), HEADER_LEN);
        assert_eq!(Envelope::parse(&frame).expect("解析").payload(), b"");
    }

    /// 帧头里的长度来自对端：16 进制的极限值不能变成算术溢出。
    #[test]
    fn an_absurd_declared_length_is_an_error_not_an_overflow() {
        let mut frame = envelope().frame().expect("编码");
        frame[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = Envelope::parse(&frame).expect_err("必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("载荷声明"), "{error}");
    }

    /// 一个缓冲里有多帧时只解析第一帧，剩下的留给调用方。
    #[test]
    fn trailing_bytes_belong_to_the_next_frame() {
        let first = envelope().frame().expect("编码");
        let second = Envelope::new(Version::current(), Kind::Done, b"bye".to_vec())
            .frame()
            .expect("编码");
        let mut stream = first.clone();
        stream.extend_from_slice(&second);
        let parsed = Envelope::parse(&stream).expect("只解析第一帧");
        assert_eq!(parsed, envelope());
        let consumed = HEADER_LEN + parsed.payload().len();
        assert_eq!(Envelope::parse(&stream[consumed..]).expect("第二帧"), {
            Envelope::new(Version::current(), Kind::Done, b"bye".to_vec())
        });
    }

    #[test]
    fn a_wrong_magic_is_rejected() {
        let mut frame = envelope().frame().expect("编码");
        frame[0] = 0xff;
        assert_eq!(Envelope::parse(&frame).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_future_version_is_unsupported_not_invalid() {
        let frame = Envelope::new(Version::new(99), Kind::Request, Vec::new())
            .frame()
            .expect("编码");
        assert_eq!(
            Envelope::parse(&frame).unwrap_err().code(),
            Code::Unsupported
        );
    }

    #[test]
    fn truncated_frames_and_short_payloads_are_rejected() {
        assert_eq!(
            Envelope::parse(&[0u8; 4]).unwrap_err().code(),
            Code::Invalid
        );
        let frame = envelope().frame().expect("编码");
        let truncated = &frame[..frame.len() - 1];
        assert_eq!(
            Envelope::parse(truncated).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn kind_codes_are_a_closed_set() {
        for kind in Kind::ALL {
            assert_eq!(Kind::from_code(kind.code()), Some(kind));
        }
        assert_eq!(Kind::from_code(0), None);
        assert_eq!(Kind::from_code(9), None);
    }
}
