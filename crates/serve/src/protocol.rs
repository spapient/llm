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

pub use crate::frame::{Decoder, Envelope, Frame};

pub mod macros {
    //! 编译期表驱动宏：闭集由一张表展开变体、`ALL` 与映射，同一个清单不再手抄多遍。

    /// 编号闭集：变体、`ALL` 与 `code()` 由同一张表展开。
    macro_rules! code_set {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $( $(#[$vmeta:meta])* $variant:ident => $code:expr ),+ $(,)?
    }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// 全部成员，供遍历与用例。
            $vis const ALL: &'static [Self] = &[ $(Self::$variant),+ ];

            /// 对应的编码。
            $vis fn code(self) -> u16 {
                match self { $(Self::$variant => $code),+ }
            }
        }
    };
}

    /// 短名闭集：变体、`ALL`、`as_str()` 与 `Display` 由同一张表展开。
    macro_rules! name_set {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $( $(#[$vmeta:meta])* $variant:ident => $text:literal ),+ $(,)?
    }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// 全部成员，供遍历与用例。
            $vis const ALL: &'static [Self] = &[ $(Self::$variant),+ ];

            /// 稳定的短名：写进日志与诊断，不随翻译变化。
            $vis fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

    /// 双轴闭集：编号 + 短名；`from_code` 由 `ALL` 驱动，不再手写第二张解码表。
    macro_rules! code_name_set {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $( $(#[$vmeta:meta])* $variant:ident => $code:expr, $text:literal ),+ $(,)?
    }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// 全部成员，供遍历与用例。
            $vis const ALL: &'static [Self] = &[ $(Self::$variant),+ ];

            /// 线上编码。
            $vis fn code(self) -> u16 {
                match self { $(Self::$variant => $code),+ }
            }

            /// 稳定的短名：写进日志与诊断，不随翻译变化（线上用的是 [`Self::code`]）。
            $vis fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }

            /// 由编码还原；未知编码返回 `None`。
            $vis fn from_code(code: u16) -> Option<Self> {
                Self::ALL.iter().find(|item| item.code() == code).copied()
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

    pub(crate) use {code_name_set, code_set, name_set};
}

use self::macros::code_name_set;

/// 魔数：`0x4C4C`（"LL"），用来在第一时间拒绝"连错端口"。
pub const MAGIC: u16 = 0x4c4c;

/// 帧头长度：魔数 2 + 版本 2 + 类型 2 + 载荷长度 4。
pub const HEADER_LEN: usize = 10;

/// 校验尾长度：第 2 版起，每帧末尾 4 字节 CRC-32，覆盖帧头与载荷。
pub const TRAILER_LEN: usize = 4;

/// 单帧载荷上限：1 MiB。
pub const MAX_PAYLOAD: usize = 1 << 20;

/// 协议版本。
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
mod tests {
    use super::*;
    use diagnostic::Code;

    fn envelope() -> Envelope {
        Envelope::new(Version::current(), Kind::Chunk, b"payload".to_vec())
    }

    testkit::test_cases! {
    fn framing_round_trips() {
        let frame = envelope().frame().expect("编码");
        assert_eq!(frame.len(), HEADER_LEN + 7 + TRAILER_LEN);
        assert_eq!(Envelope::parse(&frame).expect("解析"), envelope());
    }

    fn an_empty_payload_is_legal() {
        let empty = Envelope::new(Version::current(), Kind::Done, Vec::new());
        let frame = empty.frame().expect("编码");
        assert_eq!(frame.len(), HEADER_LEN + TRAILER_LEN);
        assert_eq!(Envelope::parse(&frame).expect("解析").payload(), b"");
    }

    /// 字节布局是对端契约：帧头一个字段都不能挪，校验尾只在第 2 版及以后存在。
    fn the_wire_layout_is_pinned() {
        let first = Envelope::new(Version::FIRST, Kind::Chunk, vec![1, 2, 3])
            .frame()
            .expect("编码");
        assert_eq!(
            first,
            [
                0x4c, 0x4c, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 1, 2, 3
            ]
        );

        let current = Envelope::new(Version::current(), Kind::Chunk, vec![1, 2, 3])
            .frame()
            .expect("编码");
        assert_eq!(
            &current[4..first.len()],
            &first[4..],
            "版本之后逐字节相同：类型、长度与载荷都没有挪位置"
        );
        assert_eq!(
            &current[first.len()..],
            &crate::crc32(&current[..first.len()]).to_le_bytes()
        );
    }

    fn a_frame_keeps_the_version_it_was_written_with() {
        for version in [Version::FIRST, Version::current()] {
            let envelope = Envelope::new(version, Kind::Done, vec![9]);
            let frame = envelope.frame().expect("编码");
            let trailer = if version == Version::FIRST {
                0
            } else {
                TRAILER_LEN
            };
            assert_eq!(frame.len(), HEADER_LEN + 1 + trailer);
            let parsed = Envelope::parse(&frame).expect("解析");
            assert_eq!(parsed.version(), version, "版本是往返的一部分");
            assert_eq!(parsed, envelope);
        }
    }

    /// 帧头里的长度来自对端：16 进制的极限值不能变成算术溢出，也不能变成一次大分配。
    fn an_absurd_declared_length_is_an_error_not_an_overflow() {
        let mut frame = envelope().frame().expect("编码");
        frame[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = Envelope::parse(&frame).expect_err("必须拒绝");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("上限"), "{error}");
    }

    /// 一个缓冲里有多帧时只解析第一帧，剩下的交给调用方。
    fn trailing_bytes_belong_to_the_next_frame() {
        let first = envelope().frame().expect("编码");
        let second = Envelope::new(Version::current(), Kind::Done, b"bye".to_vec())
            .frame()
            .expect("编码");
        let mut stream = first;
        stream.extend_from_slice(&second);

        assert_eq!(Envelope::parse(&stream).expect("只解析第一帧"), envelope());
        let (parsed, rest) = Envelope::parse_rest(&stream).expect("第一帧");
        assert_eq!(parsed, envelope());
        assert_eq!(rest, &second[..]);

        // 借用的那条路必须与拥有的那条路给出同一个结论，连"剩下多少"都一致。
        let (view, view_rest) = Envelope::view(&stream).expect("第一帧");
        assert_eq!(view.envelope(), parsed);
        assert_eq!(view_rest, rest);
        assert_eq!(view.payload(), b"payload");
    }

    /// 一次编码失败不该在流里留下半帧：调用方的缓冲原样不动。
    fn a_failed_encode_leaves_the_buf_untouched() {
        let mut out = b"prefix".to_vec();
        let huge = Envelope::new(Version::current(), Kind::Chunk, vec![0; MAX_PAYLOAD + 1]);
        assert_eq!(huge.frame().expect_err("超限").code(), Code::Invalid);
        assert!(huge.encode_into(&mut out).is_err());
        assert_eq!(out, b"prefix");
    }

    fn encode_into_appends_without_disturbing_earlier_frames() {
        let first = envelope();
        let second = Envelope::new(Version::FIRST, Kind::Done, b"bye".to_vec());
        let mut stream = Vec::new();
        first.encode_into(&mut stream).expect("编码");
        second.encode_into(&mut stream).expect("编码");
        assert_eq!(
            stream.len(),
            first.encoded_len().expect("长度") + second.encoded_len().expect("长度")
        );
        assert_eq!(
            &stream[first.encoded_len().expect("长度")..],
            &second.frame().expect("编码")[..]
        );
    }

    fn a_wrong_magic_is_rejected() {
        let mut frame = envelope().frame().expect("编码");
        frame[0] = 0xff;
        assert_eq!(Envelope::parse(&frame).unwrap_err().code(), Code::Invalid);
    }

    fn a_future_version_is_unsupported_not_invalid() {
        let mut frame = Envelope::new(Version::FIRST, Kind::Request, Vec::new())
            .frame()
            .expect("编码");
        let future = Version::current().get() + 1;
        frame[2..4].copy_from_slice(&future.to_le_bytes());
        assert_eq!(
            Envelope::parse(&frame).unwrap_err().code(),
            Code::Unsupported
        );
    }

    /// 第 0 版不是"将来"，是本端或对端的 bug：类别必须与"版本过新"分开。
    fn version_zero_is_invalid_and_cannot_be_written() {
        assert_eq!(
            Envelope::new(Version::new(0), Kind::Request, Vec::new())
                .frame()
                .expect_err("本端写不出版本 0")
                .code(),
            Code::Invalid
        );

        let mut frame = Envelope::new(Version::FIRST, Kind::Request, Vec::new())
            .frame()
            .expect("编码");
        frame[2..4].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(Envelope::parse(&frame).unwrap_err().code(), Code::Invalid);
    }

    /// 载荷的每一位都被校验和覆盖：长度对得上不等于内容对得上。
    fn a_flipped_payload_bit_fails_the_checksum() {
        for idx in [HEADER_LEN, HEADER_LEN + 3] {
            let mut frame = envelope().frame().expect("编码");
            frame[idx] ^= 0x01;
            let error = Envelope::parse(&frame).expect_err("必须拒绝");
            assert_eq!(error.code(), Code::Invalid);
            assert!(error.to_string().contains("校验和"), "{error}");
        }
    }

    /// 帧头也在校验范围内，而帧头里唯一“改成另一个值也合法”的字段是类型：
    /// 把 `chunk` 改成 `error` 之后帧头检查全部通过，只有校验和能发现它。
    fn a_tampered_type_is_caught_only_by_the_checksum() {
        let mut frame = envelope().frame().expect("编码");
        assert_eq!(frame[4], 2, "第 5 字节是 `Kind::Chunk` 的线上编码");
        frame[4] = 3;
        assert_eq!(
            Kind::from_code(3),
            Some(Kind::Error),
            "改后的值本身是合法类型"
        );
        let error = Envelope::parse(&frame).expect_err("必须拒绝");
        assert!(error.to_string().contains("校验和"), "{error}");
    }

    /// 第 1 版没有校验尾：同一个载荷的第 1 版帧必须照旧能读，不能因为"本端升级了"就拒收。
    fn a_first_version_frame_still_parses_after_the_format_grew_a_trailer() {
        let old = Envelope::new(Version::FIRST, Kind::Error, b"gone".to_vec())
            .frame()
            .expect("编码");
        let parsed = Envelope::parse(&old).expect("旧版本照旧可读");
        assert_eq!(parsed.version(), Version::FIRST);
        assert_eq!(parsed.payload(), b"gone");
    }

    fn truncated_frames_and_short_payloads_are_rejected() {
        assert_eq!(
            Envelope::parse(&[0u8; 4]).unwrap_err().code(),
            Code::Invalid
        );
        let frame = envelope().frame().expect("编码");
        for cut in [HEADER_LEN, frame.len() - 1] {
            assert_eq!(
                Envelope::parse(&frame[..cut]).unwrap_err().code(),
                Code::Invalid,
                "切到 {cut} 字节必须被拒"
            );
        }
    }

    fn kind_codes_are_a_closed_set() {
        // 字面锁定线上编码：`ALL` 顺序是唯一会被手改的地方，改了这里必须显式改期望。
        assert_eq!(
            Kind::ALL.iter().map(|kind| kind.code()).collect::<Vec<_>>(),
            [1, 2, 3, 4],
            "线上编码是对端契约，不允许随枚举顺序漂移"
        );
        for &kind in Kind::ALL {
            assert_eq!(Kind::from_code(kind.code()), Some(kind));
            assert!(!kind.as_str().is_empty());
            assert_eq!(kind.to_string(), kind.as_str());
        }
        for code in 0..=u16::MAX {
            assert_eq!(
                Kind::from_code(code).is_some(),
                Kind::ALL.iter().any(|kind| kind.code() == code),
                "`from_code({code})` 与线上编码表不一致"
            );
        }
    }

    fn a_version_knows_which_side_of_the_range_it_is_on() {
        assert!(!Version::new(0).is_supported());
        assert!(Version::FIRST.is_supported());
        assert!(Version::current().is_supported());
        assert!(!Version::new(Version::current().get() + 1).is_supported());
        assert_eq!(Version::CURRENT.to_string(), "v2");
    }
    }
}
