//! 帧的编解码：[`Envelope`]（拥有载荷）与 [`Frame`]（借用载荷）。
//!
//! 两者都从同一段字节出发、经同一趟 [`scan`]：分帧、版本判定、上限判定与校验只有一份实现，
//! 所以"拥有的那条路"与"借用的那条路"不可能对同一段字节给出不同结论。

use diagnostic::{Code, Error, Result};

use crate::proto::crc::crc32;
use crate::proto::{HEADER_LEN, Kind, MAGIC, MAX_PAYLOAD, TRAILER_LEN, Version};

/// 解析出来的帧头。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Header {
    /// 帧的版本：决定有没有校验尾。
    pub(crate) version: Version,
    /// 帧类型。
    pub(crate) kind: Kind,
    /// 载荷字节数（已被上限校验过）。
    pub(crate) len: usize,
}

impl Header {
    /// 整帧字节数：帧头 + 载荷 + 校验尾。
    ///
    /// 不会溢出：`len` 在 [`header`] 里已被 [`MAX_PAYLOAD`] 挡住。
    fn total(&self) -> usize {
        HEADER_LEN + self.len + trailer_len(self.version)
    }
}

/// 校验尾字节数：第 2 版起每帧 4 字节。
///
/// 帧头的 [`Header::total`] 与 [`Envelope::encoded_len`] 都从这里取：两处各写一遍
/// `if version.has_checksum()`，将来加一版更长的尾时就会变成「帧头说 N 字节、实际切 N+4」，
/// 错位之后整条流都解不开。
fn trailer_len(version: Version) -> usize {
    if version.has_checksum() {
        TRAILER_LEN
    } else {
        0
    }
}

/// 一次扫描的结局。`Err` 是坏帧，三态才是"字节不够"与"有一帧"——把"还没到齐"混进错误里，
/// 流式解码就只能在"等"与"断"之间猜。
pub(crate) enum Scan<'a> {
    /// 帧头还没到齐。
    Short,
    /// 帧头齐了，整帧还没到齐：`total` 是声明的整帧字节数。
    Partial {
        /// 声明的整帧字节数。
        total: usize,
        /// 声明的载荷字节数。
        payload: usize,
    },
    /// 有一帧。
    Frame {
        /// 帧头。
        header: Header,
        /// 借来的载荷。
        payload: &'a [u8],
        /// 整帧字节数。
        total: usize,
    },
}

/// 扫描 `bytes` 开头的一帧；多出来的字节不动，调用方按返回的 `total` 切下一帧。
pub(crate) fn scan(bytes: &[u8]) -> Result<Scan<'_>> {
    if bytes.len() < HEADER_LEN {
        return Ok(Scan::Short);
    }
    let header = header(bytes)?;
    let total = header.total();
    if bytes.len() < total {
        return Ok(Scan::Partial {
            total,
            payload: header.len,
        });
    }
    let payload = &bytes[HEADER_LEN..HEADER_LEN + header.len];
    verify_checksum(bytes, &header)?;
    Ok(Scan::Frame {
        header,
        payload,
        total,
    })
}

/// 校验尾必须与「帧头 + 载荷」算出的 CRC 一致；第 1 版没有校验尾，直接通过。
///
/// 回答「这帧是不是被改过」，不回答「帧有多长」——切帧与上限在 [`scan`] 里。
fn verify_checksum(bytes: &[u8], header: &Header) -> Result<()> {
    if !header.version.has_checksum() {
        return Ok(());
    }
    let at = HEADER_LEN + header.len;
    let written = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let computed = crc32(&bytes[..at]);
    if written == computed {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("校验和不符：帧头与载荷算出 {computed:#010x}，帧尾写的是 {written:#010x}"),
    ))
}

/// 解析 10 字节帧头：魔数、版本、类型与载荷长度各判一次。
///
/// [`Decoder::push`] 用它"一收到帧头就先判一次"，所以可见性是 `pub(crate)` 而不是私有。
pub(crate) fn header(bytes: &[u8]) -> Result<Header> {
    let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
    if magic != MAGIC {
        return Err(Error::new(
            Code::Invalid,
            format!("魔数不匹配：期望 {MAGIC:#06x}，得到 {magic:#06x}"),
        ));
    }
    Ok(Header {
        version: version_at(bytes)?,
        kind: kind_at(bytes)?,
        len: length_at(bytes)?,
    })
}

/// 版本字段：只有本端支持的版本才算读出来，其余交给 [`version_error`] 说清是哪一种。
fn version_at(bytes: &[u8]) -> Result<Version> {
    let version = Version::new(u16::from_le_bytes([bytes[2], bytes[3]]));
    if version.is_supported() {
        return Ok(version);
    }
    Err(version_error(version))
}

/// 不受支持的版本：比本端新只需升级本端（`Unsupported`），版本 0 之类是本端或对端的 bug
/// （`Invalid`）。合成一个码，脚本就没法区分该升级还是该查日志。
fn version_error(version: Version) -> Error {
    let current = Version::current().get();
    if version > Version::current() {
        return Error::new(
            Code::Unsupported,
            format!("协议版本 {} 高于本端支持的 {current}", version.get()),
        );
    }
    Error::new(
        Code::Invalid,
        format!(
            "协议版本 {} 无效：本端支持 {}..={current}",
            version.get(),
            Version::FIRST.get()
        ),
    )
}

/// 帧类型字段：未知 code 是坏帧。
fn kind_at(bytes: &[u8]) -> Result<Kind> {
    let code = u16::from_le_bytes([bytes[4], bytes[5]]);
    Kind::from_code(code).ok_or_else(|| Error::new(Code::Invalid, format!("未知帧类型 {code}")))
}

/// 载荷长度字段：先判上限再等比载荷——`len` 是对端写的数，可以是 `u32::MAX`，
/// 替对端攒够 4 GiB 等于用十个字节换一次拒绝服务。
fn length_at(bytes: &[u8]) -> Result<usize> {
    let declared = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    let len = usize::try_from(declared).unwrap_or(usize::MAX);
    if len <= MAX_PAYLOAD {
        return Ok(len);
    }
    Err(Error::new(
        Code::Invalid,
        format!("载荷声明 {len} 字节，超过单帧上限 {MAX_PAYLOAD}"),
    ))
}

/// 一帧的零拷贝视图：帧头三段 + 借来的载荷。
///
/// [`crate::proto::Decoder`] 交付的是它；要留住这帧、或要把它放进别的结构里时，用 [`Frame::envelope`] 换一份
/// 拥有载荷的 [`Envelope`]。一帧最大 [`MAX_PAYLOAD`]，所以"复制一次"的代价是有上界的。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    version: Version,
    kind: Kind,
    payload: &'a [u8],
}

impl<'a> Frame<'a> {
    /// 由扫描结果组装：只有本模块与 [`crate::proto::Decoder`] 能造，格式已在上游校验过。
    pub(crate) fn new(version: Version, kind: Kind, payload: &'a [u8]) -> Self {
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
    pub fn payload(&self) -> &'a [u8] {
        self.payload
    }

    /// 复制载荷，变成拥有的 [`Envelope`]。
    pub fn envelope(&self) -> Envelope {
        Envelope::new(self.version, self.kind, self.payload.to_vec())
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
    /// 由版本、类型与载荷构造。载荷是否超限、版本是否可编码，都在写出去的那一刻才判——
    /// 构造本身没有失败的理由。
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

    /// 取走载荷，不复制。
    pub fn into_payload(self) -> Vec<u8> {
        self.payload
    }

    /// 整帧字节数（帧头 + 载荷 + 校验尾）。
    ///
    /// 这一帧写不出去（版本本端不认识、载荷超限）时是错误——一个解不出来的帧也不该被写出去。
    pub fn encoded_len(&self) -> Result<usize> {
        self.check()
            .map_err(|error| error.with("proto::Envelope::encoded_len"))?;
        Ok(HEADER_LEN + self.payload.len() + trailer_len(self.version))
    }

    /// 编码成一帧，复用调用方已有的缓冲。
    ///
    /// 失败时 `out` 保持原样：调用方可以接着写下一帧，不会在流里留下半帧。
    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let len = self
            .check()
            .map_err(|error| error.with("proto::Envelope::encode_into"))?;
        let start = out.len();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&self.version.get().to_le_bytes());
        out.extend_from_slice(&self.kind.code().to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&self.payload);
        if self.version.has_checksum() {
            // 校验范围是"帧头 + 载荷"：帧尾自己不在里面，否则它无法被独立核对。
            out.extend_from_slice(&crc32(&out[start..]).to_le_bytes());
        }
        Ok(())
    }

    /// 编码成可以写进连接的字节串。
    pub fn frame(&self) -> Result<Vec<u8>> {
        let mut frame = Vec::with_capacity(self.encoded_len()?);
        self.encode_into(&mut frame)?;
        Ok(frame)
    }

    /// 解析一帧；短于帧头、魔数不符、版本不认识、类型未知、载荷不够或校验和不符都报错。
    ///
    /// 多出来的字节不动：流式连接上一个缓冲里可能躺着好几帧。要按帧切下一段，用
    /// [`Envelope::parse_rest`]；要避免复制载荷，用 [`Envelope::view`]。
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::parse_rest(bytes).map(|(envelope, _)| envelope)
    }

    /// 解析一帧，并把剩下的字节一并交还——调用方不必自己算"这一帧多长"。
    pub fn parse_rest(bytes: &[u8]) -> Result<(Self, &[u8])> {
        let (frame, rest) = Self::view(bytes)?;
        Ok((frame.envelope(), rest))
    }

    /// 解析一帧并借用它的载荷，不复制。
    pub fn view(bytes: &[u8]) -> Result<(Frame<'_>, &[u8])> {
        match scan(bytes).map_err(|error| error.with("proto::Envelope::view"))? {
            Scan::Short => Err(Error::new(
                Code::Invalid,
                format!("帧长 {} 字节，短于帧头 {HEADER_LEN} 字节", bytes.len()),
            )
            .with("proto::Envelope::view")),
            Scan::Partial { total, payload } => Err(Error::new(
                Code::Invalid,
                format!(
                    "帧声明 {total} 字节（载荷 {payload}），只有 {} 字节",
                    bytes.len()
                ),
            )
            .with("proto::Envelope::view")),
            Scan::Frame {
                header,
                payload,
                total,
            } => Ok((
                Frame::new(header.version, header.kind, payload),
                &bytes[total..],
            )),
        }
    }

    /// 写出去之前的自检：版本必须是本端支持的，载荷必须能装进 `u32` 且不超过上限。
    fn check(&self) -> Result<u32> {
        if !self.version.is_supported() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "协议版本 {} 无法编码：本端支持 {}..={}",
                    self.version.get(),
                    Version::FIRST.get(),
                    Version::current().get()
                ),
            ));
        }
        let len = u32::try_from(self.payload.len()).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("载荷 {} 字节超出 u32", self.payload.len()),
            )
        })?;
        if self.payload.len() > MAX_PAYLOAD {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "载荷 {} 字节超过单帧上限 {MAX_PAYLOAD}：切成多帧",
                    self.payload.len()
                ),
            ));
        }
        Ok(len)
    }
}
