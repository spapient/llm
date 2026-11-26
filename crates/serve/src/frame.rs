//! 帧的编解码：[`Envelope`]（拥有载荷）与 [`Frame`]（借用载荷）。

use diagnostic::{Code, Error, Result};

use crate::crc32;
use crate::protocol::{HEADER_LEN, Kind, MAGIC, MAX_PAYLOAD, TRAILER_LEN, Version};

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
    fn total(&self) -> usize {
        HEADER_LEN + self.len + trailer_len(self.version)
    }
}

/// 校验尾字节数：第 2 版起每帧 4 字节。
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    version: Version,
    kind: Kind,
    payload: &'a [u8],
}

impl<'a> Frame<'a> {
    /// 由扫描结果组装：只有本模块与 [`crate::protocol::Decoder`] 能造，格式已在上游校验过。
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

    /// 整帧字节数（帧头 + 载荷 + 校验尾）。
    pub fn encoded_len(&self) -> Result<usize> {
        self.check_frame()
            .map_err(|error| error.with("proto::Envelope::encoded_len"))?;
        Ok(HEADER_LEN + self.payload.len() + trailer_len(self.version))
    }

    /// 编码成一帧，复用调用方已有的缓冲。
    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let len = self
            .check_frame()
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
    fn check_frame(&self) -> Result<u32> {
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

/// 增量解码器。
#[derive(Debug, Default)]
pub struct Decoder {
    buf: Vec<u8>,
    /// 已交付前缀的终点：`buf[..start]` 是废料，下次 `push` 时再压缩。
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
            buf: Vec::with_capacity(bytes),
            ..Self::default()
        }
    }

    /// 追加刚读到的字节。
    pub fn push(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(error) = &self.poisoned {
            return Err(error.clone().with("proto::Decoder::push"));
        }
        self.compact();
        self.buf.extend_from_slice(bytes);
        let available = &self.buf[self.start..];
        if available.len() >= HEADER_LEN
            && let Err(error) = header(available)
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
        match scan(&self.buf[self.start..]) {
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
        self.buf.len() - self.start
    }

    /// 连接关闭时调用：只有正好停在帧边界才是正常关闭。
    pub fn finish(self) -> Result<()> {
        if let Some(error) = self.poisoned {
            return Err(error);
        }
        let left = self.buf.len() - self.start;
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
        self.buf.clear();
        self.start = 0;
        self.poisoned = None;
    }

    /// 把已交付的前缀挪掉。只在没有借出去的帧时调用（即 [`Decoder::push`]）。
    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
            return;
        }
        // 拷一次的代价是"留下的字节数"，所以只在挪掉的不少于留下的时候才做：每次 push 至多
        // 一次，摊还下来仍是每字节 O(1)，而不是随帧数增长的二次搬运。
        if self.start * 2 >= self.buf.len() {
            self.buf.copy_within(self.start.., 0);
            self.buf.truncate(self.buf.len() - self.start);
            self.start = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    //! `serve::frame::Decoder` 的用例：头、校验和与分帧。
    //!
    //! 生产文件只留执行与判据，用例集中于此。

    use super::*;
    use crate::protocol::{Envelope, Kind, Version};

    /// 一条混着四种类型、两种版本、若干载荷长度的流。
    fn stream() -> (Vec<u8>, Vec<Envelope>) {
        let mut bytes = Vec::new();
        let mut frames = Vec::new();
        for (idx, kind) in Kind::ALL.iter().copied().enumerate() {
            let version = if idx % 2 == 0 {
                Version::FIRST
            } else {
                Version::current()
            };
            let envelope = Envelope::new(version, kind, vec![idx as u8; idx * 3]);
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

    testkit::test_cases! {
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

    fn frames_arriving_together_are_all_delivered() {
        let (bytes, frames) = stream();
        let mut decoder = Decoder::new();
        decoder.push(&bytes).expect("一次全给");
        assert_eq!(drain(&mut decoder), frames);
        decoder.finish().expect("正好停在帧边界");
    }

    /// 每一个切法都必须得到同一条帧序列：`start` / 压缩 / 半帧这些状态机错误只在某个切法下出现。
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
}
