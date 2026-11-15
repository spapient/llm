//! 写端：收集段，按对齐布局一次产出 canonical 字节，同一份编码可写内存或流式落盘。
//!
//! 所有正文都通过 [`sink::Sink`] 输出：[`Writer::encoded_len`] 用计数 Sink 跑同一编码路径，
//! 与实际字节零漂移；[`Writer::write_to_path`] 用文件 Sink 边编码边写，整份容器从不进内存。
//! 全局 CRC 用 [`sink::Sealed`] 覆盖正文每一个字节，文件路径下也不会漏字段。

use std::borrow::Cow;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use diagnostic::{Code, Error, Result};

use crate::Codec;
use crate::crc;
use crate::leb;
use crate::rans;
use crate::seg::Segment;
use crate::sink::{Count, FileSink, Measured, Sealed, Sink, SliceSink};

/// 魔数 `"LLPK"`。
pub const MAGIC: u32 = 0x4B50_4C4C;

/// 当前格式版本。
pub const VERSION: u16 = 1;

/// `flags` 位：全局 CRC 存在。
pub const GLOBAL_CRC: u32 = 1 << 0;

/// `flags` 位：段级 CRC 存在。
pub const SEG_CRC: u32 = 1 << 1;

/// 最大对齐指数（2^16 = 64 KiB）。
pub const MAX_ALIGN_LOG: u8 = 16;

const WRITE_BUFFER: usize = 8 * 1024;

/// 权重写者。
#[derive(Debug, Default)]
pub struct Writer {
    segs: Vec<Segment>,
    checksum: bool,
    compress: bool,
}

impl Writer {
    /// 空写者，默认开启两级 CRC 与自适应压缩。
    pub fn new() -> Self {
        Self {
            segs: Vec::new(),
            checksum: true,
            compress: true,
        }
    }

    /// 开关两级校验和（默认开）。
    pub fn checksum(&mut self, on: bool) -> &mut Self {
        self.checksum = on;
        self
    }

    /// 开关自适应 rANS 压缩（默认开）；关闭后所有段原样存储。
    pub fn compress(&mut self, on: bool) -> &mut Self {
        self.compress = on;
        self
    }

    /// 追加一个已自洽的段；重名在这里被拒。
    pub fn push(&mut self, segment: Segment) -> Result<&mut Self> {
        segment.validate()?;
        if self.segs.iter().any(|known| known.name() == segment.name()) {
            return Err(
                Error::new(Code::Invalid, format!("段 `{}` 重复", segment.name()))
                    .with("weights::Writer::push"),
            );
        }
        self.segs.push(segment);
        Ok(self)
    }

    /// 段数。
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// 是否没有段。
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// 基于全部段的逻辑字节计算逐张量 Merkle 根；空容器返回全零。
    #[must_use]
    pub fn merkle_root(&self) -> [u8; 32] {
        let leaves: Vec<_> = self
            .segs
            .iter()
            .map(|segment| crate::proof::leaf(segment.name(), &segment.data))
            .collect();
        crate::proof::root(&leaves)
    }

    /// 为每段选定存储表示：压缩开启且 rans 帧严格变小时才用 rans，否则原样存储。
    fn prepare(&self) -> Vec<(Codec, Cow<'_, [u8]>)> {
        let mut packed = Vec::with_capacity(self.segs.len());
        for segment in &self.segs {
            if self.compress {
                let frame = rans::encode(&segment.data);
                if frame.len() < segment.data.len() {
                    packed.push((Codec::Rans, Cow::Owned(frame)));
                    continue;
                }
            }
            packed.push((Codec::Store, Cow::Borrowed(segment.data.as_slice())));
        }
        packed
    }

    fn flags(&self) -> u32 {
        if self.checksum {
            GLOBAL_CRC | SEG_CRC
        } else {
            0
        }
    }

    /// 把除全局 CRC 外的全部正文写进 `out`；位置由 [`Measured`] 精确跟踪以做对齐。
    fn write_body(&self, out: &mut impl Sink, align_log: u8) -> Result<()> {
        let packed = self.prepare();
        self.write_body_in(&packed, out, align_log)
    }

    /// 用已算好的 `packed` 写正文；长度预演与实际写出共用同一份编码结果，避免再编一遍。
    fn write_body_in(
        &self,
        packed: &[(Codec, Cow<'_, [u8]>)],
        out: &mut impl Sink,
        align_log: u8,
    ) -> Result<()> {
        let mut measured = Measured::new(out);
        measured.put(&MAGIC.to_le_bytes());
        measured.put(&VERSION.to_le_bytes());
        measured.put(&self.flags().to_le_bytes());
        measured.put(&[align_log]);
        leb::write(&mut measured, self.segs.len());
        write_descs(&self.segs, packed, self.checksum, &mut measured)?;
        write_data(packed, align_log, &mut measured);
        Ok(())
    }

    /// 产出容器字节；`align_log` 是对齐边界的 2 的幂指数（0..=16，常用 6 = 64 B）。
    pub fn finish(&self, align_log: u8) -> Result<Vec<u8>> {
        Self::check_align(align_log)?;
        let packed = self.prepare();
        let expected = {
            let mut count = Count::default();
            self.write_body_in(&packed, &mut count, align_log)?;
            count.0 + 4
        };
        let mut buf = Vec::with_capacity(expected);
        let mut sealed = Sealed::new(&mut buf);
        self.write_body_in(&packed, &mut sealed, align_log)?;
        let global = if self.checksum { sealed.checksum() } else { 0 };
        // 全局 CRC 本身不进 CRC，直接写底层缓冲。
        buf.extend_from_slice(&global.to_le_bytes());
        debug_assert_eq!(buf.len(), expected, "长度零漂移");
        Ok(buf)
    }

    /// 整份容器的精确字节数；与 [`Writer::finish`] 走同一编码路径。
    pub fn encoded_len(&self, align_log: u8) -> Result<usize> {
        let packed = self.prepare();
        let mut count = Count::default();
        self.write_body_in(&packed, &mut count, align_log)?;
        Ok(count.0 + 4)
    }

    /// 写进调用方固定缓冲，返回写入字节数；装不下时返回错误而不截断。
    pub fn encode_into(&self, align_log: u8, out: &mut [u8]) -> Result<usize> {
        Self::check_align(align_log)?;
        let mut sink = SliceSink::new(out);
        {
            let mut sealed = Sealed::new(&mut sink);
            self.write_body(&mut sealed, align_log)?;
            let global = if self.checksum { sealed.checksum() } else { 0 };
            sink.put(&global.to_le_bytes());
        }
        if sink.overflow() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲区 {} 字节装不下容器", out.len()),
            )
            .with("weights::Writer::encode_into"));
        }
        Ok(sink.written())
    }

    /// 原子地写到路径：同目录临时文件 → 边编码边写 → fsync → rename，崩溃不留半截文件。
    pub fn write_to_path(&self, align_log: u8, path: &Path) -> Result<()> {
        Self::check_align(align_log)?;
        let temp = temp_path(path);
        if let Err(error) = self.write_temp(&temp, path, align_log) {
            let _ = std::fs::remove_file(&temp);
            return Err(
                Error::new(Code::Io, format!("写入 {} 失败：{error}", path.display()))
                    .with("weights::Writer::write_to_path"),
            );
        }
        // 目录 fsync 尽力而为：部分文件系统不支持，失败不应把成功改判成失败。
        if let Some(parent) = path.parent()
            && let Ok(directory) = std::fs::File::open(parent)
        {
            let _ = directory.sync_all();
        }
        Ok(())
    }

    /// 写临时文件、刷盘、改名为目标路径。
    fn write_temp(&self, temp: &Path, target: &Path, align_log: u8) -> io::Result<()> {
        let file = std::fs::File::create(temp)?;
        let buffered = io::BufWriter::with_capacity(WRITE_BUFFER, file);
        self.flush_sink(FileSink::new(buffered), align_log)?;
        std::fs::rename(temp, target)
    }

    /// 边编码边写正文与全局 CRC，随后刷盘并 fsync。
    fn flush_sink(
        &self,
        mut file_sink: FileSink<io::BufWriter<std::fs::File>>,
        align_log: u8,
    ) -> io::Result<()> {
        let global = {
            let mut sealed = Sealed::new(&mut file_sink);
            self.write_body(&mut sealed, align_log)
                .map_err(|error| io::Error::other(error.to_string()))?;
            sealed.checksum()
        };
        file_sink.put(&global.to_le_bytes());
        if let Some(error) = file_sink.error() {
            return Err(io::Error::new(error.kind(), error.to_string()));
        }
        debug_assert_eq!(
            file_sink.written(),
            self.encoded_len(align_log)
                .map_err(|error| io::Error::other(error.to_string()))?,
            "流式写出与长度承诺零漂移"
        );
        let writer = file_sink.into_inner()?;
        flush_file(writer)
    }

    fn check_align(align_log: u8) -> Result<()> {
        if align_log > MAX_ALIGN_LOG {
            return Err(Error::new(
                Code::Invalid,
                format!("align_log {align_log} 超过 {MAX_ALIGN_LOG}"),
            )
            .with("weights::Writer"));
        }
        Ok(())
    }
}

/// 描述符：段级 CRC 覆盖实际存储字节，在这里算好写进头部。
fn write_descs(
    segs: &[Segment],
    packed: &[(Codec, Cow<'_, [u8]>)],
    checksum: bool,
    out: &mut impl Sink,
) -> Result<()> {
    for (segment, (codec, stored)) in segs.iter().zip(packed) {
        let seg_crc = if checksum { crc::crc32(stored) } else { 0 };
        segment.encode_desc(out, *codec, stored.len() as u64, seg_crc)?;
    }
    Ok(())
}

/// 数据区：每段先按对齐补零，再写实际存储字节。
fn write_data<S: Sink>(
    packed: &[(Codec, Cow<'_, [u8]>)],
    align_log: u8,
    measured: &mut Measured<'_, S>,
) {
    let align = 1usize << align_log;
    for (_, stored) in packed {
        let pos = measured.len();
        let aligned = pos.next_multiple_of(align);
        let pad = aligned - pos;
        if pad > 0 {
            measured.put(&vec![0u8; pad]);
        }
        measured.put(stored);
    }
}

/// 刷出 BufWriter、取出文件并 fsync。
fn flush_file(mut writer: io::BufWriter<std::fs::File>) -> io::Result<()> {
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(io::IntoInnerError::into_error)?;
    file.sync_all()
}

/// 同目录临时路径：放在目标目录才能保证 rename 在同一文件系统上原子。
fn temp_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(test)]
#[path = "../tests/writer_tests.rs"]
mod tests;
