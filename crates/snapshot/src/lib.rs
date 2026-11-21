//! 检查点字节格式：[`Snapshot`] 把一次恢复所需的事实与权重条目表编成字节，字节自带校验。
//!
//! **拥有**：`Format`、`Facts`、`Snapshot`——字节布局、完整性校验与原子落盘。**不拥有**：训练循环
//! 状态（`train::Checkpoint` 是内存里的最小事实集，引用本成员而不是被本成员引用）、参数数据
//! （`weights::Header` 只被引用）、张量存储（`tensor`）、检查点路径（`cli` 决定根目录）。
//!
//! **依赖**：`diagnostic`、`tensor`、`weights`、`wire`——四者都在本层之下，本成员不依赖任何训练侧
//! 类型，因此可以脱离 `train` 单独替换实现（写出同样的字节即可）。
//!
//! 四条判据都属于"错了会很贵"那一类：**字节自证**（`LLMC` 魔数 + 版本 + CRC-32 尾部，截断或改一个
//! 比特都读不出来）；**编码单射**（LEB128、偏移增量、名字与上一条共享前后缀、同布局只写标签 `0`；
//! 解码器重算一遍并拒掉任何"还能省却没这么写"的编码，于是 `decode(encode(x)) == x` 与
//! `encode(decode(b)) == b` 同时成立）；**长度先算准**（[`Snapshot::encoded_len`] 与
//! [`Snapshot::encode`] 共用一个编码器，写盘时边编码边写，整份检查点从不进内存）；**恢复两道前置校验**
//! （[`Snapshot::verify`] 只验字节，[`Snapshot::check_weights`] 再确认条目表与手头的权重文件逐条相同）。
//!
//! 职责轴拆分：`cursor` 读取游标与错误词汇，`entry` 权重条目表编解码，`state` 矩状态、累积
//! 梯度段与权重绑定段；字节格式仍是本成员单源。
//!
//! # 例子
//!
//! ```
//! use snapshot::{Facts, Snapshot};
//! use tensor::{DType, Shape};
//! use weights::{Entry, Header};
//!
//! let header = Header::new(
//!     vec![Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64)],
//!     64,
//! )
//! .expect("头部");
//! let facts = Facts::new(7, 2, 2, 3, 3).expect("恢复事实");
//! let snapshot = Snapshot::new(facts, header);
//!
//! let bytes = snapshot.encode().expect("编码");
//! assert_eq!(bytes.len(), snapshot.encoded_len().expect("长度"));
//! Snapshot::verify(&bytes).expect("字节自证");
//!
//! // 恢复前的第二道关：这份检查点描述的条目表必须与手头的权重表逐条相同。
//! let restored = Snapshot::decode(&bytes).expect("解码");
//! assert_eq!(restored.facts(), facts);
//! restored.check_weights(restored.weights()).expect("同一份权重表");
//! ```

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use diagnostic::{Code, Error, Result};
use tensor::DType;
use weights::Header;
use wire::leb::write;
use wire::sink::{Count, FileSink, Sealed, Sink, SliceSink};

mod cursor;
mod entry;
mod state;

pub use state::{GradAccum, Moments, SlotMoments, Ties};

use cursor::{Cursor, invalid, mismatch, need, truncated};
use entry::{common_prefix, common_suffix, difference, read_entries, tag};
use state::{
    read_grad_accum, read_moments, read_ties, write_grad_accum, write_moments, write_ties,
};

/// 一个条目最少占 5 字节：解码据此把声明的条目数与剩余字节对账，伪造的计数撑不大内存。
const MIN_ENTRY: usize = 5;

/// 写盘时的缓冲：够把几千条目的头部一次交给内核，又不至于为一个小检查点占太多内存。
const WRITE_BUFFER: usize = 8 * 1024;

/// 条目表与维度的预分配上限：更大的表按需增长，于是伪造的计数最多多要一块固定大小的内存。
const PREALLOC: usize = 1024;

/// 检查点文件头：4 字节魔数 + 2 字节小端版本号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format;

impl Format {
    /// 魔数 `LLMC`。
    pub const MAGIC: [u8; 4] = *b"LLMC";

    /// 当前格式版本：字段集、编码顺序或校验方式变化时必须递增；版本之间不兼容，只支持当前版本。
    pub const VERSION: u16 = 5;

    /// 文件头字节数（魔数 + 版本）。
    pub const HEADER_BYTES: usize = Self::MAGIC.len() + 2;

    /// 尾部校验和字节数（小端 `u32`）。
    pub const FOOTER_BYTES: usize = 4;

    /// 校验文件头：魔数一致且版本相同，不符时返回 [`Code::Invalid`]。
    pub fn check(bytes: &[u8]) -> Result<()> {
        let magic = need(bytes, 0..Self::MAGIC.len(), 0)?;
        let raw = need(
            bytes,
            Self::MAGIC.len()..Self::HEADER_BYTES,
            Self::MAGIC.len(),
        )?;
        if magic != Self::MAGIC.as_slice() {
            let message = format!("检查点魔数不是 `LLMC`：{magic:?}");
            return Err(Error::new(Code::Invalid, message).with("train::snapshot::Format::check"));
        }
        let version = u16::from_le_bytes([raw[0], raw[1]]);
        if version == Self::VERSION {
            return Ok(());
        }
        Err(version_error(version))
    }
}

/// 版本不符时说清是哪一边旧：读端旧要重存，写端旧要升级。
fn version_error(version: u16) -> Error {
    let current = Format::VERSION;
    let message = if version > current {
        format!("检查点版本 {version} 比当前版本 {current} 新：请用写它的那个版本读取")
    } else {
        format!(
            "检查点版本 {version} 比当前版本 {current} 旧：本 crate 只读第 {current} 版，请重存一次"
        )
    };
    Error::new(Code::Invalid, message).with("train::snapshot::Format::check")
}

/// 一次训练恢复后必须逐项相同的事实：步号、并行度、层数与批几何（行、列整体比较才是判据）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    step: u64,
    world: usize,
    layers: usize,
    batch_rows: usize,
    batch_cols: usize,
}

impl Facts {
    /// 构造：并行度、层数与批行列必须非零（步号可以为 0）；任一为零时返回 [`Code::Invalid`]。
    pub fn new(
        step: u64,
        world: usize,
        layers: usize,
        batch_rows: usize,
        batch_cols: usize,
    ) -> Result<Self> {
        let zeros = [
            ("并行度", world),
            ("层数", layers),
            ("批行数", batch_rows),
            ("批列宽", batch_cols),
        ];
        for (name, value) in zeros {
            require_nonzero(name, value)?;
        }
        Ok(Self {
            step,
            world,
            layers,
            batch_rows,
            batch_cols,
        })
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 保存时的并行度（恢复时必须一致）。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 训练层数。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// 批的列宽。
    pub fn batch_cols(&self) -> usize {
        self.batch_cols
    }
}

/// 一项恢复事实为零：把字段名带在消息里。
fn require_nonzero(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        let message = format!("检查点的{name}不能为 0");
        return Err(Error::new(Code::Invalid, message).with("train::snapshot::Facts::new"));
    }
    Ok(())
}

/// 一次落盘的快照：恢复事实 + 权重条目表 + 优化器矩状态 + 累积窗口梯度 + 权重绑定表。
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    facts: Facts,
    weights: Header,
    moments: Moments,
    accum: GradAccum,
    ties: Ties,
}

impl Snapshot {
    /// 由事实与权重条目表构造（矩状态、累积梯度与绑定表为空）。
    #[must_use]
    pub fn new(facts: Facts, weights: Header) -> Self {
        Self::new_with_state(facts, weights, Moments::empty(), GradAccum::empty())
    }

    /// 由事实、权重条目表与优化器矩状态构造（累积梯度与绑定表为空）。
    #[must_use]
    pub fn new_with_moments(facts: Facts, weights: Header, moments: Moments) -> Self {
        Self::new_with_state(facts, weights, moments, GradAccum::empty())
    }

    /// 由事实、权重条目表、矩状态与累积梯度构造（绑定表为空）。
    #[must_use]
    pub fn new_with_state(
        facts: Facts,
        weights: Header,
        moments: Moments,
        accum: GradAccum,
    ) -> Self {
        Self::new_with_all(facts, weights, moments, accum, Ties::empty())
    }

    /// 由事实、权重条目表、矩状态、累积梯度与权重绑定表构造。
    #[must_use]
    pub fn new_with_all(
        facts: Facts,
        weights: Header,
        moments: Moments,
        accum: GradAccum,
        ties: Ties,
    ) -> Self {
        Self {
            facts,
            weights,
            moments,
            accum,
            ties,
        }
    }

    /// 恢复事实（五个数整体比较）。
    pub fn facts(&self) -> Facts {
        self.facts
    }

    /// 优化器矩状态表。
    pub fn moments(&self) -> &Moments {
        &self.moments
    }

    /// 累积窗口内的梯度汇总。
    pub fn accum(&self) -> &GradAccum {
        &self.accum
    }

    /// 权重绑定表：源槽 → 回填槽的下标对。
    pub fn ties(&self) -> &Ties {
        &self.ties
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.facts.step()
    }

    /// 权重条目表：解码时已经重新校验过偏移与重名。
    pub fn weights(&self) -> &Header {
        &self.weights
    }

    /// 恢复前的前置校验：条目表必须与手头的权重表逐条相同，否则返回 [`Code::Invalid`]。
    pub fn check_weights(&self, weights: &Header) -> Result<()> {
        if self.weights.data_start() != weights.data_start() {
            return Err(mismatch(format!(
                "权重数据段起点：检查点里是 {}，权重文件里是 {}",
                self.weights.data_start(),
                weights.data_start()
            )));
        }
        let saved = self.weights.entries();
        let live = weights.entries();
        if saved.len() != live.len() {
            return Err(mismatch(format!(
                "检查点引用 {} 个权重条目，权重文件有 {} 个",
                saved.len(),
                live.len()
            )));
        }
        for (index, (want, have)) in saved.iter().zip(live).enumerate() {
            if let Some(message) = difference(index, want, have) {
                return Err(mismatch(message));
            }
        }
        Ok(())
    }

    /// 编码后的精确字节数（与 [`Snapshot::encode`] 走同一个编码器，不含校验和）。
    pub fn encoded_len(&self) -> Result<usize> {
        let tags = self.tags()?;
        Ok(self.counted(&tags))
    }

    /// 条目类型标签表：把"不支持的 `DType`"一次判完，编码器因此没有可失败的操作。
    fn tags(&self) -> Result<Vec<u8>> {
        self.weights
            .entries()
            .iter()
            .map(|entry| tag(entry.dtype()))
            .collect()
    }

    /// 数出整份检查点的字节数（标签已校验过）。
    fn counted(&self, tags: &[u8]) -> usize {
        let mut count = Count(0);
        self.write_body(&mut count, tags);
        count.0 + Format::FOOTER_BYTES
    }

    /// 编码成字节，容量正好等于 [`Snapshot::encoded_len`]。
    pub fn encode(&self) -> Result<Vec<u8>> {
        let tags = self.tags()?;
        let capacity = self.counted(&tags);
        let mut bytes = Vec::with_capacity(capacity);
        self.write_to(&mut bytes, &tags);
        debug_assert_eq!(bytes.len(), capacity, "长度算术与写出的字节必须一致");
        Ok(bytes)
    }

    /// 写进调用方给的缓冲区，返回写入的字节数；`out` 至少要有 [`Snapshot::encoded_len`] 字节。
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize> {
        let capacity = out.len();
        let mut sink = SliceSink::new(out);
        let tags = self.tags()?;
        self.write_to(&mut sink, &tags);
        if sink.overflow() {
            let needed = self.counted(&tags);
            let message = format!("缓冲区 {capacity} 字节，检查点需要 {needed} 字节");
            return Err(
                Error::new(Code::Invalid, message).with("train::snapshot::Snapshot::encode_into")
            );
        }
        Ok(sink.written())
    }

    /// 原子地写到一个路径：同目录临时文件 → 落盘 → 改名（失败返回 [`Code::Io`]，临时文件被清掉）。
    pub fn write(&self, path: &Path) -> Result<()> {
        let tags = self.tags()?;
        let temp = temp_path(path);
        if let Err(error) = self.write_temp(&temp, path, &tags) {
            // 半截临时文件比错误本身更麻烦：它会被下一次 `ls` 当成一个检查点。
            let _ = std::fs::remove_file(&temp);
            let message = format!("写入 {} 失败：{error}", path.display());
            return Err(Error::new(Code::Io, message).with("train::snapshot::Snapshot::write"));
        }
        // 目录 fsync 尽力而为：为一个"可能不持久"的目录项把成功改判成失败，比不 fsync 更糟。
        if let Some(parent) = path.parent()
            && let Ok(directory) = std::fs::File::open(parent)
        {
            let _ = directory.sync_all();
        }
        Ok(())
    }

    /// 写临时文件、刷盘、改名为目标路径（三段式：改名前任何失败都不影响旧检查点）。
    fn write_temp(&self, temp: &Path, path: &Path, tags: &[u8]) -> io::Result<()> {
        self.flush(temp, tags)?;
        std::fs::rename(temp, path)
    }

    /// 边编码边写并刷盘：长度承诺在改名之前对账，第一次写失败之后剩下的追加都是空操作。
    fn flush(&self, temp: &Path, tags: &[u8]) -> io::Result<()> {
        let file = io::BufWriter::with_capacity(WRITE_BUFFER, std::fs::File::create(temp)?);
        let mut sink = FileSink::new(file);
        self.write_to(&mut sink, tags);
        if let Some(error) = sink.take_error() {
            return Err(error);
        }
        debug_assert_eq!(
            sink.written(),
            self.counted(tags),
            "写下的字节数必须等于承诺的长度"
        );
        let file = sink
            .into_inner()?
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        file.sync_all()
    }

    /// 从一个路径读出并解码：路径不存在是 [`Code::Missing`]、读不出来是 [`Code::Io`]。
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| Error::from_io(path, &error))?;
        Self::decode(&bytes).map_err(|error| error.with(path.display().to_string()))
    }

    /// 只验字节：魔数、版本与 CRC-32，不解析条目表（巡检、备份、下载后自检不必付出一次解码）。
    pub fn verify(bytes: &[u8]) -> Result<()> {
        Format::check(bytes)?;
        let Some(end) = bytes.len().checked_sub(Format::FOOTER_BYTES) else {
            return Err(truncated(bytes.len()));
        };
        // 连文件头都装不下时，问题不是"校验和对不上"而是"少了一截"；把话说准。
        if end < Format::HEADER_BYTES {
            return Err(truncated(end));
        }
        let (payload, footer) = bytes.split_at(end);
        let recorded = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
        let actual = weights::crc::crc32(payload);
        if recorded != actual {
            return Err(invalid(format!(
                "检查点校验和不符（记录 {recorded:#010x}，实测 {actual:#010x}）：\
                 文件被截断或被改动"
            )));
        }
        Ok(())
    }

    /// 从字节解码：先由 [`Snapshot::verify`] 判文件头与校验和，再解析正文。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Self::verify(bytes)?;
        let (facts, weights, moments, accum, ties) =
            body(&bytes[..bytes.len() - Format::FOOTER_BYTES])?;
        Ok(Self::new_with_all(facts, weights, moments, accum, ties))
    }

    /// 编码正文与校验和：正文写进一个"密封"的接收器，写出的每个字节都自动进入 CRC。
    fn write_to(&self, sink: &mut impl Sink, tags: &[u8]) {
        let crc = {
            let mut sealed = Sealed::new(sink);
            self.write_body(&mut sealed, tags);
            sealed.checksum()
        };
        sink.put(&crc.to_le_bytes());
    }

    /// 写文件头与正文（不含校验和）。
    fn write_body(&self, sink: &mut impl Sink, tags: &[u8]) {
        let facts = self.facts;
        sink.put(Format::MAGIC.as_slice());
        sink.put(&Format::VERSION.to_le_bytes());
        varint(sink, facts.step());
        varint(sink, facts.world() as u64);
        varint(sink, facts.layers() as u64);
        varint(sink, facts.batch_rows() as u64);
        varint(sink, facts.batch_cols() as u64);
        varint(sink, self.weights.data_start() as u64);

        let entries = self.weights.entries();
        varint(sink, entries.len() as u64);
        let mut previous_name: &[u8] = &[];
        let mut previous_offset = 0usize;
        let mut previous_layout: Option<(DType, &[usize])> = None;
        for (index, entry) in entries.iter().enumerate() {
            let name = entry.name().as_bytes();
            let front = common_prefix(previous_name, name);
            // 后缀只从"两条名字都还没被前缀吃掉"的那一段里取，于是前后两段永远不会重叠。
            let back = common_suffix(&previous_name[front..], &name[front..]);
            varint(sink, front as u64);
            varint(sink, back as u64);
            varint(sink, (name.len() - front - back) as u64);
            sink.put(&name[front..name.len() - back]);
            // 布局（元素类型 + 形状）也相对上一条：同型同形只占一个字节，否则写 `秩 + 1` 加各维度。
            let dims = entry.shape().dims();
            if previous_layout == Some((entry.dtype(), dims)) {
                varint(sink, 0);
            } else {
                varint(sink, (dims.len() + 1) as u64);
                sink.put(&[tags[index]]);
                for dim in dims {
                    varint(sink, *dim as u64);
                }
                previous_layout = Some((entry.dtype(), dims));
            }
            // 增量而不是绝对值：`Header` 按偏移排序，相邻差值就是"空隙 + 上一条长度"。
            debug_assert!(entry.offset() >= previous_offset, "`Header` 必须按偏移排序");
            varint(sink, entry.offset().saturating_sub(previous_offset) as u64);
            previous_offset = entry.offset();
            previous_name = name;
        }

        write_moments(sink, &self.moments);
        write_grad_accum(sink, &self.accum);
        write_ties(sink, &self.ties);
    }
}

/// 正文解析：头部整数、条目表、矩状态段、累积梯度段与绑定段，最后交给 [`Header`] 与 [`Facts`] 自证。
fn body(payload: &[u8]) -> Result<(Facts, Header, Moments, GradAccum, Ties)> {
    let mut cursor = Cursor::new(payload, Format::HEADER_BYTES)?;
    read_sections(&mut cursor)
}

/// 依次读矩状态、累积梯度与绑定段；读完对账尾部余量。
fn read_sections(cursor: &mut Cursor<'_>) -> Result<(Facts, Header, Moments, GradAccum, Ties)> {
    let (facts, weights) = head_and_weights(cursor)?;
    let moments = read_moments(cursor)?;
    let (accum, ties) = read_accum_ties(cursor)?;
    reject_tail(cursor)?;
    Ok((facts, weights, moments, accum, ties))
}

/// 累积梯度段紧随矩状态；绑定段收尾在前。
fn read_accum_ties(cursor: &mut Cursor<'_>) -> Result<(GradAccum, Ties)> {
    let accum = tail_state(cursor)?;
    let ties = read_ties(cursor)?;
    Ok((accum, ties))
}

/// 正文开头整数与权重条目表。
fn head_and_weights(cursor: &mut Cursor<'_>) -> Result<(Facts, Header)> {
    let facts = decode_facts(cursor)?;
    let weights = read_weights_header(cursor)?;
    Ok((facts, weights))
}

/// 权重数据段起点、条目计数对账后重建条目表并装配 [`Header`]。
fn read_weights_header(cursor: &mut Cursor<'_>) -> Result<Header> {
    let data_start = cursor.count("权重数据段起点")?;
    let count = cursor.count("权重条目数")?;
    let entries = read_entries(cursor, count, MIN_ENTRY, PREALLOC)?;
    Header::new(entries, data_start).map_err(error_context)
}

/// 矩状态之后的累积梯度段；绑定段与尾部余量对账在 [`body`](self::body) 收尾。
fn tail_state(cursor: &mut Cursor<'_>) -> Result<GradAccum> {
    read_grad_accum(cursor)
}

/// 条目表之后若还有未消费字节，就是半截或多写。
fn reject_tail(cursor: &Cursor<'_>) -> Result<()> {
    let leftover = cursor.remaining();
    if leftover > 0 {
        return Err(invalid(format!("检查点尾部多出 {leftover} 字节")));
    }
    Ok(())
}

/// 五个恢复事实，顺序即格式顺序；四个形状计数共用一条读取循环。
fn decode_facts(cursor: &mut Cursor<'_>) -> Result<Facts> {
    let step = cursor.varint()?;
    let labels = ["并行度", "层数", "批行数", "批列宽"];
    let mut counts = [0usize; 4];
    for (slot, label) in counts.iter_mut().zip(labels) {
        *slot = cursor.count(label)?;
    }
    Facts::new(step, counts[0], counts[1], counts[2], counts[3]).map_err(error_context)
}

/// Header 构造失败挂上解码上下文。
fn error_context(error: Error) -> Error {
    error.with(cursor::CONTEXT)
}

/// 写一个 LEB128 变长整数；算术与规范性判据归 `wire`，这里只保留快照侧的调用名。
#[inline]
fn varint(sink: &mut impl Sink, value: u64) {
    write(sink, value);
}

/// 临时文件名：同目录 + 进程号 + 进程内序号（两个写者不会互相搬走半截文件）。
fn temp_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(test)]
#[path = "../tests/snapshot_support.rs"]
mod snapshot_support;

#[cfg(test)]
#[path = "../tests/snapshot_roundtrip_tests.rs"]
mod roundtrip_tests;

#[cfg(test)]
#[path = "../tests/snapshot_canonical_tests.rs"]
mod canonical_tests;
