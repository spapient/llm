//! 检查点文件：[`Snapshot`] 把一次训练恢复所需的事实与权重条目表编成字节，字节自带校验。
//!
//! **拥有**：`Format`、`Facts`、`Snapshot`——字节布局、完整性校验与原子落盘。**不拥有**：训练循环
//! 状态（[`crate::Checkpoint`] 是内存里的最小事实集）、参数数据（`weights::Header` 只被引用）、
//! 张量存储（`tensor`）、检查点路径（`cli::settings` 决定根目录）。**依赖**：`weights`、`tensor`、
//! `diagnostic`、`wire`（字节码闭集）。
//!
//! 四条判据都属于"错了会很贵"那一类：**字节自证**（`LLMC` 魔数 + 版本 + CRC-32 尾部，截断或改一个
//! 比特都读不出来）；**编码单射**（LEB128、偏移增量、名字与上一条共享前后缀、同布局只写标签 `0`；
//! 解码器重算一遍并拒掉任何"还能省却没这么写"的编码，于是 `decode(encode(x)) == x` 与
//! `encode(decode(b)) == b` 同时成立）；**长度先算准**（`encoded_len` 与 `encode` 共用一个编码器，
//! 写盘时边编码边写，整份检查点从不进内存）；**恢复两道前置校验**（[`Snapshot::verify`] 只验字节，
//! [`Snapshot::check_weights`] 再确认条目表与手头的权重文件逐条相同——形状对、内容错位不在读取时
//! 炸，而是训练若干步之后才表现成 loss 不降）。
//!
//! # 例子
//!
//! ```
//! use llm_train::snapshot::{Facts, Snapshot};
//! use tensor::{DType, Shape};
//! use weights::{Entry, Header};
//!
//! let header = Header::new(
//!     vec![Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64)],
//!     64,
//! )
//! .expect("头部");
//! let facts = Facts::new(7, 2, 2, 3).expect("恢复事实");
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
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};
use weights::{Entry, Header};
use wire::leb::{Cursor as WireCursor, write};
use wire::sink::{Count, FileSink, Sealed, Sink, SliceSink};

/// 解码失败的上下文：错误只由这一层产生，排查时不必再猜是哪一段。
const CONTEXT: &str = "train::snapshot::Snapshot::decode";

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
    pub const VERSION: u16 = 3;

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

/// 取一段字节；不足时按起始偏移报截断。
fn need(bytes: &[u8], span: Range<usize>, offset: usize) -> Result<&[u8]> {
    match bytes.get(span) {
        Some(slice) => Ok(slice),
        None => Err(truncated(offset)),
    }
}

/// 一次训练恢复后必须逐项相同的事实：步号、并行度、层数与批行数（整体比较才是判据）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    step: u64,
    world: usize,
    layers: usize,
    batch_rows: usize,
}

impl Facts {
    /// 构造：并行度、层数与批行数必须非零（步号可以为 0）；任一为零时返回 [`Code::Invalid`]。
    pub fn new(step: u64, world: usize, layers: usize, batch_rows: usize) -> Result<Self> {
        let zeros = [("并行度", world), ("层数", layers), ("批行数", batch_rows)];
        for (name, value) in zeros {
            if value == 0 {
                let message = format!("检查点的{name}不能为 0");
                return Err(Error::new(Code::Invalid, message).with("train::snapshot::Facts::new"));
            }
        }
        Ok(Self {
            step,
            world,
            layers,
            batch_rows,
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
}

/// 一次落盘的快照：恢复事实 + 权重条目表。
///
/// 字节布局（全部小端）：
///
/// ```text
/// LLMC | 版本 u16
/// 步号 | 并行度 | 层数 | 批行数 | 权重数据段起点 | 条目数            （LEB128）
/// 每条目：共享前缀 | 共享后缀 | 中段长度 | 中段
///         | 布局 | [类型 u8 | 秩 | 各维度…] | 偏移增量              （LEB128 + 字节）
/// CRC-32 u32                                                         （覆盖以上全部）
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    facts: Facts,
    weights: Header,
}

impl Snapshot {
    /// 由事实与权重条目表构造（两个入参已各自自证合法，这一层没有第二条判据可加）。
    #[must_use]
    pub fn new(facts: Facts, weights: Header) -> Self {
        Self { facts, weights }
    }

    /// 恢复事实（四个数整体比较）。
    pub fn facts(&self) -> Facts {
        self.facts
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
        let (facts, weights) = body(&bytes[..bytes.len() - Format::FOOTER_BYTES])?;
        Ok(Self::new(facts, weights))
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
    }
}

/// 正文解析：六个头部整数、条目表，最后交给 [`Header`] 与 [`Facts`] 自证。
fn body(payload: &[u8]) -> Result<(Facts, Header)> {
    let mut cursor = Cursor::new(payload, Format::HEADER_BYTES)?;
    let (facts, data_start, count) = head(&mut cursor)?;
    let entries = read_entries(&mut cursor, count)?;
    let weights = Header::new(entries, data_start).map_err(|error| error.with(CONTEXT))?;
    Ok((facts, weights))
}

/// 正文开头的六个整数：恢复事实加上权重数据段起点与条目数。
fn head(cursor: &mut Cursor<'_>) -> Result<(Facts, usize, usize)> {
    let facts = facts(cursor)?;
    let data_start = cursor.count("权重数据段起点")?;
    let count = cursor.count("权重条目数")?;
    Ok((facts, data_start, count))
}

/// 四个恢复事实，顺序即格式顺序。
fn facts(cursor: &mut Cursor<'_>) -> Result<Facts> {
    let step = cursor.varint()?;
    let world = cursor.count("并行度")?;
    let layers = cursor.count("层数")?;
    let batch_rows = cursor.count("批行数")?;
    Facts::new(step, world, layers, batch_rows).map_err(|error| error.with(CONTEXT))
}

/// 条目表：先按最小体积对账（伪造的计数撑不大内存），再逐条重建，最后确认没留尾巴。
fn read_entries(cursor: &mut Cursor<'_>, count: usize) -> Result<Vec<Entry>> {
    if count > cursor.remaining() / MIN_ENTRY {
        return Err(invalid(format!(
            "检查点声明了 {count} 个权重条目，但剩余 {} 字节最多容纳 {} 个",
            cursor.remaining(),
            cursor.remaining() / MIN_ENTRY
        )));
    }
    let mut entries = Vec::with_capacity(count.min(PREALLOC));
    let mut rebuild = Vec::new();
    for index in 0..count {
        let entry = entry(cursor, index, entries.last(), &mut rebuild)?;
        entries.push(entry);
    }
    if cursor.remaining() > 0 {
        return Err(invalid(format!(
            "检查点尾部多出 {} 字节",
            cursor.remaining()
        )));
    }
    Ok(entries)
}

/// 一个条目：名字、布局与偏移；上一条同时提供名字共享与布局重复的基准。
fn entry(
    cursor: &mut Cursor<'_>,
    index: usize,
    previous: Option<&Entry>,
    rebuild: &mut Vec<u8>,
) -> Result<Entry> {
    let previous_name = previous.map_or(&[][..], |entry| entry.name().as_bytes());
    let name = rebuild_name(cursor, index, previous_name, rebuild)?;
    let (dtype, dims) = layout(cursor, index, previous)?;
    let offset = offset(cursor, index, previous)?;
    Ok(Entry::new(name, dtype, Shape::new(dims)).at(offset))
}

/// 按上一条名字拼回整串，并拒掉"共享不是最长"的写法（否则同一份快照有多种字节形态）。
fn rebuild_name<'b>(
    cursor: &mut Cursor<'_>,
    index: usize,
    previous: &[u8],
    out: &'b mut Vec<u8>,
) -> Result<&'b str> {
    let (front, back, len) = name_lengths(cursor, index, previous.len())?;
    let middle = cursor.take(len).map_err(context(index))?;
    out.clear();
    out.extend_from_slice(&previous[..front]);
    out.extend_from_slice(middle);
    out.extend_from_slice(&previous[previous.len() - back..]);
    let want_front = common_prefix(previous, out);
    let want_back = common_suffix(&previous[want_front..], &out[want_front..]);
    if (front, back) != (want_front, want_back) {
        return Err(context(index)(invalid(format!(
            "条目名的共享不是最长写法（写了前 {front} 后 {back}，\
             应为前 {want_front} 后 {want_back}）"
        ))));
    }
    std::str::from_utf8(out)
        .map_err(|error| context(index)(invalid(format!("条目名不是 UTF-8：{error}"))))
}

/// 共享长度得分得开：前后两段不能重叠，也不能超过上一条名字。
fn fits_share(shared: usize, front: usize, back: usize, index: usize) -> Result<()> {
    if front > shared || back > shared - front {
        let message = format!("上一条目名只有 {shared} 字节，却要共享前 {front} 后 {back} 字节");
        return Err(context(index)(invalid(message)));
    }
    Ok(())
}

/// 名字的三段长度：共享前后缀必须分得开，中段长度随后读出。
fn name_lengths(
    cursor: &mut Cursor<'_>,
    index: usize,
    shared: usize,
) -> Result<(usize, usize, usize)> {
    let front = cursor.count("名字的共享前缀").map_err(context(index))?;
    let back = cursor.count("名字的共享后缀").map_err(context(index))?;
    fits_share(shared, front, back, index)?;
    let len = cursor.count("条目名中段长度").map_err(context(index))?;
    Ok((front, back, len))
}

/// 布局标签：`0` 表示与上一条同型同形，否则是 `秩 + 1` 加类型与各维度。
fn layout(
    cursor: &mut Cursor<'_>,
    index: usize,
    previous: Option<&Entry>,
) -> Result<(DType, Vec<usize>)> {
    let label = cursor.count("布局标签").map_err(context(index))?;
    if label == 0 {
        return repeated(previous, index);
    }
    fresh(cursor, index, label - 1, previous)
}

/// 与上一条同型同形；第一个条目没有上一条。
fn repeated(previous: Option<&Entry>, index: usize) -> Result<(DType, Vec<usize>)> {
    let Some(previous) = previous else {
        return Err(context(index)(invalid(
            "第一个条目不能与上一条共享布局（还没有上一条）",
        )));
    };
    Ok((previous.dtype(), previous.shape().dims().to_vec()))
}

/// 重写布局：类型标签与各维度。
fn fresh(
    cursor: &mut Cursor<'_>,
    index: usize,
    rank: usize,
    previous: Option<&Entry>,
) -> Result<(DType, Vec<usize>)> {
    let dtype = dtype(cursor.u8().map_err(context(index))?)?;
    let dims = dims(cursor, index, rank)?;
    canonical(previous, dtype, &dims, index).map(|()| (dtype, dims))
}

/// 与上一条同型同形却重写了布局是非规范写法：同一份快照不能既有"重复"又有"重写"。
fn canonical(previous: Option<&Entry>, dtype: DType, dims: &[usize], index: usize) -> Result<()> {
    if let Some(previous) = previous
        && previous.dtype() == dtype
        && previous.shape().dims() == dims
    {
        return Err(context(index)(invalid(
            "布局与上一条相同却没有写成重复（标签 0）",
        )));
    }
    Ok(())
}

/// 形状的各维度：预分到 [`PREALLOC`] 为止，伪造的秩也只能骗到一块固定大小的内存。
fn dims(cursor: &mut Cursor<'_>, index: usize, rank: usize) -> Result<Vec<usize>> {
    let mut dims = Vec::with_capacity(rank.min(PREALLOC));
    for _ in 0..rank {
        dims.push(cursor.count("维度").map_err(context(index))?);
    }
    Ok(dims)
}

/// 偏移按增量存：上一条的偏移加增量，溢出就是坏数据。
fn offset(cursor: &mut Cursor<'_>, index: usize, previous: Option<&Entry>) -> Result<usize> {
    let previous = previous.map_or(0, Entry::offset);
    let delta = cursor.count("偏移增量").map_err(context(index))?;
    let overflow = || context(index)(invalid(format!("偏移 {previous} + {delta} 溢出")));
    previous.checked_add(delta).ok_or_else(overflow)
}

/// 写一个 LEB128 变长整数；算术与规范性判据归 `wire`，这里只保留快照侧的调用名。
#[inline]
fn varint(sink: &mut impl Sink, value: u64) {
    write(sink, value);
}

/// 逐字节读取器：在 `wire::Cursor` 之上保留快照侧的字段上下文与错误词汇。
struct Cursor<'a> {
    inner: WireCursor<'a>,
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Result<Self> {
        let mut inner = WireCursor::new(bytes);
        inner.take(offset).map_err(|_| truncated(offset))?;
        Ok(Self { inner, offset })
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let slice = self
            .inner
            .take(len)
            .map_err(|_| truncated(self.inner.pos()))?;
        self.offset = self.inner.pos();
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        let byte = self
            .inner
            .read_u8()
            .map_err(|_| truncated(self.inner.pos()))?;
        self.offset = self.inner.pos();
        Ok(byte)
    }

    /// 读一个 LEB128 变长整数；非规范写法（如 `0x87 0x00`）或超出 `u64` 宽度都拒绝。
    fn varint(&mut self) -> Result<u64> {
        let value = self.inner.read::<u64>().map_err(|error| {
            if error.message().starts_with("字节不足") {
                truncated(self.inner.pos())
            } else {
                invalid("检查点里的整数非规范写法（含前导零字节）或超出 u64 上限")
            }
        })?;
        self.offset = self.inner.pos();
        Ok(value)
    }

    /// 读一个不能超过 `usize` 的计数；`what` 是字段名，用来把错误指到具体位置。
    fn count(&mut self, what: &str) -> Result<usize> {
        let value = self.varint()?;
        usize::try_from(value)
            .map_err(|_| invalid(format!("检查点的{what} {value} 超出 usize 上限")))
    }

    fn remaining(&self) -> usize {
        self.inner.remaining().len()
    }
}

/// 条目错误的上下文：一个几千条目的检查点里，"第 3 个条目"比"某个偏移"有用得多。
fn context(index: usize) -> impl Fn(Error) -> Error {
    move |error| error.with(format!("第 {index} 个权重条目"))
}

/// 两条名字共有的最大字节前缀（按字节比：共享长度最终要能直接切出字节区间）。
fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut at = 0;
    while at < limit && left[at] == right[at] {
        at += 1;
    }
    at
}

/// 两条名字共有的最大字节后缀；调用方传进来的总是已经被前缀吃掉的那两段尾巴。
fn common_suffix(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut at = 0;
    while at < limit && left[left.len() - 1 - at] == right[right.len() - 1 - at] {
        at += 1;
    }
    at
}

/// 快照格式的元素类型标签：闭集 5 种（`tag` 与 `dtype` 共用同一张表）。
const TYPES: [(DType, u8); 5] = [
    (DType::F32, 0),
    (DType::F16, 1),
    (DType::BF16, 2),
    (DType::I8, 3),
    (DType::U4, 4),
];

/// 元素类型在快照字节里的标签：闭集外的类型显式报错。
fn tag(dtype: DType) -> Result<u8> {
    TYPES
        .iter()
        .find(|(known, _)| *known == dtype)
        .map(|(_, byte)| *byte)
        .ok_or_else(|| invalid(format!("快照格式不支持 {dtype:?} 元素类型")))
}

/// 标签 → 元素类型。
fn dtype(tag: u8) -> Result<DType> {
    TYPES
        .iter()
        .find(|(_, byte)| *byte == tag)
        .map(|(known, _)| *known)
        .ok_or_else(|| invalid(format!("未知的元素类型标签 {tag}")))
}

/// 第 `index` 个条目的第一处不同；两者一致时为 `None`。
fn difference(index: usize, want: &Entry, have: &Entry) -> Option<String> {
    let name = want.name();
    if name != have.name() {
        let other = have.name();
        return Some(format!(
            "第 {index} 个条目：检查点里是 `{name}`，权重文件里是 `{other}`"
        ));
    }
    if want.dtype() != have.dtype() {
        let (a, b) = (want.dtype(), have.dtype());
        return Some(format!(
            "条目 `{name}` 的元素类型：检查点里是 {a:?}，权重文件里是 {b:?}"
        ));
    }
    if want.shape() != have.shape() {
        let (a, b) = (want.shape().dims(), have.shape().dims());
        return Some(format!(
            "条目 `{name}` 的形状：检查点里是 {a:?}，权重文件里是 {b:?}"
        ));
    }
    if want.offset() != have.offset() {
        let (a, b) = (want.offset(), have.offset());
        return Some(format!(
            "条目 `{name}` 的偏移：检查点里是 {a}，权重文件里是 {b}"
        ));
    }
    None
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

fn truncated(offset: usize) -> Error {
    Error::new(Code::Invalid, format!("检查点在偏移 {offset} 处截断")).with(CONTEXT)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(CONTEXT)
}

/// 检查点与手头的权重表不一致：只由 [`Snapshot::check_weights`] 产生。
fn mismatch(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("train::snapshot::Snapshot::check_weights")
}

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod tests;
