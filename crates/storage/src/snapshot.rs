//! 检查点字节格式：[`Snapshot`] 把一次恢复所需的事实与条目表编成字节，字节自带校验。
//!
//! **拥有**：`Format`、`Facts`、`Snapshot` 与两条契约 [`Manifest`] / [`ManifestEntry`]——字节布局、
//! 完整性校验与原子落盘。**不拥有**：训练循环状态（`train::Checkpoint` 是内存里的最小事实集，
//! 引用本模块而不是被本模块引用）、条目表本体（`weights::Header` 实现 [`Manifest`]，本模块只认识
//! "有序条目 + 数据段起点"）、张量存储（`tensor`）、检查点路径（`cli` 决定根目录）。
//!
//! **为什么是泛型**：字节格式需要一个条目表，却不需要知道权重文件怎么校验自己。把条目表收成
//! [`Manifest`]，`storage` 就不必反向依赖 `weights`——于是 `storage → weights → storage` 的环
//! 根本不存在。上层看到的仍是同一个类型：`weights::Snapshot` 是
//! `storage::snapshot::Snapshot<weights::Header>` 的别名，消费方不必写泛型。
//!
//! 四条判据都属于"错了会很贵"那一类：**字节自证**（`LLMC` 魔数 + 版本 + CRC-32 尾部，截断或改
//! 一个比特都读不出来）；**编码单射**（LEB128、偏移增量、名字与上一条共享前后缀、同布局只写
//! 标签 `0`；解码器重算一遍并拒掉任何"还能省却没这么写"的编码，于是 `decode(encode(x)) == x`
//! 与 `encode(decode(b)) == b` 同时成立）；**长度先算准**（[`Snapshot::encoded_len`] 与
//! [`Snapshot::encode`] 共用一个编码器，写盘时边编码边写，整份检查点从不进内存）；**恢复两道
//! 前置校验**（[`Snapshot::verify`] 只验字节，[`Snapshot::check_weights`] 再确认条目表与手头的
//! 权重文件逐条相同）。
//!
//! 职责轴拆分：`format` 版本与恢复事实，`codec` 读取游标与条目表编解码，`state` 矩状态、
//! 累积梯度段与权重绑定段；字节格式仍是本模块单源。

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::codec;
pub use crate::format::{Facts, Format};
use crate::leb::write;
use crate::sink::{Count, FileSink, Sealed, Sink, SliceSink};
pub use crate::state::{GradAccum, Moments, SlotMoments, Ties};
use diagnostic::{Code, Error, Result};
use tensor::DType;

use crate::codec::{Cursor, difference, invalid, mismatch, read_entries, tag, truncated};
use crate::state::{
    read_grad_accum, read_moments, read_ties, write_grad_accum, write_moments, write_ties,
};

/// 一个条目最少占 5 字节：解码据此把声明的条目数与剩余字节对账，伪造的计数撑不大内存。
const MIN_ENTRY: usize = 5;

/// 写盘时的缓冲：够把几千条目的头部一次交给内核，又不至于为一个小检查点占太多内存。
pub(crate) const WRITE_BUF: usize = 8 * 1024;

/// 条目表与维度的预分配上限：更大的表按需增长，于是伪造的计数最多多要一块固定大小的内存。
pub(crate) const PREALLOC: usize = 1024;

/// 快照里的一个条目：名字、元素类型、形状维度与它在数据段里的字节偏移。
///
/// 进字节的只有这四项，因此实现者不必是 `weights::Entry`——测试夹具与将来的第二种权重容器
/// 都能直接当快照的条目用，而字节格式一处都不用改。
pub trait ManifestEntry {
    /// 条目名（权重文件里的键）。
    fn name(&self) -> &str;

    /// 元素类型；闭集外的类型在编码期报错。
    fn dtype(&self) -> DType;

    /// 形状的各维度（秩就是它的长度）。
    fn dims(&self) -> &[usize];

    /// 在数据段里的起始偏移。
    fn offset(&self) -> usize;

    /// 由字节里的四个字段重建一个条目（解码路径）。
    fn from_parts(name: String, dtype: DType, dims: Vec<usize>, offset: usize) -> Self;
}

/// 快照引用的条目表：有序条目 + 数据段起点。
///
/// [`Manifest::build`] 是解码的最后一步：从字节里读出来的条目表必须再过一遍实现者自己的判据
/// （名字唯一、区间不重叠、起点不早于数据段），否则"校验和正确"会被误当成"条目表合法"。
pub trait Manifest {
    /// 条目类型。
    type Entry: ManifestEntry;

    /// 全部条目，按偏移升序——增量编码依赖这个顺序。
    fn entries(&self) -> &[Self::Entry];

    /// 数据段起点（条目偏移的基准）。
    fn data_start(&self) -> usize;

    /// 从条目表重建，并重跑一遍本类型的判据。
    fn build(entries: Vec<Self::Entry>, data_start: usize) -> Result<Self>
    where
        Self: Sized;
}

/// 一次落盘的快照：恢复事实 + 条目表 + 优化器矩状态 + 累积窗口梯度 + 权重绑定表。
///
/// `C` 是条目表（[`Manifest`]）；`weights::Snapshot` 是本类型以 `Header` 为实参的别名。
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot<C: Manifest> {
    facts: Facts,
    weights: C,
    moments: Moments,
    accum: GradAccum,
    ties: Ties,
}

impl<C: Manifest> Snapshot<C> {
    /// 由事实与条目表构造（矩状态、累积梯度与绑定表为空）。
    #[must_use]
    pub fn new(facts: Facts, weights: C) -> Self {
        Self::new_with_state(facts, weights, Moments::empty(), GradAccum::empty())
    }

    /// 由事实、条目表与优化器矩状态构造（累积梯度与绑定表为空）。
    #[must_use]
    pub fn new_with_moments(facts: Facts, weights: C, moments: Moments) -> Self {
        Self::new_with_state(facts, weights, moments, GradAccum::empty())
    }

    /// 由事实、条目表、矩状态与累积梯度构造（绑定表为空）。
    #[must_use]
    pub fn new_with_state(facts: Facts, weights: C, moments: Moments, accum: GradAccum) -> Self {
        Self::new_with_all(facts, weights, moments, accum, Ties::empty())
    }

    /// 由事实、条目表、矩状态、累积梯度与权重绑定表构造。
    #[must_use]
    pub fn new_with_all(
        facts: Facts,
        weights: C,
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

    /// 绑定训练配置指纹；默认构造器的 `0` 表示未声明。
    #[must_use]
    pub fn with_config_fingerprint(mut self, fingerprint: u64) -> Self {
        self.facts = self.facts.with_config_fingerprint(fingerprint);
        self
    }

    /// 训练配置指纹。
    pub fn config_fingerprint(&self) -> u64 {
        self.facts.config_fingerprint()
    }

    /// 恢复事实（五个数与配置指纹整体比较）。
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

    /// 条目表：解码时已经重新校验过偏移并交给 [`Manifest::build`] 复判。
    pub fn weights(&self) -> &C {
        &self.weights
    }

    /// 恢复前的前置校验：条目表必须与手头的权重表逐条相同，否则返回 [`Code::Invalid`]。
    pub fn check_weights(&self, weights: &C) -> Result<()> {
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
        for (idx, (want, have)) in saved.iter().zip(live).enumerate() {
            if let Some(message) = difference(idx, want, have) {
                return Err(mismatch(message));
            }
        }
        Ok(())
    }

    /// 编码后的精确字节数（与 [`Snapshot::encode`] 走同一个编码器，不含校验和）。
    pub fn encoded_len(&self) -> Result<usize> {
        let tags = tags::<C>(&self.weights)?;
        Ok(self.counted(&tags))
    }

    /// 数出整份检查点的字节数（标签已校验过）。
    fn counted(&self, tags: &[u8]) -> usize {
        let mut count = Count(0);
        self.write_body(&mut count, tags);
        count.0 + Format::FOOTER_BYTES
    }

    /// 编码成字节，容量正好等于 [`Snapshot::encoded_len`]。
    pub fn encode(&self) -> Result<Vec<u8>> {
        let tags = tags::<C>(&self.weights)?;
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
        let tags = tags::<C>(&self.weights)?;
        self.write_to(&mut sink, &tags);
        if sink.overflow() {
            let needed = self.counted(&tags);
            let message = format!("缓冲区 {capacity} 字节，检查点需要 {needed} 字节");
            let ctx = "storage::snapshot::Snapshot::encode_into";
            return Err(Error::new(Code::Invalid, message).with(ctx));
        }
        Ok(sink.written())
    }

    /// 原子地写到一个路径：同目录临时文件 → 落盘 → 改名（失败返回 [`Code::Io`]，临时文件被清掉）。
    pub fn write(&self, path: &Path) -> Result<()> {
        let tags = tags::<C>(&self.weights)?;
        let temp = temp_path(path);
        if let Err(error) = self.write_temp(&temp, path, &tags) {
            // 半截临时文件比错误本身更麻烦：它会被下一次 `ls` 当成一个检查点。
            let _ = std::fs::remove_file(&temp);
            let message = format!("写入 {} 失败：{error}", path.display());
            let ctx = "storage::snapshot::Snapshot::write";
            return Err(Error::new(Code::Io, message).with(ctx));
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
        let file = io::BufWriter::with_capacity(WRITE_BUF, std::fs::File::create(temp)?);
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
        Format::check_bytes(bytes)?;
        let Some(end) = bytes.len().checked_sub(Format::FOOTER_BYTES) else {
            return Err(truncated(bytes.len()));
        };
        // 连文件头都装不下时，问题不是"校验和对不上"而是"少了一截"；把话说准。
        if end < Format::HEADER_BYTES {
            return Err(truncated(end));
        }
        let (payload, footer) = bytes.split_at(end);
        let recorded = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
        let actual = crate::crc::crc32(payload);
        if recorded == actual {
            return Ok(());
        }
        Err(invalid(format!(
            "检查点校验和不符（记录 {recorded:#010x}，实测 {actual:#010x}）：文件被截断或被改动"
        )))
    }

    /// 从字节解码：先由 [`Snapshot::verify`] 判文件头与校验和，再解析正文。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Self::verify(bytes)?;
        let (facts, weights, moments, accum, ties) =
            body::<C>(&bytes[..bytes.len() - Format::FOOTER_BYTES])?;
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
        sink.put(&facts.config_fingerprint().to_le_bytes());
        varint(sink, self.weights.data_start() as u64);

        let entries = self.weights.entries();
        varint(sink, entries.len() as u64);
        let mut previous_name: &[u8] = &[];
        let mut previous_offset = 0usize;
        let mut previous_layout: Option<(DType, &[usize])> = None;
        for (idx, entry) in entries.iter().enumerate() {
            let name = entry.name().as_bytes();
            let front = codec::common_prefix(previous_name, name);
            // 后缀只从"两条名字都还没被前缀吃掉"的那一段里取，于是前后两段永远不会重叠。
            let back = codec::common_suffix(&previous_name[front..], &name[front..]);
            varint(sink, front as u64);
            varint(sink, back as u64);
            varint(sink, (name.len() - front - back) as u64);
            sink.put(&name[front..name.len() - back]);
            // 布局（元素类型 + 形状）也相对上一条：同型同形只占一个字节，否则写 `秩 + 1` 加各维度。
            let dims = entry.dims();
            if previous_layout == Some((entry.dtype(), dims)) {
                varint(sink, 0);
            } else {
                varint(sink, (dims.len() + 1) as u64);
                sink.put(&[tags[idx]]);
                for dim in dims {
                    varint(sink, *dim as u64);
                }
                previous_layout = Some((entry.dtype(), dims));
            }
            // 增量而不是绝对值：`Header` 按偏移排序，相邻差值就是"空隙 + 上一条长度"。
            debug_assert!(entry.offset() >= previous_offset, "条目表必须按偏移排序");
            varint(sink, entry.offset().saturating_sub(previous_offset) as u64);
            previous_offset = entry.offset();
            previous_name = name;
        }

        write_moments(sink, &self.moments);
        write_grad_accum(sink, &self.accum);
        write_ties(sink, &self.ties);
    }
}

/// 条目类型标签表：把"不支持的 `DType`"一次判完，编码器因此没有可失败的操作。
fn tags<C: Manifest>(weights: &C) -> Result<Vec<u8>> {
    weights
        .entries()
        .iter()
        .map(|entry| tag(entry.dtype()))
        .collect()
}

/// 正文解析：按格式顺序读事实、条目表，再读状态三段——顺序本身就是格式。
fn body<C: Manifest>(payload: &[u8]) -> Result<(Facts, C, Moments, GradAccum, Ties)> {
    let mut cursor = Cursor::new(payload, Format::HEADER_BYTES)?;
    let facts = decode_facts(&mut cursor)?;
    let weights = read_weights_header::<C>(&mut cursor)?;
    let (moments, accum, ties) = read_state(&mut cursor)?;
    Ok((facts, weights, moments, accum, ties))
}

/// 状态三段（矩 / 累积梯度 / 绑定）是正文的最后一段（`closure/state`，恢复时读）：读完后
/// 若还有未消费字节，就是半截或多写，故尾部归零判定跟着它，而不是另起一个只看余量的函数。
fn read_state(cursor: &mut Cursor<'_>) -> Result<(Moments, GradAccum, Ties)> {
    let moments = read_moments(cursor)?;
    let accum = read_grad_accum(cursor)?;
    let ties = read_ties(cursor)?;
    let leftover = cursor.remaining();
    if leftover > 0 {
        return Err(invalid(format!("检查点尾部多出 {leftover} 字节")));
    }
    Ok((moments, accum, ties))
}

/// 权重数据段起点、条目计数对账后重建条目表并交给 [`Manifest::build`]。
fn read_weights_header<C: Manifest>(cursor: &mut Cursor<'_>) -> Result<C> {
    let data_start = cursor.count("权重数据段起点")?;
    let count = cursor.count("权重条目数")?;
    let entries = read_entries(cursor, count, MIN_ENTRY, PREALLOC)?;
    C::build(entries, data_start).map_err(error_ctx)
}

/// 五个恢复事实，顺序即格式顺序；四个形状计数共用一条读取循环。
fn decode_facts(cursor: &mut Cursor<'_>) -> Result<Facts> {
    let step = cursor.varint()?;
    let labels = ["并行度", "层数", "批行数", "批列宽"];
    let mut counts = [0usize; 4];
    for (slot, label) in counts.iter_mut().zip(labels) {
        *slot = cursor.count(label)?;
    }
    let fingerprint = u64::from_le_bytes(cursor.take(8)?.try_into().map_err(|_| {
        codec::invalid("配置指纹必须是 8 字节小端整数")
    })?);
    Facts::new(step, counts[0], counts[1], counts[2], counts[3])
        .map(|facts| facts.with_config_fingerprint(fingerprint))
        .map_err(error_ctx)
}

/// 条目表构造失败挂上解码上下文。
fn error_ctx(error: Error) -> Error {
    error.with(codec::CTX)
}

/// 写一个 LEB128 变长整数；算术与规范性判据归 `wire`，这里只保留快照侧的调用名。
#[inline]
pub(crate) fn varint(sink: &mut impl Sink, value: u64) {
    write(sink, value);
}

/// 临时文件名：同目录 + 进程号 + 进程内序号（两个写者不会互相搬走半截文件）。
fn temp_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .map_or_default(|name| name.to_string_lossy().into_owned());
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
