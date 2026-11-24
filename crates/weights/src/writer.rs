//! 写端：收集段，按对齐布局一次产出 canonical 字节，同一份编码可写内存或流式落盘。

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use diagnostic::{Code, Error, Result};

use crate::axes::AxisSet;

use crate::Codec;
use crate::Wire;
use crate::crc;
use crate::proof::Digest;
use crate::rans;
use crate::seg::Segment;
use storage::leb;
use storage::sink::{Count, FileSink, Measured, Sealed, Sink, SliceSink};

/// 魔数 `.u\0\0`：小端字节 `[0x2E, 0x75, 0x00, 0x00]`，第三、四字节 NUL 收尾（ADR 0045）。
pub const MAGIC: u32 = 0x0000_752E;

/// 当前格式版本（自有 `.u` 容器 v2）。
pub const VERSION: u16 = 2;

/// `flags` 位：全局 CRC 存在。
pub const GLOBAL_CRC: u32 = 1 << 0;

/// `flags` 位：段级 CRC 存在。
pub const SEG_CRC: u32 = 1 << 1;

/// 最大对齐指数（2^16 = 64 KiB）。
pub const MAX_ALIGN_LOG: u8 = 16;

const WRITE_BUF: usize = 8 * 1024;

/// 写端选项：两级 CRC、自适应压缩、五轴候选与编码并行度，默认与既有行为一致。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// 两级 CRC（全局 + 段级）。
    pub checksum: bool,
    /// 自适应 rANS / 五轴压缩。
    pub compress: bool,
    /// 五轴候选集合；只在 `compress` 开启时生效。
    pub axes: AxisSet,
    /// 段编码工作线程数；`0` 与 `1` 都是串行。输出与线程数无关。
    pub workers: usize,
}

impl Default for Config {
    /// 默认开启两级 CRC 与全五轴自适应压缩，串行编码。
    fn default() -> Self {
        Self {
            checksum: true,
            compress: true,
            axes: AxisSet::ALL,
            workers: 1,
        }
    }
}

impl Config {
    /// 设置五轴候选集合。
    #[must_use]
    pub const fn with_axes(mut self, axes: AxisSet) -> Self {
        self.axes = axes;
        self
    }

    /// 设置编码并行度；`0` 与 `1` 都是串行。
    #[must_use]
    pub const fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }
}

/// 权重写者。
#[derive(Debug)]
pub struct Writer {
    segs: Vec<Segment>,
    names: HashSet<String>,
    config: Config,
}

impl Default for Writer {
    /// 与 [`Writer::new`] 同义：默认开启两级 CRC 与自适应压缩。
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    /// 空写者，默认开启两级 CRC 与自适应压缩。
    pub fn new() -> Self {
        Self::with(Config::default())
    }

    /// 用给定选项构造写者。
    pub fn with(config: Config) -> Self {
        Self {
            segs: Vec::new(),
            names: HashSet::new(),
            config,
        }
    }

    /// 开关两级校验和（默认开）。
    pub fn checksum(&mut self, on: bool) -> &mut Self {
        self.config.checksum = on;
        self
    }

    /// 开关自适应 rANS 压缩（默认开）；关闭后所有段原样存储。
    pub fn compress(&mut self, on: bool) -> &mut Self {
        self.config.compress = on;
        self
    }

    /// 设置五轴候选集合；[`AxisSet::WITHOUT_RANK`] 可完全跳过 SVD。
    pub fn axes(&mut self, axes: AxisSet) -> &mut Self {
        self.config.axes = axes;
        self
    }

    /// 设置段编码并行度；`0` 与 `1` 都是串行，输出与并行度无关。
    pub fn workers(&mut self, workers: usize) -> &mut Self {
        self.config.workers = workers;
        self
    }

    /// 追加一个已自洽的段；重名在这里被拒。
    pub fn push(&mut self, segment: Segment) -> Result<&mut Self> {
        segment.validate()?;
        if !self.names.insert(segment.name().to_owned()) {
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

    /// 为每段选定存储表示：先做保序去重，再并行编码内联段；引用关系与段序不受并行度影响。
    /// 取值与既有内联段逐位相等的段改写引用，数据区不再重复占字节。
    fn plan(&self) -> Result<Vec<Plan<'_>>> {
        let mut slots = self.classify_slots();
        self.encode_inline(&mut slots)?;
        let mut plans = Vec::with_capacity(slots.len());
        for slot in slots {
            plans.push(slot.ok_or_else(|| crate::bad("去重后段缺存储计划"))?);
        }
        Ok(plans)
    }

    /// 保序去重：逐段比对既有代表段，命中写引用槽，未命中留空待编码内联。
    fn classify_slots(&self) -> Vec<Option<Plan<'_>>> {
        let mut reps: HashMap<(u16, Vec<u64>, Digest), Vec<usize>> = HashMap::new();
        let mut slots = Vec::with_capacity(self.segs.len());
        for (idx, segment) in self.segs.iter().enumerate() {
            let key = (
                segment.dtype().code(),
                segment.shape().to_vec(),
                crate::proof::hash_bytes(&segment.data),
            );
            if let Some(rep) = self.find_rep(&reps, &key, segment) {
                slots.push(Some(Plan::Ref { rep }));
                continue;
            }
            reps.entry(key).or_default().push(idx);
            slots.push(None);
        }
        slots
    }

    /// 编码所有内联段：结果按段下标原位回填，线程调度不改变输出字节。
    fn encode_inline<'a>(&'a self, slots: &mut [Option<Plan<'a>>]) -> Result<()> {
        let idxs: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.is_none().then_some(idx))
            .collect();
        if idxs.is_empty() {
            return Ok(());
        }
        let workers = self.config.workers.max(1).min(idxs.len());
        if workers == 1 {
            self.encode_serial(slots, &idxs);
            return Ok(());
        }
        self.encode_parallel(slots, &idxs, workers)
    }

    /// 单线程回填：逐个编码并按段下标落槽。
    fn encode_serial<'a>(&'a self, slots: &mut [Option<Plan<'a>>], idxs: &[usize]) {
        for &idx in idxs {
            slots[idx] = Some(self.pack_inline(&self.segs[idx]));
        }
    }

    /// 作用域并行编码：按 `workers` 分段派发线程，join 后按下标回填；线程异常按错误上抛。
    fn encode_parallel<'a>(
        &'a self,
        slots: &mut [Option<Plan<'a>>],
        idxs: &[usize],
        workers: usize,
    ) -> Result<()> {
        let chunk_len = idxs.len().div_ceil(workers);
        let encoded = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for chunk in idxs.chunks(chunk_len) {
                handles.push(scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&idx| (idx, self.pack_inline(&self.segs[idx])))
                        .collect::<Vec<_>>()
                }));
            }
            Self::join_plans(handles)
        })?;
        Self::apply_encoded(slots, encoded);
        Ok(())
    }

    /// 等待各段线程收尾并按下标汇总；线程异常按错误上抛。
    fn join_plans<'scope, 'a>(
        handles: Vec<std::thread::ScopedJoinHandle<'scope, Vec<(usize, Plan<'a>)>>>,
    ) -> Result<Vec<(usize, Plan<'a>)>> {
        let mut plans = Vec::new();
        for handle in handles {
            let chunk = handle
                .join()
                .map_err(|_| crate::bad("段编码工作线程异常终止"))?;
            plans.extend(chunk);
        }
        Ok(plans)
    }

    /// 并行编码结果按段下标原位落槽，槽序与输入一致。
    fn apply_encoded<'a>(slots: &mut [Option<Plan<'a>>], encoded: Vec<(usize, Plan<'a>)>) {
        for (idx, plan) in encoded {
            slots[idx] = Some(plan);
        }
    }

    /// 哈希命中后逐位比对元数据与字节；碰撞只错过复用，绝不错配。
    fn find_rep(
        &self,
        reps: &HashMap<(u16, Vec<u64>, Digest), Vec<usize>>,
        key: &(u16, Vec<u64>, Digest),
        segment: &Segment,
    ) -> Option<usize> {
        let candidates = reps.get(key)?;
        candidates
            .iter()
            .copied()
            .find(|&rep| self.segs[rep].same_value(segment) && self.segs[rep].data == segment.data)
    }

    /// 内联段存储表示：压缩开启时先试配置允许的五轴无损选码，再退 rans 帧，
    /// 都不更小则原样 `Store`。
    fn pack_inline<'a>(&self, segment: &'a Segment) -> Plan<'a> {
        if self.config.compress {
            if let Some(artifact) = axes_frame(segment, self.config.axes) {
                return Plan::Inline {
                    codec: Codec::Axes,
                    stored: Cow::Owned(artifact),
                };
            }
            let frame = rans::encode(&segment.data);
            if frame.len() < segment.data.len() {
                return Plan::Inline {
                    codec: Codec::Rans,
                    stored: Cow::Owned(frame),
                };
            }
        }
        Plan::Inline {
            codec: Codec::Store,
            stored: Cow::Borrowed(segment.data.as_slice()),
        }
    }

    fn flags(&self) -> u32 {
        if self.config.checksum {
            GLOBAL_CRC | SEG_CRC
        } else {
            0
        }
    }

    /// 校验对齐并排好每段的存储表示：所有写出路径共用这一份计划，只算一次。
    fn prepare(&self, align_log: u8) -> Result<Vec<Plan<'_>>> {
        Self::check_align(align_log)?;
        self.plan()
    }

    /// 用已算好的 `plan` 写正文；长度预演与实际写出共用同一份编码结果，避免再编一遍。
    fn write_body_in(&self, plan: &[Plan<'_>], out: &mut impl Sink, align_log: u8) -> Result<()> {
        let mut measured = Measured::new(out);
        measured.put(&MAGIC.to_le_bytes());
        measured.put(&VERSION.to_le_bytes());
        measured.put(&self.flags().to_le_bytes());
        measured.put(&[align_log]);
        leb::write(&mut measured, self.segs.len());
        write_descs(&self.segs, plan, self.config.checksum, &mut measured)?;
        write_data(plan, align_log, &mut measured);
        Ok(())
    }

    /// 产出容器字节；`align_log` 是对齐边界的 2 的幂指数（0..=16，常用 6 = 64 B）。
    pub fn finish(&self, align_log: u8) -> Result<Vec<u8>> {
        let plan = self.prepare(align_log)?;
        let expected = self.expected_len(&plan, align_log)?;
        let mut buf = Vec::with_capacity(expected);
        self.write_sealed(&plan, align_log, &mut buf)?;
        debug_assert_eq!(buf.len(), expected, "长度零漂移");
        Ok(buf)
    }

    /// 正文写进密封接收器，全局 CRC 收尾后落回底层缓冲；全局 CRC 本身不进 CRC。
    fn write_sealed(&self, plan: &[Plan<'_>], align_log: u8, buf: &mut Vec<u8>) -> Result<()> {
        let global = {
            let mut sealed = Sealed::new(buf);
            self.write_body_in(plan, &mut sealed, align_log)?;
            if self.config.checksum {
                sealed.checksum()
            } else {
                0
            }
        };
        buf.extend_from_slice(&global.to_le_bytes());
        Ok(())
    }

    /// 预演编码路径得到的整份字节数（正文 + 4 字节全局 CRC），与实际写出共用同一 plan。
    fn expected_len(&self, plan: &[Plan<'_>], align_log: u8) -> Result<usize> {
        let mut count = Count::default();
        self.write_body_in(plan, &mut count, align_log)?;
        Ok(count.0 + 4)
    }

    /// 整份容器的精确字节数；与 [`Writer::finish`] 走同一编码路径，长度预演只此一处实现。
    pub fn encoded_len(&self, align_log: u8) -> Result<usize> {
        let plan = self.prepare(align_log)?;
        self.expected_len(&plan, align_log)
    }

    /// 写进调用方固定缓冲，返回写入字节数；装不下时返回错误而不截断。
    pub fn encode_into(&self, align_log: u8, out: &mut [u8]) -> Result<usize> {
        let plan = self.prepare(align_log)?;
        let mut sink = SliceSink::new(out);
        {
            let mut sealed = Sealed::new(&mut sink);
            self.write_body_in(&plan, &mut sealed, align_log)?;
            let global = if self.config.checksum {
                sealed.checksum()
            } else {
                0
            };
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
        let plan = self.prepare(align_log)?;
        let temp = temp_path(path);
        if let Err(error) = self.write_temp(&plan, &temp, path, align_log) {
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
    fn write_temp(
        &self,
        plan: &[Plan<'_>],
        temp: &Path,
        target: &Path,
        align_log: u8,
    ) -> io::Result<()> {
        let file = std::fs::File::create(temp)?;
        let buffered = io::BufWriter::with_capacity(WRITE_BUF, file);
        self.flush_sink(plan, FileSink::new(buffered), align_log)?;
        std::fs::rename(temp, target)
    }

    /// 边编码边写正文与全局 CRC，随后刷盘并 fsync。
    fn flush_sink(
        &self,
        plan: &[Plan<'_>],
        mut file_sink: FileSink<io::BufWriter<std::fs::File>>,
        align_log: u8,
    ) -> io::Result<()> {
        let global = self.body_checksum(plan, &mut file_sink, align_log)?;
        file_sink.put(&global.to_le_bytes());
        if let Some(error) = file_sink.error() {
            return Err(io::Error::new(error.kind(), error.to_string()));
        }
        debug_assert_eq!(
            file_sink.written(),
            self.expected_len(plan, align_log)
                .map_err(|error| io::Error::other(error.to_string()))?,
            "流式写出与长度承诺零漂移"
        );
        let writer = file_sink.into_inner()?;
        flush_file(writer)
    }

    /// 写正文并取全局 CRC：不校验时写 0。`?` 与校验开关都收在这一处，主流程只剩一次调用。
    fn body_checksum(
        &self,
        plan: &[Plan<'_>],
        file_sink: &mut FileSink<io::BufWriter<std::fs::File>>,
        align_log: u8,
    ) -> io::Result<u32> {
        let mut sealed = Sealed::new(file_sink);
        self.write_body_in(plan, &mut sealed, align_log)
            .map_err(|error| io::Error::other(error.to_string()))?;
        Ok(if self.config.checksum {
            sealed.checksum()
        } else {
            0
        })
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

// —— 写出计划的展开（原 `plan` 平铺进 `writer`）：段序列落成描述符与数据区 ——

/// 五轴无损选码：仅 f32 二维稠密权重（无量化方案）且 artifact 严格更小时才返回。
pub(crate) fn axes_frame(segment: &Segment, axes: AxisSet) -> Option<Vec<u8>> {
    let (m, n) = axes_dims(segment)?;
    crate::axes::compress_with_axes(&segment.data, m, n, axes)
        .ok()
        .flatten()
}

/// 可做五轴选码的形状：f32 二维稠密权重矩阵的 `m×n`，任一维为 0 即不适用。
pub(crate) fn axes_dims(segment: &Segment) -> Option<(usize, usize)> {
    if !axes_matrix(segment) {
        return None;
    }
    let m = usize::try_from(segment.shape()[0]).ok()?;
    let n = usize::try_from(segment.shape()[1]).ok()?;
    (m != 0 && n != 0).then_some((m, n))
}

/// 段是否是可做五轴选码的 f32 二维稠密权重。
pub(crate) fn axes_matrix(segment: &Segment) -> bool {
    segment.dtype() == Wire::F32 && segment.quant().is_none() && segment.shape().len() == 2
}

/// 写端布局：内联段带自己的存储帧与段级 CRC；引用段只记代表段下标，存储帧与校验都归代表段。
pub(crate) enum Plan<'a> {
    Inline { codec: Codec, stored: Cow<'a, [u8]> },
    Ref { rep: usize },
}

/// 描述符：内联段级 CRC 覆盖实际存储字节，在这里算好写进头部；引用段不写 CRC。
pub(crate) fn write_descs(
    segs: &[Segment],
    plan: &[Plan<'_>],
    checksum: bool,
    out: &mut impl Sink,
) -> Result<()> {
    for (segment, item) in segs.iter().zip(plan) {
        write_one_desc(segment, item, checksum, out)?;
    }
    Ok(())
}

/// 单条描述符：内联段级 CRC 覆盖实际存储字节；引用段只写代表段下标，不写 CRC。
pub(crate) fn write_one_desc(
    segment: &Segment,
    item: &Plan<'_>,
    checksum: bool,
    out: &mut impl Sink,
) -> Result<()> {
    match item {
        Plan::Inline { codec, stored } => {
            let seg_crc = if checksum { crc::crc32(stored) } else { 0 };
            segment.encode_desc(out, *codec, stored.len() as u64, seg_crc)
        }
        Plan::Ref { rep } => segment.encode_ref_desc(out, *rep as u32),
    }
}

/// 数据区：内联段先按对齐补零再写存储帧；引用段不占数据区。
pub(crate) fn write_data<S: Sink>(
    plan: &[Plan<'_>],
    align_log: u8,
    measured: &mut Measured<'_, S>,
) {
    const ZEROS: [u8; 64] = [0; 64];
    let align = 1usize << align_log;
    for item in plan {
        let Plan::Inline { stored, .. } = item else {
            continue;
        };
        let pos = measured.len();
        let aligned = pos.next_multiple_of(align);
        let mut left = aligned - pos;
        while left > 0 {
            let chunk = left.min(ZEROS.len());
            measured.put(&ZEROS[..chunk]);
            left -= chunk;
        }
        measured.put(stored);
    }
}

/// 刷出 BufWriter、取出文件并 fsync。
pub(crate) fn flush_file(mut writer: io::BufWriter<std::fs::File>) -> io::Result<()> {
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(io::IntoInnerError::into_error)?;
    file.sync_all()
}

/// 同目录临时路径：放在目标目录才能保证 rename 在同一文件系统上原子。
pub(crate) fn temp_path(path: &Path) -> PathBuf {
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
#[path = "../tests/writer.rs"]
mod tests;
