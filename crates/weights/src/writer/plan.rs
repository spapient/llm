//! 写出计划的展开：把 [`crate::writer::Writer`] 排好的段序列落成描述符与数据区。
//!
//! 从 [`crate::writer`] 拆出：那边是「顺序、去重、对齐」的决策，这里是「决策怎么变成字节」，
//! 于是同一份 [`Plan`] 可以同时喂给内存缓冲、文件与测量三种 sink。

use std::borrow::Cow;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use diagnostic::Result;

use crate::Codec;
use crate::Wire;
use crate::crc;
use crate::seg::Segment;
use wire::sink::{Measured, Sink};

/// 五轴无损选码：仅 f32 二维稠密权重（无量化方案）且 artifact 严格更小时才返回。
pub(crate) fn axes_frame(segment: &Segment) -> Option<Vec<u8>> {
    let (m, n) = axes_dims(segment)?;
    crate::axes::compress(&segment.data, m, n).ok().flatten()
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
    let align = 1usize << align_log;
    for item in plan {
        let Plan::Inline { stored, .. } = item else {
            continue;
        };
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
