//! 中性条目流导入：外部权重源 → 既有 [`Segment`] 列表，不新增第二份条目类型。
//!
//! 各适配通路只做「源字段 → 中性段」的归一：名字、元素类型、形状、字节落 [`Segment`]，
//! 量化配对、压缩与落盘仍走 `Writer`。读不通或类型不在映射闭集内显式报错
//! （[`Code::Unsupported`]），不静默转换、不丢精度、不伪造反量化。

use std::path::Path;

use diagnostic::{Code, Error, Result};

use crate::bin::Bin;
use crate::gguf::{Gguf, Kind};
use crate::safetensors::{Archive as StArchive, File as StFile};
use crate::{Entry, Loader, Reader, Segment, Wire};

/// 按扩展名把源文件分派到对应适配通路；不认识的后缀显式 [`Code::Unsupported`]，不猜格式、
/// 不把块量化伪装成标量段。
///
/// - `.safetensors`：只读头部的 `File` 后端，按条目按需 seek+read；
/// - `.gguf`：整份读入后交 GGUF 解析，标量 `F32`/`F16` 直映，块量化显式拒绝；
/// - `.bin` / `.pt`：整份读入后交 bin（ZIP_STORED + pickle 栈机）解析。
///
/// # Errors
///
/// 文件 IO 失败按 [`Code::Io`]；后缀未登记、读不通或类型不在映射闭集内按各通路既有诊断拒绝。
pub fn segments_from_path(path: &Path) -> Result<Vec<Segment>> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("safetensors") => segments_from_safetensors_file(path),
        Some("gguf") => load_gguf(path),
        Some("bin") | Some("pt") => load_bin(path),
        other => Err(unsupported(other)),
    }
}

/// 后缀未登记：不猜格式，也不把块量化伪装成标量段。
fn unsupported(other: Option<&str>) -> Error {
    Error::new(
        Code::Unsupported,
        format!(
            "不认识的源格式 `{}`：支持 .safetensors / .gguf / .bin / .pt",
            other.unwrap_or("无扩展名")
        ),
    )
    .with("weights::import::segments_from_path")
}

/// GGUF 通路：整份读入 → 解析 → 中性段流。
fn load_gguf(path: &Path) -> Result<Vec<Segment>> {
    let bytes = read_file(path)?;
    segments_from_gguf(&Gguf::parse(&bytes)?)
}

/// bin/pickle 通路：整份读入 → 解析 → 中性段流。
fn load_bin(path: &Path) -> Result<Vec<Segment>> {
    let bytes = read_file(path)?;
    segments_from_bin(&Bin::parse(&bytes)?)
}

/// 整份读入文件为自有字节；GGUF/bin 解析需要整份缓冲，safetensors 条目通路不走这里。
fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| {
        Error::new(Code::Io, format!("读取 {} 失败：{error}", path.display()))
            .with("weights::import::read_file")
    })
}

/// safetensors 条目 → 中性段：字段与长度判据同源，文件与内存归档共用。
fn push_entry_segment(out: &mut Vec<Segment>, entry: &Entry, bytes: Vec<u8>) -> Result<()> {
    let dims: Vec<u64> = entry.shape().dims().iter().map(|&d| d as u64).collect();
    out.push(segment_from_entry(
        entry.name(),
        entry.dtype(),
        &dims,
        bytes,
    )?);
    Ok(())
}

/// safetensors 文件 → 中性段流：`File` 后端按条目按需 seek+read，不把整份文件读进内存。
///
/// # Errors
///
/// 文件打开/读取失败按后端诊断；条目字段与字节不自洽时按 [`Segment::new`] 的既有规则拒绝。
pub fn segments_from_safetensors_file(path: &Path) -> Result<Vec<Segment>> {
    let file = StFile::open(path)?;
    let entries = file.header().entries();
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        push_entry_segment(&mut out, entry, file.load(entry)?)?;
    }
    Ok(out)
}

/// safetensors 内存归档 → 中性段流：借用路径与文件路径同口径，字节零拷贝读出后再落段。
///
/// # Errors
///
/// 与 [`segments_from_safetensors_file`] 同一套字段/长度判据。
pub fn segments_from_safetensors_archive(arch: &StArchive<'_>) -> Result<Vec<Segment>> {
    let entries = arch.header().entries();
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        push_entry_segment(&mut out, entry, arch.read(entry)?.to_vec())?;
    }
    Ok(out)
}

/// GGUF 归档 → 中性段流：标量 `F32`/`F16` 直映 Wire；GGML 块量化（`Q8_0`/`Q4_K`/`Q5_K`/
/// `Q6_K` 等）显式 [`Code::Unsupported`]，本刀不做反量化，也不把块字节塞进标量段伪装。
///
/// # Errors
///
/// [`Code::Unsupported`]：张量类型是未映射的 GGML 块量化；其余读取错误透传 GGUF 解析诊断。
pub fn segments_from_gguf(arch: &Gguf<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::with_capacity(arch.tensors().len());
    for tensor in arch.tensors() {
        let wire = gguf_wire(tensor.kind, &tensor.name)?;
        let raw = arch.raw(tensor)?.to_vec();
        out.push(segment_from_entry(&tensor.name, wire, &tensor.shape, raw)?);
    }
    Ok(out)
}

/// GGML 类型 → 中性 Wire：标量 `F32`/`F16` 直映；块量化显式 `Unsupported`，不反量化、不塞伪装段。
fn gguf_wire(kind: Kind, name: &str) -> Result<Wire> {
    match kind {
        Kind::F32 => Ok(Wire::F32),
        Kind::F16 => Ok(Wire::F16),
        other => Err(Error::new(
            Code::Unsupported,
            format!("GGUF 张量 `{name}` 的块量化 {other:?} 不支持导入：未实现反量化"),
        )
        .with("weights::import::segments_from_gguf")),
    }
}

/// bin-zip / bin-pickle 归档 → 同一中性段流；读不通的格式在 `bin` 解析层已显式失败，
/// 这里只做字段归一，不再补第二份容错。
///
/// # Errors
///
/// 条目字节区间或字段不自洽时按 [`Segment::new`] 既有规则拒绝。
pub fn segments_from_bin(bin: &Bin<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::with_capacity(bin.tensors().len());
    for tensor in bin.tensors() {
        let raw = bin.data(tensor)?.to_vec();
        out.push(segment_from_entry(
            tensor.name(),
            tensor.dtype(),
            tensor.shape(),
            raw,
        )?);
    }
    Ok(out)
}

/// 外部源字段 → 中性段：形状轴长统一转 `u64`；量化方案不在导入侧挂载（标度配对仍是
/// 源格式自己的条目）。类型/长度不自洽时透传 [`Segment::new`] 诊断。
fn segment_from_entry(name: &str, wire: Wire, dims: &[u64], bytes: Vec<u8>) -> Result<Segment> {
    Segment::new(name, wire, dims.to_vec(), None, bytes)
}
