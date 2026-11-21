//! 容器生命周期模式与增量补丁：训练/发布判定、`opt.` 状态剥离、逐字节 XOR 补丁。
//!
//! 模式判定与剥离只依赖已读 [`Archive`]；补丁是自逆 XOR：base 缺失或不等长按全量保留，
//! 不做第二份容错。

use diagnostic::{Code, Error, Result};

use crate::{Archive, Segment, Stored, Wire};

/// 模式标记段的保留名。
pub const MODE_NAME: &str = "llm.mode";
/// 优化器状态段的命名空间前缀。
const OPT_PREFIX: &str = "opt.";

/// 容器生命周期模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 训练断点：含优化器/动量/方差等 `opt.` 状态。
    Train,
    /// 推理发布：剥离训练状态的精简产物。
    Infer,
}

impl Mode {
    fn code(self) -> u8 {
        match self {
            Mode::Train => 0,
            Mode::Infer => 1,
        }
    }

    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Mode::Train),
            1 => Ok(Mode::Infer),
            other => Err(bad(&format!("mode: 未知码 {other}"))),
        }
    }

    /// 编码为可写进容器的标记段：保留名、`U8` 单轴、1 字节。
    pub fn to_segment(self) -> Result<Segment> {
        Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![self.code()])
    }
}

/// 读取模式；无标记段的旧文件按推理模式处理。
///
/// 有标记段但码位不在闭集内时显式报错，不静默落进推理模式：未知码通常意味着文件损坏
/// 或版本错位，按推理放行会丢掉训练状态段。
pub fn mode_of(archive: &Archive<'_>) -> Result<Mode> {
    match archive
        .get(MODE_NAME)
        .and_then(|seg| archive.read(seg).first().copied())
    {
        None => Ok(Mode::Infer),
        Some(code) => Mode::from_code(code),
    }
}

/// 段名是否属于优化器状态。
pub fn is_opt(name: &str) -> bool {
    name.starts_with(OPT_PREFIX)
}

/// 发布转换：剥离所有 `opt.` 段，保留其余段并标记为推理模式。
pub fn release(src: &Archive<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    for seg in src.segments() {
        if !is_opt(seg.name()) {
            out.push(keep_segment(src, seg)?);
        }
    }
    out.push(Mode::Infer.to_segment()?);
    Ok(out)
}

/// 把读端段还原成写端段：沿用名字/类型/形状/量化方案，`data` 是已就位的逻辑字节。
/// `release` 与补丁重建共用这一份字段装配，避免四处各拼一遍 `Segment::new`。
fn to_segment(seg: &Stored, data: Vec<u8>) -> Result<Segment> {
    Segment::new(
        seg.name(),
        seg.dtype(),
        seg.shape().to_vec(),
        seg.quant().cloned(),
        data,
    )
}

/// 把读端段还原成写端段：先加载逻辑字节，再交 [`to_segment`] 装配。
fn keep_segment(src: &Archive<'_>, seg: &Stored) -> Result<Segment> {
    let data = src.load(seg)?.into_owned();
    to_segment(seg, data)
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("mode")
}

// —— 增量补丁：逐字节 XOR，读端按 base 自逆合成 ——

/// 计算 cur 相对 base 的补丁段（逻辑字节，未压缩；交给 Writer 落盘并按需 rans）。
pub fn diff(base: &Archive<'_>, cur: &Archive<'_>) -> Result<Vec<Segment>> {
    rebuild(base, cur)
}

/// 用 base 与补丁归档合成完整段。
pub fn join(base: &Archive<'_>, patch: &Archive<'_>) -> Result<Vec<Segment>> {
    rebuild(base, patch)
}

/// diff 与 join 同构：对 right 每个段，base 有同名等长段就 XOR，否则按全量保留。
fn rebuild(base: &Archive<'_>, right: &Archive<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    for seg in right.segments() {
        let bytes = right.load(seg)?.into_owned();
        let data = apply(base, seg.name(), bytes)?;
        out.push(to_segment(seg, data)?);
    }
    Ok(out)
}

/// 同名等长则 XOR；base 缺失或长度不同则原样返回（全量）。
fn apply(base: &Archive<'_>, name: &str, bytes: Vec<u8>) -> Result<Vec<u8>> {
    let Some(prev) = base.get(name) else {
        return Ok(bytes);
    };
    let prev_bytes = base.load(prev)?;
    if prev_bytes.len() != bytes.len() {
        return Ok(bytes);
    }
    Ok(xor(&bytes, &prev_bytes))
}

fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

pub mod import {
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
}

#[cfg(test)]
#[path = "../tests/lifecycle.rs"]
mod tests;
