//! 权重文件的结构：[`Entry`]、[`Header`] 与两条读取契约（[`Reader`] / [`Loader`]），以及
//! safetensors 的两种后端（[`safetensors::Archive`] 借用整份字节、[`safetensors::File`] 只读头部）。
//!
//! **拥有**：`Header`/`Entry`/`Reader`/`Loader` 三条契约，`safetensors` 的两种后端，自有的 `.u`
//! 容器 v2（`writer`/`archive`/`seg`/`proof`/`rans`；CRC/LEB/Sink 已归 L1 `wire`）以及只能由
//! 本层判断的格式适配器与量化方案（`bin` 栈机含 ZIP_STORED 读取、`gguf`、`json`、`quant`）。
//! **不拥有**：dtype 闭集（`dtype`）、张量元数据（`tensor`）、物理布局（`storage`）。
//! **依赖**：`dtype`、`diagnostic`、`tensor`、`wire`（`storage` 经可选 `mmap` feature 引入，
//! 只在 Unix 上提供映射源；`+dev`：`testkit`）。
//! **被依赖**：`train`、`nn`、`llm`（`+dev`：`fuzz`）。
//!
//! 两条读取契约的分工是所有权而不是风格：[`Reader`] 把字节**借**出来（内存切片、映射），
//! [`Loader`] 把字节**拷进**调用方给的缓冲区——只有后者能由文件实现，因为不引入 `unsafe` 就无法
//! 自己映射，也无法承诺"借出的引用在函数返回后仍然有效"。真实权重动辄十几 GB，靠的正是后者。
//!
//! 这一层先解决“权重文件能不能自证合法”：条目名唯一、字节区间不重叠、区间起点不早于
//! 数据段，以及 safetensors 的“声明区间必须等于形状推出的字节数”。这几条不成立时，任何
//! 后续读写都会在别处以更难查的方式失败。
//!
//! 解析器只填这三个类型，不新增契约：换个文件格式（GGUF）只需再写一个“字节 → `Header`”，
//! 上层（`nn` 的层清单、`train` 的检查点）不需要跟着改。
//!
//! # 例子
//!
//! ```
//! use tensor::{DType, Shape};
//! use llm_weights::{Entry, Header};
//!
//! let entry = Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64);
//! let header = Header::new(vec![entry], 64).expect("区间合法");
//! assert_eq!(header.data_bytes(), 24);
//! assert!(header.get("blk.0.attn_q.weight").is_some());
//! ```

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

/// 校验失败时的定位串：两个判据共用同一处标签，改动时不必同步三处。
const SITE: &str = "weights::Header::new";

/// 向上取整到 `align` 的整数倍：`align` 必须是 2 的幂；非法对齐或对齐后溢出返回 `None`。
/// 调用方把 `None` 映射到各自站点的诊断，算术与幂次判据只在此处收口一处。
fn align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    value.div_ceil(align).checked_mul(align)
}

// —— 自有 `.u` 容器 v2：描述符、读/写、帧编码、增量补丁与双模式 ——
mod archive;
mod lifecycle;
mod rans;
mod seg;
mod writer;

// —— 量化方案：仿射方案与外部量化条目 ——
pub mod quant;

// —— 2/4/8-bit 对称 PTQ：f32 → 打包码的正向量化与其反解（离线换算，ADR 0018） ——
pub mod ptq;

// 自适应档位分配：ptq 的子域，平铺到顶层以过 N3（原 `ptq/adaptive`）。
mod adaptive;

// —— 完整性证明：Merkle + SHA-256 ——
mod proof;

// —— 谱隙证书链：f64 单边 Jacobi SVD、谱隙判定账本、五轴无损选码（.u v2 张量级压缩侧车）——
pub mod axes;
pub mod ledger;
pub use ptq::svd;

// —— 沉降账本：LLD §4.7 的逐张量体积 / 原因 / 无损判决，全部由落盘字节现算 ——
pub mod settle;

// —— L3 合成律：LLD §4.5 的逐段证书 → 网络级误差上界与适用域 ——
pub mod l3;

// —— 外部格式只读适配：safetensors/gguf/bin/json ——
pub mod bin;
pub mod gguf;
pub use lifecycle::import;
pub mod json;
pub mod safetensors;

// 只读内存映射源：feature + Unix 双门控，复用 `storage` 的唯一 unsafe 岛。
#[cfg(all(feature = "mmap", unix))]
mod mmap;

// 纯字节码基础件归 L1 `wire`；这里只做薄转发，保持容器内部路径不变。
/// CRC-32（转发自 `wire::crc`）。
pub mod crc {
    pub use wire::crc::*;
}

mod leb {
    pub(crate) use wire::leb::*;
}

mod sink {
    pub(crate) use wire::sink::*;
}

/// 生命周期模式薄转发：实现见内部 `lifecycle` 模块。
pub mod mode {
    pub use crate::lifecycle::{MODE_NAME, Mode, is_opt, mode_of, release};
}

/// 增量补丁薄转发：实现见内部 `lifecycle` 模块。
pub mod patch {
    pub use crate::lifecycle::{diff, join};
}

#[cfg(test)]
#[path = "../tests/testutil.rs"]
mod testutil;

/// 词表段保留名（字节容器占位，不接 tokenizer）。
pub use crate::archive::VOCAB_NAME;

pub use archive::{Archive, Metrics, Stored};
pub use dtype::DType as Wire;
pub use l3::{Bracket, Chain, Ruling, Rung};
pub use proof::Proof;
pub use ptq::{
    Adaptive, AdaptiveCompensated, Compensated, I2, I3, I4, I8, Scheme, adaptive_quantize,
    compensate, compensate_adaptive, dequantize, quantize,
};
pub use quant::{Quant, Quantized};
pub use seg::Codec;
pub use seg::Segment;
pub use settle::{Reason, Settled, Settlement};
pub use writer::Writer;

#[cfg(all(feature = "mmap", unix))]
pub use mmap::MmapSource;

/// 一个权重条目：名字、元素类型、形状与它在数据段里的字节区间。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    name: String,
    dtype: DType,
    shape: Shape,
    offset: usize,
}

impl Entry {
    /// 由名字、类型与形状构造（偏移默认 0，用 [`Entry::at`] 指定）。
    ///
    /// 字节长度不单独存：它由 `shape.element_bytes(dtype)` 推出，避免两处漂移。
    pub fn new(name: impl Into<String>, dtype: DType, shape: Shape) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            offset: 0,
        }
    }

    /// 指定在数据段里的起始偏移。
    #[must_use]
    pub fn at(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// 条目名（权重文件里的键）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 起始偏移。
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 字节长度（由形状与类型推出）。
    pub fn len(&self) -> usize {
        self.shape.element_bytes(self.dtype)
    }

    /// 是否为零字节条目。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 结束偏移（不含）。
    pub fn end(&self) -> usize {
        self.offset.saturating_add(self.len())
    }
}

/// 权重文件的头部：条目表 + 数据段起点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    entries: Vec<Entry>,
    data_start: usize,
}

impl Header {
    /// 校验并构造：名字唯一、区间按偏移有序且不重叠、起点不早于数据段。
    pub fn new(entries: Vec<Entry>, data_start: usize) -> Result<Self> {
        let mut seen = BTreeSet::new();
        for entry in &entries {
            check_entry(&mut seen, entry, data_start)?;
        }
        let mut sorted = entries;
        sorted.sort_by_key(Entry::offset);
        for pair in sorted.windows(2) {
            check_overlap(&pair[0], &pair[1])?;
        }
        Ok(Self {
            entries: sorted,
            data_start,
        })
    }

    /// 全部条目，按偏移排序。
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// 按名字查条目。
    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.name() == name)
    }

    /// 数据段起点（头部长度）。
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// 数据段实际占用的字节数：最后一个条目结束位置减数据段起点。
    pub fn data_bytes(&self) -> usize {
        self.entries
            .last()
            .map(|entry| entry.end().saturating_sub(self.data_start))
            .unwrap_or(0)
    }
}

/// 单条目的两条判据：名字不能重复、起点不能早于数据段。
///
/// 「区间不重叠」不在这里：它要按偏移排序后才能判，排序是 [`Header::new`] 的事。
fn check_entry<'a>(
    seen: &mut BTreeSet<&'a str>,
    entry: &'a Entry,
    data_start: usize,
) -> Result<()> {
    let name = entry.name();
    if !seen.insert(name) {
        let message = format!("权重条目 `{name}` 重复");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    if entry.offset < data_start {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "条目 `{name}` 的偏移 {} 早于数据段起点 {data_start}",
                entry.offset
            ),
        )
        .with(SITE));
    }
    Ok(())
}

/// 相邻两条目（已按偏移排序）不能重叠：重叠意味着后者的起点落在前者的字节区间里。
fn check_overlap(left: &Entry, right: &Entry) -> Result<()> {
    if left.end() <= right.offset() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "条目 `{}` 的区间与 `{}` 重叠（{} > {}）",
            left.name(),
            right.name(),
            left.end(),
            right.offset()
        ),
    )
    .with(SITE))
}

/// 零拷贝后端：把条目字节**借**出来（内存切片、映射或测试夹具）。
///
/// 只有这一个方法：头部解析、校验与张量构造都在别处完成，读端不需要知道格式。借用意味着
/// "后端必须活得比返回值长"，所以文件这类"字节不经我手"的来源只能用 [`Loader`]。
pub trait Reader {
    /// 读出条目的原始字节；长度必须与 `entry.len()` 一致。
    fn read(&self, entry: &Entry) -> Result<&[u8]>;
}

/// 按需后端：把条目字节**拷进**调用方给的缓冲区（文件、压缩包、远端分块）。
///
/// 只有 [`Loader::copy_into`] 是必须实现的，[`Loader::load`] 是它的分配封装——两种后端都不必
/// 重写，因此"长度对不对、越界没越界"只有一处判断，不会出现第二份边界检查。
pub trait Loader {
    /// 把条目字节写进 `out`；`out` 必须正好等于 `entry.len()` 字节。
    ///
    /// # Errors
    ///
    /// 长度不符时返回 [`Code::Invalid`]（这是调用方的错，不能截断也不能补零）；
    /// 读不出来时由具体后端给出（文件是 [`Code::Io`]，借用的后端是 [`Code::Internal`]）。
    fn copy_into(&self, entry: &Entry, out: &mut [u8]) -> Result<()>;

    /// 分配并读出条目字节。
    ///
    /// 分配大小由 `entry.len()` 决定，不由后端决定：读端想限制内存就只能自己调
    /// [`Loader::copy_into`]，这个默认实现不做额外预算。
    fn load(&self, entry: &Entry) -> Result<Vec<u8>> {
        let mut buffer = vec![0u8; entry.len()];
        self.copy_into(entry, &mut buffer)?;
        Ok(buffer)
    }
}

/// 把一段小端 f32 条目字节解包成 `Vec<f32>`；字节数必须是 4 的整数倍。
///
/// 元素解包是权重条目进入内存缓冲的唯一入口：读端不另写 `from_le_bytes` 循环，
/// 半字节目节在此报 [`Code::Invalid`]，不带进装配期。
///
/// # Errors
///
/// 余字节非零时返回 [`Code::Invalid`]（条目不是完整 f32）。
pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let (words, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "条目长度 {} 不是 f32 的整数倍（余 {} 字节）",
                bytes.len(),
                rest.len()
            ),
        )
        .with("weights::decode_f32"));
    }
    Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
}

/// 把一条 F32 条目解码进驻留 `f32` 缓冲：装载期一次性读出并解包，生成循环内不再读文件、
/// 也不二次分配元素缓冲。
///
/// 类型判据与字节解包都收口在这里：非 F32 条目不进 f32 缓冲（返回 [`Code::Unsupported`]，
/// 由装配层按格式拒绝），余字节非零的半字节目节在 [`decode_f32`] 报 [`Code::Invalid`]。
pub fn load_f32(entry: &Entry, loader: &impl Loader) -> Result<Vec<f32>> {
    let bytes = loader.load(entry)?;
    decode_entry_f32(entry, &bytes)
}

/// 借用后端（内存归档、只读映射）的对称入口：条目字节从 [`Reader`] 借出来直接解包，
/// 省掉 [`Loader::load`] 那一次 `Vec<u8>` 字节拷贝；f32 主机序解包仍是必须的，驻留缓冲
/// 拥有形态不变。类型/边界判据与 [`load_f32`] 走同一个 [`decode_entry_f32`]，两条路径不各写一份。
pub fn read_f32(entry: &Entry, reader: &impl Reader) -> Result<Vec<f32>> {
    let bytes = reader.read(entry)?;
    decode_entry_f32(entry, bytes)
}

/// 按条目类型把一段已读回的字节解包成 `f32` 驻留缓冲；`Reader` 借用路径与 `Loader` 拷贝
/// 路径共用这一份类型/边界判据。
pub fn decode_entry_f32(entry: &Entry, bytes: &[u8]) -> Result<Vec<f32>> {
    if entry.dtype() != Wire::F32 {
        return Err(Error::new(
            Code::Unsupported,
            format!("条目 `{}` 不是 F32，不能进 f32 驻留缓冲", entry.name()),
        )
        .with("weights::decode_entry_f32"));
    }
    decode_f32(bytes)
}

/// 从游标读一个线上类型码并解析成 [`Wire`]：seg 描述符与 quant 方案共用这一份解码，
/// 避免两份"u32 → u16 → from_code"的同构步骤漂移。
pub(crate) fn read_wire(cur: &mut leb::Cursor<'_>) -> Result<Wire> {
    let code = cur.read::<u32>()?;
    let code = u16::try_from(code)
        .map_err(|_| Error::new(Code::Invalid, "wire code 超出 u16").with("weights"))?;
    Wire::from_code(code).ok_or_else(|| Error::new(Code::Invalid, "未知 wire code").with("weights"))
}

/// 缓冲区长度必须与条目声明一致：短了会读出半个张量，长了会把相邻条目混进来。
fn check_capacity(entry: &Entry, out: &[u8]) -> Result<()> {
    if out.len() == entry.len() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "缓冲区 {} 字节，条目 `{}` 需要 {} 字节",
            out.len(),
            entry.name(),
            entry.len()
        ),
    )
    .with("weights::Loader::copy_into"))
}

#[cfg(test)]
#[path = "../tests/lib.rs"]
mod tests;
