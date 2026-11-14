//! 权重文件的结构：[`Entry`]、[`Header`] 与两条读取契约（[`Reader`] / [`Loader`]），以及
//! safetensors 的两种后端（[`safetensors::Archive`] 借用整份字节、[`safetensors::File`] 只读头部）。
//!
//! **拥有**：`Header`/`Entry`/`Reader`/`Loader` 三条契约，`safetensors` 的两种后端，自造的 `LLPK`
//! 容器（`writer`/`reader`/`seg` 与 `merkle`/`rans` 这些小件；CRC/LEB/Sink 已归 L1 `wire`，见 ADR 0024），以及只能由
//! 本层判断的格式适配器与完整性证明（`bin` 的 pickle 栈机、`zip`、`gguf`、`json`，`axes`/`lattice`/
//! `quant`/`scale` 的量化族，`dsa`/`envelope`/`sha256`/`keccak`/`ntt`/`golay`/`svd`/`ledger`）。
//! **不拥有**：dtype 闭集（`dtype`）、张量元数据（`tensor`）、物理布局（`storage`）。
//! **依赖**：`dtype`、`diagnostic`、`tensor`、`wire`（`+dev`：`storage`）。
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

// 自有 LLPK 容器：私有实现模块，公开类型在末尾统一 re-export。
mod affine;
pub mod axes;
pub mod golay;
mod ir;
pub mod lattice;
pub mod ledger;
mod merkle;
mod rans;
mod reader;
pub mod scale;
mod seg;
mod sha256;
pub mod svd;
mod writer;
mod zip;

// 纯字节码基础件归 L1 `wire`（ADR 0024）；这里只做薄转发，保持容器内部路径不变。
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

#[cfg(test)]
mod testutil;

pub mod dsa;
pub mod envelope;
pub mod keccak;
pub mod mode;
pub mod ntt;
pub mod patch;

// 外部格式适配（只读）：safetensors/gguf/bin 已接入。
pub mod bin;
pub mod gguf;
pub mod json;
pub mod quant;
pub mod safetensors;

pub use affine::Quant;
pub use seg::Codec;
pub use dtype::DType as Wire;
pub use envelope::Envelope;
pub use ir::{Graph, Node, Op};
pub use merkle::Proof;
pub use quant::Quantized;
pub use reader::{Archive, Stored};
pub use seg::Segment;
pub use writer::Writer;

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
mod tests {
    use super::*;

    fn entry(name: &str, offset: usize) -> Entry {
        Entry::new(name, DType::F32, Shape::new([2, 3])).at(offset)
    }

    #[test]
    fn a_valid_header_sorts_entries_and_reports_the_data_span() {
        let header = Header::new(vec![entry("b", 88), entry("a", 64)], 64).expect("合法");
        let names: Vec<&str> = header.entries().iter().map(Entry::name).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(header.data_start(), 64);
        assert_eq!(header.data_bytes(), 48);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let error = Header::new(vec![entry("a", 64), entry("a", 88)], 64).expect_err("重名");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("重复"), "{error}");
    }

    #[test]
    fn overlapping_intervals_are_rejected() {
        let error = Header::new(vec![entry("a", 64), entry("b", 80)], 64).expect_err("重叠");
        assert!(error.to_string().contains("重叠"), "{error}");
    }

    #[test]
    fn an_entry_before_the_data_section_is_rejected() {
        let error = Header::new(vec![entry("a", 8)], 64).expect_err("早于数据段");
        assert!(error.to_string().contains("数据段"), "{error}");
    }

    #[test]
    fn a_zero_sized_entry_is_allowed_and_costs_nothing() {
        let empty = Entry::new("empty", DType::U4, Shape::new([0])).at(64);
        assert!(empty.is_empty());
        let header = Header::new(vec![empty], 64).expect("合法");
        assert_eq!(header.data_bytes(), 0);
        assert!(header.get("missing").is_none());
    }
}
