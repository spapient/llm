//! 权重文件的结构：[`Entry`]、[`Header`] 与 [`Reader`]。
//!
//! **拥有**：`Header`、`Entry`、`Reader`。**不拥有**：张量本身（`tensor`）、布局（`storage`）、
//! safetensors / GGUF 的具体解析器（P1：解析器只填这三个类型，不新增契约）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`train`、`llm`。
//!
//! 这一层先解决"权重文件的元数据能不能自证合法"：条目名唯一、字节区间不重叠、区间起点不早于
//! 数据段起点。这三条不成立时，任何后续读写都会在别处以更难查的方式失败。
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

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

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
        let mut seen = std::collections::BTreeSet::new();
        for entry in &entries {
            if !seen.insert(entry.name()) {
                return Err(
                    Error::new(Code::Invalid, format!("权重条目 `{}` 重复", entry.name()))
                        .with("weights::Header::new"),
                );
            }
            if entry.offset < data_start {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "条目 `{}` 的偏移 {} 早于数据段起点 {data_start}",
                        entry.name(),
                        entry.offset
                    ),
                )
                .with("weights::Header::new"));
            }
        }
        let mut sorted = entries;
        sorted.sort_by_key(Entry::offset);
        for pair in sorted.windows(2) {
            let (left, right) = (&pair[0], &pair[1]);
            if left.end() > right.offset() {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "条目 `{}` 的区间与 `{}` 重叠（{} > {}）",
                        left.name(),
                        right.name(),
                        left.end(),
                        right.offset()
                    ),
                )
                .with("weights::Header::new"));
            }
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

/// 读取条目字节的后端：文件、内存映射或测试夹具。
///
/// 只有这一个方法：头部解析、校验与张量构造都在别处完成，读端不需要知道格式。
pub trait Reader {
    /// 读出条目的原始字节；长度必须与 `entry.len()` 一致。
    fn read(&self, entry: &Entry) -> Result<&[u8]>;
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
