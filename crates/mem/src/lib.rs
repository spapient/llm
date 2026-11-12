//! 设备内存分配：顺序分配的 [`Arena`] 与它发出的字节区间 [`Region`]。
//!
//! **拥有**：`Arena`、`Region`、`align_up`。**不拥有**：地址空间（`GM` / `FAST` / `REG` 是 DSL 的
//! 语言概念，归 `cutile` / `prof`）、布局（归 `storage`）、张量元数据（归 `tensor`）。
//! **依赖**：`diagnostic`。**被依赖**：`tensor`（`Tensor::from_region` 借用 `Arena` 发出的区间）。
//!
//! 这一层刻意只认字节：分配器不知道数据类型，也不知道地址空间——那是档案声明的属性，
//! 不是分配器的属性（`docs/ARCHITECTURE.md` 第三节）。
//!
//! # 例子
//!
//! ```
//! use llm_mem::Arena;
//!
//! let mut arena = Arena::new();
//! let first = arena.alloc(16, 64).expect("分配");
//! let second = arena.alloc(8, 64).expect("分配");
//! assert_eq!(first.offset(), 0);
//! assert_eq!(second.offset(), 64, "第二次分配必须按 64 字节对齐");
//! assert_eq!(arena.slice(second).map(<[u8]>::len), Some(8));
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};

/// 把 `value` 向上取整到 `align` 的整数倍。
///
/// `align` 必须是 2 的幂（与设备对 `malloc` 的要求一致）；`0` 或非 2 的幂直接报错，
/// 而不是悄悄按 1 处理——对齐写错只会表现为难以复现的性能或正确性问题。
pub fn align_up(value: usize, align: usize) -> Result<usize> {
    if align == 0 || !align.is_power_of_two() {
        return Err(
            Error::new(Code::Invalid, format!("对齐必须是 2 的幂，得到 {align}"))
                .with("mem::align_up"),
        );
    }
    value
        .div_ceil(align)
        .checked_mul(align)
        .ok_or_else(|| Error::new(Code::Internal, "对齐后长度溢出").with("mem::align_up"))
}

/// 一段已分配的字节区间：`[offset, offset + len)`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region {
    offset: usize,
    len: usize,
}

impl Region {
    /// 由偏移与长度构造（只用于反序列化与测试，正常路径是 [`Arena::alloc`]）。
    pub fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }

    /// 起始偏移。
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 字节数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空区间。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 结束偏移（不含）。
    pub fn end(&self) -> usize {
        self.offset + self.len
    }
}

/// 顺序分配器：只前进、不回收，适合"一次请求的临时缓冲"。
///
/// 真正的设备内存池（空闲链表、跨请求复用）是 P1 的事；先把"谁能分配、谁能拿到只读视图"
/// 这条契约定下来，`tensor` 才有东西可依赖。
#[derive(Debug, Default)]
pub struct Arena {
    bytes: Vec<u8>,
    cursor: usize,
}

impl Arena {
    /// 空分配器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 预分配 `capacity` 字节（仍可继续增长）。
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            cursor: 0,
        }
    }

    /// 分配 `len` 字节，起始地址按 `align` 对齐。
    ///
    /// 容量不足时按需增长：`Region` 的偏移与长度不会因为增长而失效。
    pub fn alloc(&mut self, len: usize, align: usize) -> Result<Region> {
        let offset = align_up(self.cursor, align)?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::new(Code::Internal, "分配长度溢出").with("mem::Arena::alloc"))?;
        if end > self.bytes.len() {
            self.bytes.resize(end, 0);
        }
        self.cursor = end;
        Ok(Region { offset, len })
    }

    /// 只读视图；`region` 超出已分配范围时返回 `None`（不 panic）。
    pub fn slice(&self, region: Region) -> Option<&[u8]> {
        self.bytes.get(region.offset..region.end())
    }

    /// 可变视图；`region` 超出已分配范围时返回 `None`。
    pub fn slice_mut(&mut self, region: Region) -> Option<&mut [u8]> {
        self.bytes.get_mut(region.offset..region.end())
    }

    /// 已分配到的偏移（下一段分配的起点候选）。
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// 当前容量（按需增长后的字节数）。
    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_must_be_a_power_of_two() {
        assert_eq!(align_up(1, 4), Ok(4));
        assert_eq!(align_up(4, 4), Ok(4));
        assert_eq!(align_up(5, 8), Ok(8));
        assert_eq!(align_up(0, 8), Ok(0));
        assert_eq!(align_up(1, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(align_up(1, 3).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn allocations_are_aligned_and_do_not_overlap() {
        let mut arena = Arena::new();
        let a = arena.alloc(3, 8).expect("分配");
        let b = arena.alloc(5, 8).expect("分配");
        assert_eq!((a.offset(), a.len()), (0, 3));
        assert_eq!((b.offset(), b.len()), (8, 5));
        assert_eq!(arena.cursor(), 13);
        assert_eq!(arena.capacity(), 13);
        assert!(arena.slice(a).is_some());
        assert_eq!(arena.slice(b).map(<[u8]>::len), Some(5));
    }

    #[test]
    fn an_empty_allocation_is_allowed_and_advances_nothing() {
        let mut arena = Arena::new();
        let region = arena.alloc(0, 4).expect("分配");
        assert!(region.is_empty());
        assert_eq!(arena.cursor(), 0);
        assert_eq!(arena.slice(region), Some(&[][..]));
    }

    #[test]
    fn a_view_is_writable_and_out_of_range_is_none() {
        let mut arena = Arena::with_capacity(4);
        let region = arena.alloc(4, 4).expect("分配");
        arena.slice_mut(region).expect("视图")[0] = 7;
        assert_eq!(arena.slice(region), Some(&[7, 0, 0, 0][..]));
        assert_eq!(arena.slice(Region::new(2, 99)), None);
    }
}
