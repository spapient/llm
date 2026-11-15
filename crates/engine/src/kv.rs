//! 分页 KV 缓存的簿记：[`Page`]、[`Table`]、[`Cache`]。
//!
//! **拥有**：`Page`、`Table`、`Cache`。**不拥有**：页里的数据（那是 `storage` 分配的缓冲，形状由
//! [`Cache::block_shape`] 给出）、注意力算子（`cutile` 的 `op` 会声明它）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`engine::Scheduler`（本模块并入后与它同 crate）。
//!
//! 这里只做"谁占了哪些页"的账：分配、释放、每个序列的页表。连续批的关键就是把页的粒度与
//! 序列长度解耦，所以页表必须是显式类型——不能藏在向量里靠约定。
//!
//! 页表随序列增长：[`Table::grow`] 抬高容量，页仍从 [`Cache`] 取、由调用方 [`Table::push`]。
//! 于是"这条序列还要几页"是一次显式计算，而不是等 KV 缓冲越界才暴露。
//!
//! [`Cache::release`] 用每页一字节的存活位图判重复归还（O(1)）：一次 `finish` 要归还整条序列的页，
//! 线性查"这页是否已经在空闲表里"会把回收变成 O(总页数 × 页数)（页数即待归还的页）。
//!
//! # 例子
//!
//! ```
//! use llm_engine::kv::Cache;
//!
//! let mut cache = Cache::new(2, 16).expect("页池");
//! let page = cache.alloc().expect("还有空闲页");
//! let mut table = llm_engine::kv::Table::new(2).expect("页表");
//! table.push(page).expect("未满");
//! assert_eq!(table.len(), 1);
//! cache.release(page).expect("归还");
//! assert_eq!(cache.in_use(), 0);
//! ```

use diagnostic::{Code, Error, Result};
use tensor::Shape;

/// 一页 KV 缓存的编号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Page {
    id: u32,
}

impl Page {
    /// 由编号构造（正常路径是 [`Cache::alloc`]）。
    pub fn new(id: u32) -> Self {
        Self { id }
    }

    /// 编号。
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// 一个序列的页表：按 token 顺序记录它占了哪些页。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pages: Vec<Page>,
    capacity: usize,
}

impl Table {
    /// 建一个最多 `capacity` 页的页表；容量为 0 直接报错（那等于无法生成任何 token）。
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(
                Error::new(Code::Invalid, "页表容量不能为 0").with("engine::kv::Table::new")
            );
        }
        Ok(Self {
            pages: Vec::with_capacity(capacity),
            capacity,
        })
    }

    /// 追加一页；已满时报错，而不是静默丢弃（丢页等于丢上下文）。
    pub fn push(&mut self, page: Page) -> Result<()> {
        if self.pages.len() == self.capacity {
            return Err(
                Error::new(Code::Invalid, format!("页表已满（容量 {}）", self.capacity))
                    .with("engine::kv::Table::push"),
            );
        }
        self.pages.push(page);
        Ok(())
    }

    /// 已用页数。
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// 是否一页未用。
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// 容量。
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 是否已满。
    pub fn is_full(&self) -> bool {
        self.pages.len() == self.capacity
    }

    /// 把容量抬高 `extra` 页（页表随序列增长）。
    ///
    /// 只改容量，不分配页：页来自 [`Cache`]，仍由调用方 [`Table::push`]。容量上限溢出时报错，
    /// 而不是回绕成一个"能装很少页"的页表。
    pub fn grow(&mut self, extra: usize) -> Result<()> {
        let capacity = self.capacity.checked_add(extra).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("页表容量 {} + {extra} 溢出", self.capacity),
            )
            .with("engine::kv::Table::grow")
        })?;
        self.pages.reserve(extra);
        self.capacity = capacity;
        Ok(())
    }

    /// 全部页，按 token 顺序。
    pub fn pages(&self) -> &[Page] {
        &self.pages
    }

    /// 第 `index` 页的编号。
    pub fn get(&self, index: usize) -> Option<u32> {
        self.pages.get(index).map(Page::id)
    }
}

/// 页池：固定 `total` 页，每页 `page_size` 个 token 的 KV。
///
/// `live` 是按页编号索引的存活位图：`release` 因此能 O(1) 判"这页是否已经在空闲表里"。
/// 位图是稠密的，每页一字节；相对页里的 KV 数据可以忽略。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache {
    page_size: usize,
    free: Vec<Page>,
    live: Vec<bool>,
    in_use: usize,
}

impl Cache {
    /// 建一个 `total` 页、每页 `page_size` 个 token 的页池。
    pub fn new(total: usize, page_size: usize) -> Result<Self> {
        if total == 0 || page_size == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("页数与页宽都必须非零（total = {total}，page_size = {page_size}）"),
            )
            .with("engine::kv::Cache::new"));
        }
        let total_pages = u32::try_from(total).map_err(|_| {
            Error::new(Code::Invalid, format!("页数 {total} 超出 u32"))
                .with("engine::kv::Cache::new")
        })?;
        // 倒序入池：`alloc` 从尾部取，这样第 0 页先被用上，调试输出更稳定。
        let free = (0..total_pages).rev().map(Page::new).collect();
        Ok(Self {
            page_size,
            free,
            live: vec![false; total],
            in_use: 0,
        })
    }

    /// 取一页；池空时返回 `None`（调用方决定是拒绝请求还是排队）。
    pub fn alloc(&mut self) -> Option<Page> {
        let page = self.free.pop()?;
        self.live[page.id() as usize] = true;
        self.in_use += 1;
        Some(page)
    }

    /// 归还一页：越界或重复归还是调用方的 bug，直接报错。
    ///
    /// 两个判定都查存活位图：越界是 `None`，重复归还是 `Some(false)`。
    pub fn release(&mut self, page: Page) -> Result<()> {
        match self.live.get_mut(page.id() as usize) {
            None => Err(
                Error::new(Code::Invalid, format!("页 {} 不在页池范围内", page.id()))
                    .with("engine::kv::Cache::release"),
            ),
            Some(live) if !*live => Err(Error::new(
                Code::Invalid,
                format!(
                    "页 {} 不在借出状态：要么被重复归还，要么从未分配",
                    page.id()
                ),
            )
            .with("engine::kv::Cache::release")),
            Some(live) => {
                *live = false;
                self.free.push(page);
                self.in_use -= 1;
                Ok(())
            }
        }
    }

    /// 页宽（每页多少 token）。
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// 页总数。
    pub fn total(&self) -> usize {
        self.live.len()
    }

    /// 已占用页数。
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// 空闲页数。
    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// 一页 KV 的逻辑形状：`[rows, page_size]`。
    pub fn block_shape(&self, rows: usize) -> Shape {
        Shape::new([rows, self.page_size])
    }
}

#[cfg(test)]
#[path = "../tests/kv_tests.rs"]
mod tests;
