//! 分页 KV 缓存：[`Page`]、[`Table`]、[`Cache`]（页号簿记）与 [`Pager`]（页池 + 页里的数据）。
//!
//! **拥有**：`Page`、`Table`、`Cache`、`Pager`。**不拥有**：注意力算子（`nn` 的层会绑定它）、
//! 多层 KV 的行内布局（层的偏移表归模型：本层只按"一个 token 一行"寻址）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`engine::Scheduler`（本模块并入后与它同 crate）。
//!
//! 簿记与数据分开是两个类型，而不是一个类型里的两个字段：
//!
//! - [`Cache`] 只回答"哪些页被借出、各被几个持有者引用"，[`Table`] 只回答"这条序列占了哪些页"
//!   ——两者都不碰页里的数；
//! - [`Pager`] 把页号与页背后的缓冲绑在一起：借出一页就同时有了可写的数据块，最后一个持有者
//!   归还一页才同时把这块数据清零。于是"页被复用时看到上一条序列的字节"这件事没有发生的路径
//!   ——清零点只有一处。
//!
//! 连续批的关键就是把页的粒度与序列长度解耦，所以页表必须是显式类型——不能藏在向量里靠约定。
//! [`Pager`] 因此也不持有页表：它只认 [`Slot`]，"第几页第几格"由页表算好再交给它。
//!
//! 页表随序列增长：[`Table::grow`] 抬高容量，页仍从 [`Cache`] 取、由调用方 [`Table::push`]。
//! 于是"这条序列还要几页"是一次显式计算，而不是等 KV 缓冲越界才暴露。
//!
//! [`Cache`] 给每页记引用计数：`alloc` 置 1，[`Cache::share`] 在另一张表里采纳同一前缀页时 +1，
//! [`Cache::release`] 减 1；计数归 0 才把页退回空闲池。重复归还、越界页与计数下溢都按 `Invalid`
//! 报错，O(1) 命中。一次 `finish` 要归还整条序列的页，线性查"这页是否已经在空闲表里"会把回收
//! 变成 O(总页数 × 页数)（页数即待归还的页）。
//!
//! 共享期内前缀行是只读的：模型只写前缀之后的新页；页数据在最后一次 `release` 之前不清零，
//! 计数归零时才回到空闲池并清零。前缀匹配的准入策略（哪条请求复用哪段前缀）、LRU 淘汰与跨序列
//! 向量复用都不在本类型内——这里只落共享句柄、计数与回收判定原语。
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

use crate::invalid;
use diagnostic::Result;
use tensor::Shape;

/// 页池上的页内数据（实现见 `crate::pager`）：给页表要页、写行、读行。
pub use crate::pager::Pager;

/// 满块前缀树（实现见 [`crate::radix`]）：页池只记「谁在看」，树记「哪段前缀对应哪一页」。
pub use crate::radix::Radix;

#[cfg(test)]
#[path = "../tests/kv.rs"]
mod tests;

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
            return Err(invalid("engine::kv::Table::new", "页表容量不能为 0"));
        }
        Ok(Self {
            pages: Vec::with_capacity(capacity),
            capacity,
        })
    }

    /// 追加一页；已满时报错，而不是静默丢弃（丢页等于丢上下文）。
    pub fn push(&mut self, page: Page) -> Result<()> {
        if self.pages.len() == self.capacity {
            return Err(invalid(
                "engine::kv::Table::push",
                format!("页表已满（容量 {}）", self.capacity),
            ));
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
    pub fn grow(&mut self, extra: usize) -> Result<()> {
        let capacity = self.capacity.checked_add(extra).ok_or_else(|| {
            invalid(
                "engine::kv::Table::grow",
                format!("页表容量 {} + {extra} 溢出", self.capacity),
            )
        })?;
        self.pages.reserve(extra);
        self.capacity = capacity;
        Ok(())
    }

    /// 全部页，按 token 顺序。
    pub fn pages(&self) -> &[Page] {
        &self.pages
    }

    /// 第 `idx` 页的编号。
    pub fn get(&self, idx: usize) -> Option<u32> {
        self.pages.get(idx).map(Page::id)
    }

    /// 第 `idx` 个 token 落在哪一页的哪一格：`page_size` 来自页池（页表自己不存页宽）。
    ///
    /// 这是“KV 数据写到哪里”的唯一算术：页表只记页号，token 在页内的偏移由页宽除出来。
    /// 把这条除法写在调用方，就会在“本页够不够”与“下一页从哪个 token 开始”两处各写一次。
    ///
    /// # Errors
    ///
    /// `page_size` 为 0，或第 `idx` 个 token 还没有页（页表只覆盖已分配的那几页）。
    pub fn slot(&self, idx: usize, page_size: usize) -> Result<Slot> {
        if page_size == 0 {
            return Err(invalid("engine::kv::Table::slot", "页宽不能为 0"));
        }
        let page = self.pages.get(idx / page_size).ok_or_else(|| {
            invalid(
                "engine::kv::Table::slot",
                format!(
                    "第 {idx} 个 token 还没有页：页表只有 {} 页",
                    self.pages.len()
                ),
            )
        })?;
        Ok(Slot {
            page: *page,
            offset: idx % page_size,
        })
    }

    /// 从尾部去掉页直到只剩 `keep` 页，返回被去掉的页（按 token 顺序）。
    ///
    /// 回收方需要“页号”本身才能把它们还给页池，所以这里不代劳 [`Cache::release`]：
    /// 页表不知道页池是谁的。容量不变——页表还会再长回去（回滚一次生成不需要重新分配）。
    ///
    /// # Errors
    ///
    /// `keep` 大于当前页数。
    pub fn truncate(&mut self, keep: usize) -> Result<Vec<Page>> {
        if keep > self.pages.len() {
            return Err(invalid(
                "engine::kv::Table::truncate",
                format!("只能删到不超过 {} 页，收到 {keep}", self.pages.len()),
            ));
        }
        Ok(self.pages.split_off(keep))
    }
}

/// 一个 token 在 KV 缓存里的位置：页号 + 页内偏移。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slot {
    pub(super) page: Page,
    pub(super) offset: usize,
}

impl Slot {
    /// 所在页。
    pub fn page(&self) -> Page {
        self.page
    }

    /// 页内第几格。
    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// 页号分配器：固定 `total` 页，每页 `page_size` 个 token 位。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache {
    page_size: usize,
    free: Vec<Page>,
    refs: Vec<u32>,
    in_use: usize,
}

impl Cache {
    /// 建一个 `total` 页、每页 `page_size` 个 token 的页池。
    pub fn new(total: usize, page_size: usize) -> Result<Self> {
        if total == 0 || page_size == 0 {
            return Err(invalid(
                "engine::kv::Cache::new",
                format!("页数与页宽都必须非零（total = {total}，page_size = {page_size}）"),
            ));
        }
        let total_pages = u32::try_from(total)
            .map_err(|_| invalid("engine::kv::Cache::new", format!("页数 {total} 超出 u32")))?;
        // 倒序入池：`alloc` 从尾部取，这样第 0 页先被用上，调试输出更稳定。
        let free = (0..total_pages).rev().map(Page::new).collect();
        Ok(Self {
            page_size,
            free,
            refs: vec![0; total],
            in_use: 0,
        })
    }

    /// 取一页；池空时返回 `None`（调用方决定是拒绝请求还是排队）。
    pub fn alloc(&mut self) -> Option<Page> {
        let page = self.free.pop()?;
        self.refs[page.id() as usize] = 1;
        self.in_use += 1;
        Some(page)
    }

    /// 把一页再交给另一张页表引用：前缀页被多条序列共享时的采纳入口。
    ///
    /// 只递增计数，不从空闲池取、也不动页里的数据。采纳方把同一 [`Page`] 推进自己的
    /// [`Table`]；独占新页仍走 [`Cache::alloc`]，两条路径分开。
    ///
    /// # Errors
    ///
    /// 页越界，或不在借出状态（从未分配、已经归还）。
    pub fn share(&mut self, page: Page) -> Result<()> {
        let Some(count) = self.refs.get_mut(page.id() as usize) else {
            return Err(invalid(
                "engine::kv::Cache::share",
                format!("页 {} 不在页池范围内", page.id()),
            ));
        };
        if *count == 0 {
            return Err(invalid(
                "engine::kv::Cache::share",
                format!(
                    "页 {} 不在借出状态：要么被重复共享，要么从未分配",
                    page.id()
                ),
            ));
        }
        *count += 1;
        Ok(())
    }

    /// 归还一页：越界、重复归还或计数下溢都是调用方的 bug，直接报错。
    pub fn release(&mut self, page: Page) -> Result<bool> {
        let Some(count) = self.refs.get_mut(page.id() as usize) else {
            return Err(invalid(
                "engine::kv::Cache::release",
                format!("页 {} 不在页池范围内", page.id()),
            ));
        };
        if *count == 0 {
            return Err(invalid(
                "engine::kv::Cache::release",
                format!(
                    "页 {} 不在借出状态：要么被重复归还，要么从未分配",
                    page.id()
                ),
            ));
        }
        *count -= 1;
        if *count > 0 {
            return Ok(false);
        }
        self.free.push(page);
        self.in_use -= 1;
        Ok(true)
    }

    /// 页宽（每页多少 token）。
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// 页总数。
    pub fn total(&self) -> usize {
        self.refs.len()
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

    /// 这一页当前是否被借出（引用计数 > 0）。
    pub fn is_live(&self, page: Page) -> bool {
        self.reference_count(page) > 0
    }

    /// 这一页当前的引用计数；越界页返回 0。
    pub fn reference_count(&self, page: Page) -> u32 {
        self.refs.get(page.id() as usize).copied().unwrap_or(0)
    }
}
