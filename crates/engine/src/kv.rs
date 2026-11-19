//! 分页 KV 缓存：[`Page`]、[`Table`]、[`Cache`]（页号簿记）与 [`Pager`]（页池 + 页里的数据）。
//!
//! **拥有**：`Page`、`Table`、`Cache`、`Pager`。**不拥有**：注意力算子（`cutile` 的 `op` 会声明它）、
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

    /// 第 `index` 个 token 落在哪一页的哪一格：`page_size` 来自页池（页表自己不存页宽）。
    ///
    /// 这是“KV 数据写到哪里”的唯一算术：页表只记页号，token 在页内的偏移由页宽除出来。
    /// 把这条除法写在调用方，就会在“本页够不够”与“下一页从哪个 token 开始”两处各写一次。
    ///
    /// # Errors
    ///
    /// `page_size` 为 0，或第 `index` 个 token 还没有页（页表只覆盖已分配的那几页）。
    pub fn slot(&self, index: usize, page_size: usize) -> Result<Slot> {
        if page_size == 0 {
            return Err(Error::new(Code::Invalid, "页宽不能为 0").with("engine::kv::Table::slot"));
        }
        let page = self.pages.get(index / page_size).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!(
                    "第 {index} 个 token 还没有页：页表只有 {} 页",
                    self.pages.len()
                ),
            )
            .with("engine::kv::Table::slot")
        })?;
        Ok(Slot {
            page: *page,
            offset: index % page_size,
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
            return Err(Error::new(
                Code::Invalid,
                format!("只能删到不超过 {} 页，收到 {keep}", self.pages.len()),
            )
            .with("engine::kv::Table::truncate"));
        }
        Ok(self.pages.split_off(keep))
    }
}

/// 一个 token 在 KV 缓存里的位置：页号 + 页内偏移。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slot {
    page: Page,
    offset: usize,
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

/// 页池 + 页里的数据：一页 = `page_size` 行、每行 `width` 个 `f32`。
///
/// 行宽是"一个 token 位置在模型里的 KV 宽度"（例如 `2 · kv_heads · head_dim`），由装载模型的
/// 一侧算好传进来：页池不知道层与头，也不该替模型猜。多层模型用 [`Pager::row_mut`] 拿整行，
/// 再按自己的层偏移切片——层的偏移表归模型，页池不复制第二份。
///
/// 布局是 `[页][页内偏移][行内元素]`，与 [`Slot`] 的寻址算术一致：一页的元素连续存放，
/// 所以"第几页第几个 token"只做一次乘加，不需要页内链表。整池在构造时一次分配完（页池就是
/// 这些元素），之后借出与归还都不再碰分配器——KV 不该在生成到一半时因为一次 `Vec` 扩容而抖动。
///
/// 数据是 `Vec<f32>`：本类型是 CPU 页池。设备后端的同类对象在边界 crate，接口形状相同
/// （[`Slot`] → 一段可写映射），本层不引入设备句柄。
#[derive(Debug, Clone, PartialEq)]
pub struct Pager {
    cache: Cache,
    width: usize,
    data: Vec<f32>,
}

impl Pager {
    /// 由页池与行宽构造；整池一次分配并清零。
    ///
    /// # Errors
    ///
    /// 行宽为 0（那是"不存 KV"，不该有一个静默的空池）、元素总数溢出、或内存不足。
    pub fn new(cache: Cache, width: usize) -> Result<Self> {
        if width == 0 {
            return Err(
                Error::new(Code::Invalid, "KV 行宽不能为 0：页池需要能放下至少一个元素")
                    .with("engine::kv::Pager::new"),
            );
        }
        let elements = cache
            .total()
            .checked_mul(cache.page_size())
            .and_then(|positions| positions.checked_mul(width))
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "KV 页池容量溢出：{} 页 × {} 行 × 行宽 {width}",
                        cache.total(),
                        cache.page_size()
                    ),
                )
                .with("engine::kv::Pager::new")
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(elements).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("KV 页池需要 {elements} 个 f32：超出可用内存"),
            )
            .with("engine::kv::Pager::new")
        })?;
        data.resize(elements, 0.0);
        Ok(Self { cache, width, data })
    }

    /// 行宽（一个 token 位置在模型里的 KV 宽度）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 借出一页；池空时返回 `None`。
    pub fn alloc(&mut self) -> Option<Page> {
        self.cache.alloc()
    }

    /// 采纳一页给另一张页表共享：只递增引用计数，不动页里的数据。
    ///
    /// 与 [`Pager::alloc`] 分开：新页从池里取（计数置 1），前缀共享页只经这里 +1。
    ///
    /// # Errors
    ///
    /// 页越界，或不在借出状态。
    pub fn share(&mut self, page: Page) -> Result<()> {
        self.cache.share(page)
    }

    /// 归还一页；只有这是最后一个持有者时才**清零它的数据**。
    ///
    /// 清零点只有这一处：借出一页时不做第二次清零，因为页只可能经这里回到池里。先让页池
    /// 判定越界与重复归还，再清数据——归还失败时那页仍然是活的，数据不能动；仍被共享时
    /// 前缀行必须保持可读，也不能清零。
    ///
    /// # Errors
    ///
    /// 页不在池里，或不在借出状态。
    pub fn release(&mut self, page: Page) -> Result<()> {
        let last = self.cache.release(page)?;
        if last {
            self.clear(page);
        }
        Ok(())
    }

    /// 第 `slot` 个位置的那一行（只读）。
    ///
    /// # Errors
    ///
    /// 页已归还/不存在，或页内偏移超出页宽。
    pub fn row(&self, slot: Slot) -> Result<&[f32]> {
        let span = self.span(slot)?;
        Ok(&self.data[span])
    }

    /// 第 `slot` 个位置的那一行（可写）：多层模型按自己的层偏移切它。
    ///
    /// # Errors
    ///
    /// 同 [`Pager::row`]。
    pub fn row_mut(&mut self, slot: Slot) -> Result<&mut [f32]> {
        let span = self.span(slot)?;
        Ok(&mut self.data[span])
    }

    /// 把一整行写成 `values`：长度必须恰好等于行宽。
    ///
    /// # Errors
    ///
    /// 长度不符（半行会让同一页里的行错位），或同 [`Pager::row`]。
    pub fn write(&mut self, slot: Slot, values: &[f32]) -> Result<()> {
        if values.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "KV 行宽 {} 与实际写入的 {} 个元素不符",
                    self.width,
                    values.len()
                ),
            )
            .with("engine::kv::Pager::write"));
        }
        self.row_mut(slot)?.copy_from_slice(values);
        Ok(())
    }

    /// 页池（只读）：页宽与占用由它回答，数据不在这里。
    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    /// 页宽（每页多少 token 位置）。
    pub fn page_size(&self) -> usize {
        self.cache.page_size()
    }

    /// 页总数。
    pub fn total(&self) -> usize {
        self.cache.total()
    }

    /// 已借出页数。
    pub fn in_use(&self) -> usize {
        self.cache.in_use()
    }

    /// 空闲页数。
    pub fn free_pages(&self) -> usize {
        self.cache.free_pages()
    }

    /// 一行的元素区间；页必须活着，偏移必须落在页内。
    fn span(&self, slot: Slot) -> Result<std::ops::Range<usize>> {
        if !self.cache.is_live(slot.page()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "页 {} 不在借出状态：它已经归还（或从未分配），这一行不再属于任何序列",
                    slot.page().id()
                ),
            )
            .with("engine::kv::Pager"));
        }
        if slot.offset() >= self.cache.page_size() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "页内偏移 {} 超出页宽 {}",
                    slot.offset(),
                    self.cache.page_size()
                ),
            )
            .with("engine::kv::Pager"));
        }
        let start = self.start(slot.page(), slot.offset());
        Ok(start..start + self.width)
    }

    /// 线性起点：`(页号 × 页宽 + 页内偏移) × 行宽`。
    ///
    /// 只在这里写一次：`span` 与清零共用它，两处各写一遍就会在"清到哪"与"读到哪"之间漂。
    fn start(&self, page: Page, offset: usize) -> usize {
        (page.id() as usize * self.cache.page_size() + offset) * self.width
    }

    /// 把一页的全部行清零（归还时调用）。
    fn clear(&mut self, page: Page) {
        let start = self.start(page, 0);
        let end = start + self.cache.page_size() * self.width;
        self.data[start..end].fill(0.0);
    }
}

/// 页号分配器：固定 `total` 页，每页 `page_size` 个 token 位。
///
/// `refs` 是按页编号索引的引用计数：`release` 因此能 O(1) 判"这页是否已经在空闲表里"，
/// 也能回答"它还被几个持有者引用"。计数是稠密的，每页一个 `u32`；相对页里的 KV 数据可以忽略。
/// 页里的数据不在本类型：那是 [`Pager`] 的事，本类型只管"哪些页被借出、被几个人借"。
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
            return Err(
                Error::new(Code::Invalid, format!("页 {} 不在页池范围内", page.id()))
                    .with("engine::kv::Cache::share"),
            );
        };
        if *count == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "页 {} 不在借出状态：要么被重复共享，要么从未分配",
                    page.id()
                ),
            )
            .with("engine::kv::Cache::share"));
        }
        *count += 1;
        Ok(())
    }

    /// 归还一页：越界、重复归还或计数下溢都是调用方的 bug，直接报错。
    ///
    /// 计数减 1，**只有归 0** 才把页退回空闲池；仍被引用时页保持存活、数据不动。返回
    /// `true` 表示这是最后一个持有者（调用方据此清零页数据），`false` 表示仍有共享者。
    pub fn release(&mut self, page: Page) -> Result<bool> {
        let Some(count) = self.refs.get_mut(page.id() as usize) else {
            return Err(
                Error::new(Code::Invalid, format!("页 {} 不在页池范围内", page.id()))
                    .with("engine::kv::Cache::release"),
            );
        };
        if *count == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "页 {} 不在借出状态：要么被重复归还，要么从未分配",
                    page.id()
                ),
            )
            .with("engine::kv::Cache::release"));
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
    ///
    /// [`Pager`] 用它拒绝"往已归还的页里写"：[`Slot`] 是可以在 `finish` 之后继续存在的值，
    /// 而它指向的页此时可能已经属于另一条序列——写进去就是一次静默的串数据。
    pub fn is_live(&self, page: Page) -> bool {
        self.reference_count(page) > 0
    }

    /// 这一页当前的引用计数；越界页返回 0。
    pub fn reference_count(&self, page: Page) -> u32 {
        self.refs.get(page.id() as usize).copied().unwrap_or(0)
    }
}

/// 满块前缀树（实现见 [`crate::radix`]）：页池只记「谁在看」，树记「哪段前缀对应哪一页」。
pub use crate::radix::Radix;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_are_handed_out_in_order_and_reused() {
        let mut cache = Cache::new(2, 4).expect("页池");
        let first = cache.alloc().expect("第一页");
        let second = cache.alloc().expect("第二页");
        assert_eq!((first.id(), second.id()), (0, 1));
        assert!(cache.alloc().is_none(), "池空必须返回 None");
        cache.release(first).expect("归还");
        assert_eq!(cache.alloc().map(|page| page.id()), Some(0));
    }

    #[test]
    fn double_release_and_out_of_range_are_rejected() {
        let mut cache = Cache::new(1, 4).expect("页池");
        let page = cache.alloc().expect("分配");
        cache.release(page).expect("归还");
        assert_eq!(cache.release(page).unwrap_err().code(), Code::Invalid);
        assert_eq!(
            cache.release(Page::new(9)).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn a_table_refuses_to_drop_pages_silently() {
        let mut table = Table::new(1).expect("页表");
        assert!(table.is_empty());
        table.push(Page::new(0)).expect("未满");
        assert!(table.is_full());
        assert!(table.push(Page::new(1)).is_err(), "满页表必须报错");
        assert_eq!(table.get(0), Some(0));
        assert_eq!(table.get(1), None);
        assert_eq!(Table::new(0).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_zero_sized_pool_is_rejected() {
        assert_eq!(Cache::new(0, 16).unwrap_err().code(), Code::Invalid);
        assert_eq!(Cache::new(4, 0).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_block_shape_matches_the_page_width() {
        let cache = Cache::new(4, 16).expect("页池");
        assert_eq!(cache.block_shape(2).dims(), [2, 16]);
        assert_eq!(cache.total(), 4);
        assert_eq!(cache.free_pages(), 4);
    }

    #[test]
    fn a_table_can_grow_but_never_past_its_capacity() {
        let mut table = Table::new(1).expect("页表");
        table.push(Page::new(0)).expect("未满");
        assert!(table.push(Page::new(1)).is_err(), "满页表仍然拒绝追加");
        table.grow(1).expect("抬高容量");
        assert_eq!(table.capacity(), 2);
        table.push(Page::new(1)).expect("容量已抬高");
        assert!(table.push(Page::new(2)).is_err(), "抬容量不能顺手放开越界");
        table.grow(0).expect("零页增长是恒等");
        assert_eq!(table.capacity(), 2);
    }

    /// 位图判定要同时认得出"从没分配过"与"已经还过"：两者都不是合法的归还。
    #[test]
    fn releasing_a_page_that_was_never_allocated_is_rejected() {
        let mut cache = Cache::new(2, 4).expect("页池");
        assert_eq!(
            cache.release(Page::new(1)).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(cache.in_use(), 0, "失败不能改变占用");
        assert_eq!(cache.free_pages(), 2, "失败不能把页塞进空闲表");
    }

    /// 稳态：整池分配、整池归还重复多轮，账目与页编号都不漂。
    #[test]
    fn allocate_and_release_cycles_keep_the_accounts_balanced() {
        let mut cache = Cache::new(8, 4).expect("页池");
        let mut held = Vec::new();
        for round in 0..1000 {
            while let Some(page) = cache.alloc() {
                held.push(page);
            }
            assert_eq!(
                (cache.in_use(), cache.free_pages()),
                (8, 0),
                "第 {round} 轮"
            );
            let mut ids: Vec<u32> = held.iter().map(Page::id).collect();
            ids.sort_unstable();
            assert_eq!(
                ids,
                (0..8).collect::<Vec<u32>>(),
                "第 {round} 轮：同一个页不能发两次"
            );
            for page in held.drain(..) {
                cache.release(page).expect("归还");
            }
            assert_eq!(
                (cache.in_use(), cache.free_pages()),
                (0, 8),
                "第 {round} 轮"
            );
        }
        assert_eq!(cache.total(), 8, "页总数与分配历史无关");
    }

    /// 寻址是唯一的除法：第 `index` 个 token 落在 `index / page_size` 页的 `index % page_size` 格。
    #[test]
    fn a_token_index_maps_to_one_page_and_one_offset() {
        let mut table = Table::new(4).expect("页表");
        for id in [7, 3, 11] {
            table.push(Page::new(id)).expect("追加");
        }
        assert_eq!(
            table.slot(0, 4).expect("第 0 个"),
            Slot {
                page: Page::new(7),
                offset: 0
            }
        );
        assert_eq!(table.slot(5, 4).expect("第 5 个").page().id(), 3);
        assert_eq!(table.slot(5, 4).expect("第 5 个").offset(), 1);
        assert_eq!(table.slot(11, 4).expect("第 11 个").offset(), 3);
        // 页表只盖住已分配的 3 页：第 12 个 token 还没有落点。
        assert_eq!(table.slot(12, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(table.slot(0, 0).unwrap_err().code(), Code::Invalid);
    }

    /// 截断只摘页、不动容量：回滚一次生成之后页表还要长回去。
    #[test]
    fn truncating_returns_the_pages_it_dropped() {
        let mut table = Table::new(4).expect("页表");
        for id in [0, 1, 2] {
            table.push(Page::new(id)).expect("追加");
        }
        assert_eq!(
            table.truncate(1).expect("截断"),
            [Page::new(1), Page::new(2)]
        );
        assert_eq!(table.len(), 1);
        assert_eq!(table.capacity(), 4, "容量不变");
        assert!(table.truncate(2).is_err(), "不能截到比现有页数更多的位置");
        assert_eq!(table.truncate(1).expect("重复截断是恒等"), []);
    }

    /// 一页 `page_size` 行、每行 `width` 个元素：借出就有一块可写的行，归还就清零。
    ///
    /// 这一条是页池存在的理由：`Cache` 只发页号时，"这页被复用"与"页里还留着上一条序列的
    /// 字节"可以同时成立，而两种错都不会让任何断言失败——只有读回旧值时才暴露。
    #[test]
    fn a_pager_writes_rows_and_clears_them_on_release() {
        let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 3).expect("行宽");
        let page = pager.alloc().expect("第一页");
        let first = Slot { page, offset: 0 };
        let second = Slot { page, offset: 1 };
        pager.write(first, &[1.0, 2.0, 3.0]).expect("写第一行");
        pager.write(second, &[4.0, 5.0, 6.0]).expect("写第二行");
        assert_eq!(pager.row(first).expect("读第一行"), [1.0, 2.0, 3.0]);
        assert_eq!(pager.row(second).expect("读第二行"), [4.0, 5.0, 6.0]);

        pager.release(page).expect("归还");
        assert_eq!(pager.in_use(), 0);
        let again = pager.alloc().expect("同一页再借出来");
        assert_eq!(again, page, "空闲表按页号复用");
        assert_eq!(
            pager.row(first).expect("读第一行"),
            [0.0, 0.0, 0.0],
            "复用页不能带出上一条序列的字节"
        );
        assert_eq!(pager.row(second).expect("读第二行"), [0.0, 0.0, 0.0]);
    }

    /// 行在两个维度上都校验：页必须是活的，偏移必须落在页内，长度必须恰好是行宽。
    #[test]
    fn a_pager_refuses_stale_slots_and_partial_rows() {
        let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 3).expect("行宽");
        let page = pager.alloc().expect("页");
        let row = Slot { page, offset: 0 };
        assert_eq!(
            pager.write(row, &[1.0, 2.0]).unwrap_err().code(),
            Code::Invalid,
            "半行会让同一页里的后续行错位"
        );
        assert_eq!(
            pager.row(Slot { page, offset: 2 }).unwrap_err().code(),
            Code::Invalid,
            "偏移等于页宽已经越界"
        );
        pager.release(page).expect("归还");
        assert_eq!(
            pager.row(row).unwrap_err().code(),
            Code::Invalid,
            "已归还的页不是合法地址：这张页可能已经属于别人"
        );
        assert_eq!(
            pager.release(page).unwrap_err().code(),
            Code::Invalid,
            "重复归还由页池判定"
        );
    }

    /// 行宽为零是"不存 KV"，不该有一个静默的空池。
    #[test]
    fn a_zero_width_pager_is_rejected() {
        let cache = Cache::new(2, 4).expect("页池");
        let error = Pager::new(cache, 0).expect_err("行宽为零");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// 整池一次分配：容量与页池相等，跨页的行不共享元素。
    #[test]
    fn pages_do_not_share_rows() {
        let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 1).expect("行宽");
        let first = pager.alloc().expect("第一页");
        let second = pager.alloc().expect("第二页");
        pager
            .write(
                Slot {
                    page: first,
                    offset: 1,
                },
                &[7.0],
            )
            .expect("写");
        assert_eq!(
            pager
                .row(Slot {
                    page: second,
                    offset: 1
                })
                .expect("读"),
            [0.0],
            "另一页的同一格不受影响"
        );
        assert_eq!((pager.total(), pager.page_size()), (2, 2));
    }

    /// 两表共引一页：第一张表归还后页仍存活、前缀行未被清零；最后一个持有者归还才回池并清零。
    #[test]
    fn a_shared_page_survives_the_first_table_releasing_it() {
        let mut pager = Pager::new(Cache::new(2, 2).expect("页池"), 3).expect("行宽");
        let page = pager.alloc().expect("第一页");
        let slot = Slot { page, offset: 0 };
        pager.write(slot, &[1.0, 2.0, 3.0]).expect("写前缀行");

        pager.share(page).expect("另一张表采纳前缀页");
        assert_eq!(pager.cache().reference_count(page), 2);

        pager.release(page).expect("第一张表归还");
        assert!(pager.cache().is_live(page), "还有共享者，页仍存活");
        assert_eq!(
            pager.row(slot).expect("读"),
            [1.0, 2.0, 3.0],
            "共享期内前缀行只读，不能清零"
        );
        assert_eq!(pager.in_use(), 1);

        pager.release(page).expect("最后一个持有者归还");
        assert!(!pager.cache().is_live(page));
        assert_eq!(pager.in_use(), 0);
        let again = pager.alloc().expect("同一页再借出来");
        assert_eq!(again, page, "空闲表按页号复用");
        assert_eq!(
            pager.row(slot).expect("读"),
            [0.0, 0.0, 0.0],
            "归零时才清零"
        );
    }

    /// 引用计数口径：共享两次、释放两次后仍剩一个持有者；归零时才回空闲池。
    #[test]
    fn reference_count_is_accounted_across_share_and_release() {
        let mut cache = Cache::new(2, 4).expect("页池");
        let page = cache.alloc().expect("分配");
        assert_eq!(cache.reference_count(page), 1);

        cache.share(page).expect("第一次共享");
        cache.share(page).expect("第二次共享");
        assert_eq!(cache.reference_count(page), 3);
        assert_eq!((cache.in_use(), cache.free_pages()), (1, 1));

        cache.release(page).expect("第一次归还");
        cache.release(page).expect("第二次归还");
        assert_eq!(cache.reference_count(page), 1);
        assert!(cache.is_live(page), "仍有一个持有者，页不能回池");

        cache.release(page).expect("最后一次归还");
        assert_eq!(cache.reference_count(page), 0);
        assert_eq!((cache.in_use(), cache.free_pages()), (0, 2));
    }

    /// 共享入口同样校验越界与非存活页；失败不能改账目。
    #[test]
    fn sharing_an_out_of_range_or_dead_page_is_rejected() {
        let mut cache = Cache::new(1, 4).expect("页池");
        assert_eq!(cache.share(Page::new(9)).unwrap_err().code(), Code::Invalid);
        let page = cache.alloc().expect("分配");
        cache.release(page).expect("先归还");
        assert_eq!(cache.share(page).unwrap_err().code(), Code::Invalid);
        assert_eq!(cache.in_use(), 0, "共享失败不动占用");
        assert_eq!(cache.free_pages(), 1, "共享失败不把页塞进空闲表");
    }

    /// 多层模型共用一行：每层切自己那段，层与层不重叠。
    ///
    /// 页池不知道层，所以"第 `l` 层从哪个元素开始"只能由模型给偏移——这条测试锁住的是
    /// "给整行、按偏移切"这套用法本身，而不是某个层宽度的具体数值。
    #[test]
    fn layers_slice_their_own_part_of_one_row() {
        let mut pager = Pager::new(Cache::new(1, 2).expect("页池"), 4).expect("行宽");
        let page = pager.alloc().expect("页");
        let slot = Slot { page, offset: 1 };

        pager.row_mut(slot).expect("整行")[..2].copy_from_slice(&[1.0, 2.0]);
        pager.row_mut(slot).expect("整行")[2..].copy_from_slice(&[3.0, 4.0]);

        assert_eq!(
            pager.row(slot).expect("读"),
            [1.0, 2.0, 3.0, 4.0],
            "两段各写各的，读回时拼成一行"
        );
    }

}
