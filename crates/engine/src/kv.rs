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

#[cfg(test)]
mod tests {
    //! 页缓存的账目夹具：分配、复用、引用计数与前缀复用。

    use super::*;
    use diagnostic::Code;

    testkit::test_cases! {
    fn pages_are_handed_out_in_order_and_reused() {
        let mut cache = Cache::new(2, 4).expect("页池");
        let first = cache.alloc().expect("第一页");
        let second = cache.alloc().expect("第二页");
        assert_eq!((first.id(), second.id()), (0, 1));
        assert!(cache.alloc().is_none(), "池空必须返回 None");
        cache.release(first).expect("归还");
        assert_eq!(cache.alloc().map(|page| page.id()), Some(0));
    }

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

    fn a_zero_sized_pool_is_rejected() {
        assert_eq!(Cache::new(0, 16).unwrap_err().code(), Code::Invalid);
        assert_eq!(Cache::new(4, 0).unwrap_err().code(), Code::Invalid);
    }

    fn a_block_shape_matches_the_page_width() {
        let cache = Cache::new(4, 16).expect("页池");
        assert_eq!(cache.block_shape(2).dims(), [2, 16]);
        assert_eq!(cache.total(), 4);
        assert_eq!(cache.free_pages(), 4);
    }

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

    /// 寻址是唯一的除法：第 `idx` 个 token 落在 `idx / page_size` 页的 `idx % page_size` 格。
    fn a_token_idx_maps_to_one_page_and_one_offset() {
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
    fn a_zero_width_pager_is_rejected() {
        let cache = Cache::new(2, 4).expect("页池");
        let error = Pager::new(cache, 0).expect_err("行宽为零");
        assert_eq!(error.code(), Code::Invalid);
    }

    /// 整池一次分配：容量与页池相等，跨页的行不共享元素。
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

    /// E12 随机历史访问 + 块表 O(1)（`#[ignore]` 实测：
    /// `cargo test --release -p llm-engine --lib -- e12_random_access --ignored --nocapture`）。
    ///
    /// 冻结工作集 4096 页 × 16 token/页 = 65536 历史 token；确定性生成 10 万次随机访问，
    /// `Table::slot` 只做 `idx / page_size` + `idx % page_size` + 一次 `Vec` 下标（源码即
    /// O(1)，不随历史长度遍历），这里再用不同历史长度下的查表墙钟做经验佐证。
    #[ignore = "E12 块表基准：cargo test --release -p llm-engine --lib -- e12_random_access --ignored --nocapture"]
    fn e12_random_access() {
        let page_size = 16usize;
        let pages = 4096usize;
        let history = pages * page_size;
        let mut table = Table::new(pages).expect("页表");
        for id in 0..pages as u32 {
            table.push(Page::new(id)).expect("追加");
        }

        // 确定性 xorshift64 随机访问集（冻结种子，可复现）。
        let mut seed = 0xdead_beef_cafe_1234u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let samples = 100_000usize;
        let mut hits = 0usize;
        for _ in 0..samples {
            let idx = (next() % history as u64) as usize;
            let slot = table.slot(idx, page_size).expect("冻结工作集内必有页");
            let expected_page = (idx / page_size) as u32;
            let expected_offset = idx % page_size;
            if slot.page().id() == expected_page && slot.offset() == expected_offset {
                hits += 1;
            }
        }
        let recall = hits as f64 / samples as f64;
        println!(
            "E12 随机访问 recall={recall:.6}（{hits}/{samples}，history={history}，page_size={page_size}）"
        );

        // 操作计数佐证：不同历史长度下 slot 的墙钟（每点取 min），应不随页数增长。
        for page_count in [1024usize, 4096, 16384, 65536] {
            let mut big = Table::new(page_count).expect("页表");
            for id in 0..page_count as u32 {
                big.push(Page::new(id)).expect("追加");
            }
            let reps = 1_000_000usize;
            let idx = page_count * page_size - 1; // 最远端 token，也不该更慢
            for _ in 0..3 {
                big.slot(idx, page_size).expect("有页");
            }
            let start = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(big.slot(idx, page_size).expect("有页"));
            }
            let per = start.elapsed().as_secs_f64() / reps as f64;
            println!(
                "E12 slot 查表 pages={page_count:<6} history={:<8} 每次={:.3} ns",
                page_count * page_size,
                per * 1e9
            );
        }
    }
    }
}
