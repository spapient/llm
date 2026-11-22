//! 连续批调度：[`Scheduler`] 的准入、页账、前缀复用与结束归还。

use crate::sequence::{Progress, Sequence, pages_for};
use diagnostic::{Code, Error, Result};

/// 批里一个正在跑的序列：它的页表就是它占的页。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Running {
    id: u64,
    table: crate::kv::Table,
    needs_page: bool,
}

/// 一次准入落定的页：建好的页表与被前缀树接走的 token 数（共享页 × 页宽）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Plan {
    table: crate::kv::Table,
    shared: usize,
}

/// 连续批调度器：准入、占用页、结束后归还。
#[derive(Debug, Clone, PartialEq)]
pub struct Scheduler {
    pager: crate::kv::Pager,
    running: Vec<Running>,
    /// 满块前缀索引：跨请求复用 KV 的唯一来源（见 [`kv::Radix`]）。
    radix: crate::kv::Radix,
}

mod step;

impl Scheduler {
    /// 由 KV 页池构造：页号簿记与页里的数据一起进来。
    pub fn new(pager: crate::kv::Pager) -> Self {
        Self {
            pager,
            running: Vec::new(),
            radix: crate::kv::Radix::new(),
        }
    }

    /// 尝试准入：页够就占页并返回 `true`，不够返回 `false`（由调用方决定排队还是拒绝）。
    ///
    /// 预留的是**当前**（提示）长度的页：按 `max_new` 全量预留会把池子提前锁死，而请求可能
    /// 很早就结束；之后每一页由 [`Scheduler::advance`] 在真的需要时才占。
    ///
    /// 前缀页来自 [`kv::Radix`] 的满块命中：那是**跨请求**的复用，源序列结束、页归还之后仍
    /// 命中得到（树自己持一份引用）。整页共享因此一定按页边界对齐，不满一页的尾巴落在新页里。
    ///
    /// 但"能开始"不等于"能收尾"：若 [`Sequence::pages_required`]（跑满 `max_new` 的上界）已经
    /// 超过整个页池，这条序列独占页池也跑不完，放进来只会让批卡在 [`Progress::Blocked`] 上、
    /// 而没有任何序列能结束，页也永远回不来。这种请求同样返回 `false`。
    ///
    /// 空闲页不够时先按 LRU 摘掉前缀树的叶再试一次：缓存是自愿的，让它挡住一次本来能跑的
    /// 准入是本末倒置。摘完仍不够才拒。
    ///
    /// 返回值是**已被接走的 token 数**（共享页 × 页宽，只会小于 `prompt_len`，因为前缀树只
    /// 记整块）：调用方据此把「已算位置」推上去，这些位置的 K/V 在本页池里已经算好且就绪，
    /// 不必再算一遍。`None` 是「现在装不下」。页与位置的账必须是同一份：只共享页不改位置，
    /// 下一个请求会把同一段提示重算一遍，而重算的结果正好是它要去读的那些行。
    ///
    /// # Errors
    ///
    /// 序列自身不合法（重名、已结束、页宽不符）时返回 `invalid`——那是调用方的状态错误，
    /// 与"页不够"不是一回事。
    pub fn admit(&mut self, sequence: &Sequence) -> Result<Option<usize>> {
        self.check_admissible(sequence)?;
        if !self.fits(sequence) {
            return Ok(None);
        }
        let Some(plan) = self.plan(sequence)? else {
            return Ok(None);
        };
        let needs_page = self.needs_page(sequence, &plan.table);
        self.running.push(Running {
            id: sequence.id(),
            table: plan.table,
            needs_page,
        });
        Ok(Some(plan.shared * self.pager.page_size()))
    }

    /// 一次准入要占的页：共享的前缀页与新建页，以及被接走的 token 数。
    pub(super) fn plan(&mut self, sequence: &Sequence) -> Result<Option<Plan>> {
        let needed = sequence.pages_needed();
        let width = self.pager.page_size();
        let prefix = self.radix.longest(sequence.ctx(), width);
        let shared = prefix.len().min(needed);
        let fresh = needed - shared;
        if !self.make_room(fresh)? {
            return Ok(None);
        }
        let table = self.take_pages(&prefix[..shared], fresh, needed)?;
        Ok(Some(Plan { table, shared }))
    }

    /// 空闲页不够时先按 LRU 摘掉前缀树的叶再试一次：缓存是自愿的，让它挡住一次本来能跑的
    /// 准入是本末倒置。摘完仍不够返回 `false`。
    pub(super) fn make_room(&mut self, fresh: usize) -> Result<bool> {
        let Some(short) = fresh.checked_sub(self.pager.free_pages()) else {
            return Ok(true);
        };
        if short > 0 {
            self.radix.evict(&mut self.pager, short)?;
        }
        Ok(self.pager.free_pages() >= fresh)
    }

    /// 建页表：先按块序共享命中页，再补 `fresh` 个新页。共享与新建的账必须同一份——
    /// 只共享页不改位置，下一个请求会把同一段提示重算一遍，而重算的结果正好是它要去读的行。
    pub(super) fn take_pages(
        &mut self,
        prefix: &[crate::kv::Page],
        fresh: usize,
        needed: usize,
    ) -> Result<crate::kv::Table> {
        let mut table = crate::kv::Table::new(needed)?;
        self.link_prefix(&mut table, prefix)?;
        Self::fill(
            &mut self.pager,
            &mut table,
            fresh,
            "engine::Scheduler::admit",
        )?;
        Ok(table)
    }

    /// 按块序把命中页接进页表：共享一份引用，不新建页。
    pub(super) fn link_prefix(
        &mut self,
        table: &mut crate::kv::Table,
        prefix: &[crate::kv::Page],
    ) -> Result<()> {
        for page in prefix {
            self.pager.share(*page)?;
            table.push(*page)?;
        }
        Ok(())
    }

    /// 把一个写满的块登记进前缀树，返回是否新登记（重复登记返回 `false`）。
    ///
    /// 由调用方在**写完**这一块的那一刻调用：前缀复用的单位是"算好的块"，只有写的那一刻
    /// 才知道它算完了。[`kv::Radix::publish`] 自己校验块边界与页的存活状态。
    ///
    /// # Errors
    ///
    /// 页宽为 0、`block` 超出现有 token 时返回 `invalid`；页不在借出状态由页池报错。
    pub fn publish(&mut self, tokens: &[u32], block: usize, page: crate::kv::Page) -> Result<bool> {
        let width = self.pager.page_size();
        self.radix
            .publish(&mut self.pager, tokens, width, block, page)
    }

    /// 前缀树里的块数（只读诊断）：服务侧报"共享缓存有多深"时用，不暴露树本身。
    pub fn cached_blocks(&self) -> usize {
        self.radix.len()
    }

    /// 准入的状态判据：重名、已结束、页宽不符都是调用方的状态错误；页够不够由调用点判。
    pub(super) fn check_admissible(&self, sequence: &Sequence) -> Result<()> {
        if self.position(sequence.id()).is_some() {
            return Err(
                invalid("engine::Scheduler::admit", format!("序列 {} 已经在批里", sequence.id())),
            );
        }
        if sequence.is_done() {
            return Err(invalid("engine::Scheduler::admit", format!("序列 {} 已结束，准入只会白占页", sequence.id()),));
        }
        self.check_width(sequence, "engine::Scheduler::admit")
    }

    /// 往页表里追加 `extra` 个新页：分配与入表成对发生，缺页说明空闲页数已被推翻。
    pub(super) fn fill(
        pager: &mut crate::kv::Pager,
        table: &mut crate::kv::Table,
        extra: usize,
        site: &'static str,
    ) -> Result<()> {
        for _ in 0..extra {
            let page = pager
                .alloc()
                .ok_or_else(|| Error::new(Code::Internal, "空闲页数与实际分配不符").with(site))?;
            table.push(page)?;
        }
        Ok(())
    }

    /// 这条序列在这个页池里有没有可能跑完：最坏情况的页数不超过页池总数。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        sequence.pages_required() <= self.pager.total()
    }
}
