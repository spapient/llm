//! 连续批调度：[`Scheduler`] 的准入、页账、前缀复用、生成一步与结束归还。
//!
//! 一个调度器只回答三件事：这条序列**能不能进来**（[`Scheduler::admit`]）、它下一个 token 的
//! KV **写到哪里**（[`Scheduler::slot`]）、它**什么时候把页还回来**（[`Scheduler::finish`]）。
//! 页号与页里的数据在同一个 [`Pager`] 上，所以"占了页"与"能写这一行"不会是两笔账。

use crate::invalid;
use crate::kv::{Cache, Page, Pager, Radix, Slot, Table};
use crate::sequence::{Progress, Sequence, pages_for};
use diagnostic::{Code, Error, Result};

/// 批里一个正在跑的序列：它的页表就是它占的页。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Running {
    id: u64,
    table: Table,
    needs_page: bool,
}

/// 一次准入落定的页：建好的页表与被前缀树接走的块数。
struct Plan {
    table: Table,
    shared: usize,
}

/// 连续批调度器：准入、占用页、结束后归还。
#[derive(Debug, Clone, PartialEq)]
pub struct Scheduler {
    pager: Pager,
    running: Vec<Running>,
    /// 满块前缀索引：跨请求复用 KV 的唯一来源。
    radix: Radix,
}

impl Scheduler {
    /// 由 KV 页池构造：页号簿记与页里的数据一起进来。
    pub fn new(pager: Pager) -> Self {
        Self {
            pager,
            running: Vec::new(),
            radix: Radix::new(),
        }
    }

    /// 尝试准入：页够就占页并返回**已被接走的 token 数**，不够返回 `None`。
    ///
    /// 预留的是**当前**（提示）长度的页：按 `max_new` 全量预留会把池子提前锁死，而请求可能
    /// 很早就结束；之后每一页由 [`Scheduler::advance`] 在真的需要时才占。
    ///
    /// 前缀页来自前缀树的满块命中：那是**跨请求**的复用，源序列结束、页归还之后仍命中得到
    /// （树自己持一份引用）。整页共享因此一定按页边界对齐，不满一页的尾巴落在新页里。
    ///
    /// 但"能开始"不等于"能收尾"：若 [`Sequence::pages_required`]（跑满 `max_new` 的上界）已经
    /// 超过整个页池，这条序列独占页池也跑不完，放进来只会让批卡在 [`Progress::Blocked`] 上、
    /// 而没有任何序列能结束，页也永远回不来。这种请求同样返回 `None`。
    ///
    /// 返回值与页账是同一份：只共享页不改位置，下一个请求会把同一段提示重算一遍，而重算的
    /// 结果正好是它要去读的那些行。所以调用方必须据这个数把「已算位置」推上去。
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

    /// 生成一个 token：先为它要落的那个位置预留页，再推进序列。
    ///
    /// 这是批里唯一的生成入口：`Sequence` 的计数器只在这里推进，所以"生成了却没有页账"
    /// 写不出来。池不够时返回 [`Progress::Blocked`] 且**不做任何改动**——序列没前进、页表
    /// 没长、页池没动，调用方可以等别的序列 [`Scheduler::finish`] 之后原样重试。
    ///
    /// # Errors
    ///
    /// 序列不在批里（没 `admit`、或已经 `finish`）、已经结束、或页宽与页池不一致时报错；
    /// 这些都属于调用方的状态错误，而不是"页不够"。
    pub fn advance(&mut self, sequence: &mut Sequence) -> Result<Progress> {
        let idx = self.check_advance(sequence)?;
        if !self.extend(idx, self.next_pages(sequence))? {
            return Ok(Progress::Blocked);
        }
        self.take_step(idx, sequence)
    }

    /// 把一个写满的块登记进前缀树，返回是否新登记（重复登记返回 `false`）。
    ///
    /// 由调用方在**写完**这一块的那一刻调用：前缀复用的单位是"算好的块"，只有写的那一刻
    /// 才知道它算完了。[`Radix::publish`] 自己校验块边界与页的存活状态。
    ///
    /// # Errors
    ///
    /// 页宽为 0、`block` 超出现有 token 时返回 `invalid`；页不在借出状态由页池报错。
    pub fn publish(&mut self, tokens: &[u32], block: usize, page: Page) -> Result<bool> {
        let width = self.pager.page_size();
        self.radix
            .publish(&mut self.pager, tokens, width, block, page)
    }

    /// 前缀树里的块数（只读诊断）：服务侧报"共享缓存有多深"时用，不暴露树本身。
    pub fn cached_blocks(&self) -> usize {
        self.radix.len()
    }

    /// 这条序列在这个页池里有没有可能跑完：最坏情况的页数不超过页池总数。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        sequence.pages_required() <= self.pager.total()
    }

    /// 现在拿不到页、必须先等别人结束的序列（按插入顺序）。
    pub fn blocked(&self) -> Vec<u64> {
        if self.pager.free_pages() > 0 {
            return Vec::new();
        }
        self.running
            .iter()
            .filter(|item| item.needs_page)
            .map(|item| item.id)
            .collect()
    }

    /// 结束一个序列并归还它的页；返回它是否在批里。
    pub fn finish(&mut self, id: u64) -> Result<bool> {
        let Some(idx) = self.position(id) else {
            return Ok(false);
        };
        let mut running = self.running.remove(idx);
        for page in running.table.truncate(0)? {
            self.pager.release(page)?;
        }
        Ok(true)
    }

    /// 批里正在跑的序列数。
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// 某个序列占的页表。
    pub fn pages_of(&self, id: u64) -> Option<&Table> {
        self.position(id).map(|idx| &self.running[idx].table)
    }

    /// 页号分配器（只读）：服务层报"占了多少页"时用它，页里的数据不在这里。
    pub fn cache(&self) -> &Cache {
        self.pager.cache()
    }

    /// KV 页池（只读）：页号与页里的数据是同一个对象。
    pub fn pager(&self) -> &Pager {
        &self.pager
    }

    /// 页宽（每页多少 token）：调用方算"要几页"时不必再去拿页池。
    pub fn page_size(&self) -> usize {
        self.pager.page_size()
    }

    /// 批里某个序列的第 `token` 个位置落在哪一页的哪一格。
    ///
    /// 这是"KV 写到哪里"的唯一入口：页号归本层，页内偏移由页宽除出来。写 kernel 的一侧
    /// 因此不需要自己拿页表再算一遍偏移。
    ///
    /// # Errors
    ///
    /// 序列不在批里，或那个 token 还没有页（页表只盖住已经生成的位置）。
    pub fn slot(&self, id: u64, token: usize) -> Result<Slot> {
        let table = self.pages_of(id).ok_or_else(|| {
            invalid(
                "engine::Scheduler::slot",
                format!("序列 {id} 不在批里：先 `admit` 再问页位置"),
            )
        })?;
        table.slot(token, self.pager.page_size())
    }

    /// 把第 `token` 个位置的 KV 行写进页池（长度必须等于页池的行宽）。
    ///
    /// 位置必须已经有页：页表只盖住"已经生成到的地方"，所以这条路径写不出超前于生成的位置；
    /// 也不会写进一张已经归还的页（[`Pager`] 会拒绝一个失效的 [`Slot`]）。
    ///
    /// # Errors
    ///
    /// 序列不在批里、位置还没有页、行宽不符、或那页已归还。
    pub fn write_kv(&mut self, id: u64, token: usize, values: &[f32]) -> Result<()> {
        let slot = self.slot(id, token)?;
        self.pager.write(slot, values)
    }

    /// 读回第 `token` 个位置的 KV 行（只读）。
    ///
    /// # Errors
    ///
    /// 同 [`Scheduler::write_kv`] 的前两种情形。
    pub fn kv(&self, id: u64, token: usize) -> Result<&[f32]> {
        let slot = self.slot(id, token)?;
        self.pager.row(slot)
    }

    /// 把页表扩到盖住前 `rows` 行（1-based 行账：第 `n` 个 token 落在行 `n`）。
    ///
    /// 预填充一次写多行、不走 [`Scheduler::advance`] 的逐步预留，所以在写第一批行之前要先把页
    /// 一次要到。
    ///
    /// 返回「要到了没有」：`false` 是现在没页——调用方这一轮先不算，页紧是常态，不是别人的状态
    /// 错；只有序列不在批里、页宽为零这类调用方状态错误才是 `Err`。
    ///
    /// # Errors
    ///
    /// 序列不在批里或页宽为零（都算不出要几页）。
    pub fn reserve(&mut self, id: u64, rows: usize) -> Result<bool> {
        let Some(idx) = self.position(id) else {
            return Err(invalid(
                "engine::Scheduler::reserve",
                format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
            ));
        };
        let width = self.pager.page_size();
        if width == 0 {
            return Err(invalid(
                "engine::Scheduler::reserve",
                "页宽为零：算不出要几页",
            ));
        }
        self.extend(idx, pages_for(rows, width))
    }

    /// 一次准入要占的页：先按 LRU 腾出位置，再共享命中的前缀页、补新页。
    fn plan(&mut self, sequence: &Sequence) -> Result<Option<Plan>> {
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
    fn make_room(&mut self, fresh: usize) -> Result<bool> {
        let Some(short) = fresh.checked_sub(self.pager.free_pages()) else {
            return Ok(true);
        };
        if short > 0 {
            self.radix.evict(&mut self.pager, short)?;
        }
        Ok(self.pager.free_pages() >= fresh)
    }

    /// 建页表：先按块序共享命中页，再补 `fresh` 个新页。
    fn take_pages(&mut self, prefix: &[Page], fresh: usize, needed: usize) -> Result<Table> {
        let mut table = Table::new(needed)?;
        for page in prefix {
            self.pager.share(*page)?;
            table.push(*page)?;
        }
        Self::fill(&mut self.pager, &mut table, fresh)?;
        Ok(table)
    }

    /// 准入的状态判据：重名、已结束、页宽不符都是调用方的状态错误；页够不够由调用点判。
    fn check_admissible(&self, sequence: &Sequence) -> Result<()> {
        if self.position(sequence.id()).is_some() {
            return Err(invalid(
                "engine::Scheduler::admit",
                format!("序列 {} 已经在批里", sequence.id()),
            ));
        }
        if sequence.is_done() {
            return Err(invalid(
                "engine::Scheduler::admit",
                format!("序列 {} 已结束，准入只会白占页", sequence.id()),
            ));
        }
        self.check_width(sequence, "engine::Scheduler::admit")
    }

    /// 往页表里追加 `extra` 个新页：分配与入表成对发生。
    ///
    /// `alloc` 返回 `None` 说明有人刚刚推翻了 `free_pages()` 的回答——那是本层自己的账错了，
    /// 不是调用方的状态错，所以报 `Internal`。
    fn fill(pager: &mut Pager, table: &mut Table, extra: usize) -> Result<()> {
        for _ in 0..extra {
            let page = pager.alloc().ok_or_else(|| {
                Error::new(Code::Internal, "空闲页数与实际分配不符").with("engine::Scheduler::fill")
            })?;
            table.push(page)?;
        }
        Ok(())
    }

    /// 生成的前置判据：序列必须在批里、还没结束、页宽与页池一致。
    fn check_advance(&self, sequence: &Sequence) -> Result<usize> {
        let idx = self.position(sequence.id()).ok_or_else(|| {
            invalid(
                "engine::Scheduler::advance",
                format!("序列 {} 不在批里：生成必须先 `admit`", sequence.id()),
            )
        })?;
        if sequence.is_done() {
            return Err(invalid(
                "engine::Scheduler::advance",
                format!("序列 {} 已结束", sequence.id()),
            ));
        }
        self.check_width(sequence, "engine::Scheduler::advance")?;
        Ok(idx)
    }

    /// 把第 `idx` 条页表扩到 `needed` 页；池不够时返回 `false`，三者都不动。
    fn extend(&mut self, idx: usize, needed: usize) -> Result<bool> {
        let extra = needed.saturating_sub(self.running[idx].table.len());
        if extra == 0 {
            return Ok(true);
        }
        if self.pager.free_pages() < extra {
            return Ok(false);
        }
        self.running[idx].table.grow(extra)?;
        Self::fill(&mut self.pager, &mut self.running[idx].table, extra)?;
        Ok(true)
    }

    /// 走完一步并刷新批级派生值：推进计数器、重算「下一次要不要新页」、给出结局。
    fn take_step(&mut self, idx: usize, sequence: &mut Sequence) -> Result<Progress> {
        sequence.step()?;
        let needs_page = self.needs_page(sequence, &self.running[idx].table);
        self.running[idx].needs_page = needs_page;
        Ok(if sequence.is_done() {
            Progress::Done
        } else {
            Progress::Running
        })
    }

    /// 某个序列在批里的下标（插入顺序即批内顺序）。
    fn position(&self, id: u64) -> Option<usize> {
        self.running.iter().position(|item| item.id == id)
    }

    /// 下一次生成要不要新页：页表按 `len` 刚好够用，所以差数只可能是 0 或 1。
    fn needs_page(&self, sequence: &Sequence, table: &Table) -> bool {
        if sequence.is_done() {
            return false;
        }
        self.next_pages(sequence) > table.len()
    }

    /// 写到下一个 token 需要的页数：这一步落在 `len()` 这个位置，所以页表要盖住 `len + 1` 个位置。
    fn next_pages(&self, sequence: &Sequence) -> usize {
        pages_for(sequence.len() + 1, self.pager.page_size())
    }

    /// 页宽必须与页池一致：页数是按它算的，差一档就会少占一页。
    fn check_width(&self, sequence: &Sequence, site: &'static str) -> Result<()> {
        if sequence.page_size() == self.pager.page_size() {
            return Ok(());
        }
        Err(invalid(
            site,
            format!(
                "序列 {} 的页宽 {} 与页池的 {} 不一致：页数会算错",
                sequence.id(),
                sequence.page_size(),
                self.pager.page_size()
            ),
        ))
    }
}
