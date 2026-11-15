//! 服务侧状态机：[`Sequence`]、[`State`]、[`Scheduler`]。
//!
//! **拥有**：`Sequence`、`Scheduler`、`State`，以及页账（[`kv`]）与提交契约（[`exec`]）。
//! **不拥有**：采样策略（P1：它只决定 `advance` 取哪个 token，不改变状态机）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`serve`、`llm`。
//!
//! 页账与提交契约原是两个独立成员（`llm-kv` / `llm-exec`）：它们的生产消费者只有本 crate，
//! 合并后"谁在借页"与"谁在提交"的不变量在同一处断言，成员与依赖边各少一个。
//!
//! 连续批的关键不是"批"，而是**准入**：每个序列占多少页是显式算出来的，页不够就拒绝，
//! 而不是让 KV 缓冲在跑到一半时溢出。
//!
//! # 页账跟着序列一起长
//!
//! 准入只预留**提示长度**的页：按 `max_new` 全量预留会把池子提前锁死，而请求可能很早就结束。
//! 之后每生成一个 token，由 [`Scheduler::advance`] 预留它要落的那一页；池不够时它返回
//! [`Progress::Blocked`]，序列与页表都不动——调用方可以等一个序列 [`Scheduler::finish`] 之后
//! 原样重试，而不是让 KV 缓冲在生成到一半时越界。
//!
//! `Sequence::step` 因此不是公开 API：它只推进计数、不碰页，唯一的调用者是 [`Scheduler::advance`]。
//! "生成必须经过页账"因此是编译期就能保证的，而不是一句约定。
//!
//! # 活性：准入保证"各自跑得完"，批内仍需调用方腾页
//!
//! 准入还会拒绝 [`Sequence::pages_required`] 超过整个页池的请求——那种序列独占页池也跑不完，
//! 放进来只会让整批卡在 [`Progress::Blocked`] 上而无人能结束。但"每条都跑得完"不等于"同时放
//! 多少都行"：几条都跑得完的序列挤在一起仍可能互相占满页而全部 [`Progress::Blocked`]。所以
//! `Blocked` 不是错误，而是调用方的动作信号——结束最接近完成的那一条（它的页会回来），再原样
//! 重试同一步。这一层不替调用方选牺牲者：选谁结束是调度策略，不属于状态机。但"现在谁在等"
//! 必须由这一层回答，于是有 [`Scheduler::blocked`]：它与单步结局同源，调用方据此挑牺牲者，
//! 不必自己重算页数。
//!
//! # 例子
//!
//! ```
//! use llm_engine::{Progress, Scheduler, Sequence};
//! use llm_engine::exec::Stream;
//! use llm_engine::kv::Cache;
//!
//! let mut scheduler = Scheduler::new(Cache::new(4, 16).expect("页池"));
//! let mut sequence = Sequence::new(0, 8, 4, 16, Stream::new(0)).expect("序列");
//! assert!(scheduler.admit(&sequence).expect("准入"));
//! assert_eq!(
//!     scheduler.advance(&mut sequence).expect("生成一个 token"),
//!     Progress::Running,
//! );
//! assert_eq!(sequence.produced(), 1);
//! assert_eq!(scheduler.cache().in_use(), 1, "长度 9 仍在第一页里");
//! assert!(scheduler.finish(sequence.id()).expect("释放页"));
//! ```

#![deny(unsafe_code)]

pub mod exec;
pub mod kv;

use diagnostic::{Code, Error, Result};
use tensor::Shape;

/// 覆盖 `tokens` 个 token 需要多少页：向上取整，至少一页。
///
/// 页数口径只在这里定义一次。合并 `kv` 之前它被写了四遍（准入上界、当前长度、增长判定、
/// 下一次是否要页），四处里只要有一处写成"向下取整"或漏掉 `.max(1)`，就会出现"准入说能放下、
/// 增长时却发现没页"这种只在长提示上复现的错。
fn pages_for(tokens: usize, page_size: usize) -> usize {
    tokens.div_ceil(page_size).max(1)
}

/// [`Scheduler::advance`] 的结局："生成了一步"与"这一步没生成"必须能分开。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Progress {
    /// 生成了一个 token，序列还在跑。
    Running,
    /// 生成了一个 token，序列已结束（页仍然占着，直到 [`Scheduler::finish`]）。
    Done,
    /// 这一步需要的页不够：没有生成 token，序列与页表都没变，可以稍后原样重试。
    Blocked,
}

/// 序列状态：三个状态构成一条单向链路。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    /// 已提交、还没被准入。
    Waiting,
    /// 在批里，正在逐 token 生成。
    Running,
    /// 生成结束（达到上限或遇到停止符）。
    Done,
}

/// 一个推理请求的生命周期。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    id: u64,
    prompt_len: usize,
    produced: usize,
    max_new: usize,
    page_size: usize,
    state: State,
    stream: crate::exec::Stream,
}

impl Sequence {
    /// 由请求参数构造：`page_size` 来自页池，用于算需要多少页。
    pub fn new(
        id: u64,
        prompt_len: usize,
        max_new: usize,
        page_size: usize,
        stream: crate::exec::Stream,
    ) -> Result<Self> {
        if page_size == 0 {
            return Err(Error::new(Code::Invalid, "页宽不能为 0").with("engine::Sequence::new"));
        }
        if max_new == 0 {
            return Err(
                Error::new(Code::Invalid, "max_new 不能为 0（否则请求一进来就结束）")
                    .with("engine::Sequence::new"),
            );
        }
        // 上下文长度在整条生命周期里只增不减，`len()` 与 `len() + 1` 都不该溢出：一次校验
        // 比每处调用点各写一个 `saturating_add` 便宜，也避免“同一族量一处回绕、一处饱和”。
        if prompt_len.checked_add(max_new).is_none() {
            return Err(Error::new(
                Code::Invalid,
                format!("prompt_len {prompt_len} 加 max_new {max_new} 溢出：上下文长度装不下"),
            )
            .with("engine::Sequence::new"));
        }
        Ok(Self {
            id,
            prompt_len,
            produced: 0,
            max_new,
            page_size,
            state: State::Waiting,
            stream,
        })
    }

    /// 请求编号。
    pub fn id(&self) -> u64 {
        self.id
    }

    /// 提示长度。
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// 已生成 token 数。
    pub fn produced(&self) -> usize {
        self.produced
    }

    /// 最多生成多少 token。
    pub fn max_new(&self) -> usize {
        self.max_new
    }

    /// 当前状态。
    pub fn state(&self) -> State {
        self.state
    }

    /// 提交所在的流。
    pub fn stream(&self) -> crate::exec::Stream {
        self.stream
    }

    /// 推进一个 token；已经结束时再推进是调用方的 bug。
    ///
    /// 只改状态与计数，**不碰页**：页由 [`Scheduler::advance`] 在调用它之前预留。因此这条路径
    /// 只对同 crate 可见——绕过页账的生成不应该能写出来。
    pub(crate) fn step(&mut self) -> Result<()> {
        if self.state == State::Done {
            return Err(Error::new(Code::Invalid, "序列已结束").with("engine::Sequence::step"));
        }
        self.state = State::Running;
        self.produced += 1;
        if self.produced >= self.max_new {
            self.state = State::Done;
        }
        Ok(())
    }

    /// 页宽（与页池一致；准入与增长都按它算页数）。
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// 跑满 `max_new` 需要多少页（最坏情况的上界）。
    ///
    /// 与 [`Sequence::pages_needed`] 不同：那个是"按当前长度"的数，随生成增长；这个是准入时就能
    /// 算出的上界，用来回答"这条序列在这个页池里有没有可能跑完"。准入只看当前页数会把一条永远
    /// 跑不完的序列放进批里：它迟早 [`Progress::Blocked`]，而批里没有任何序列能结束，于是谁也
    /// 没法 [`Scheduler::finish`]、页也永远不会被归还。
    pub fn pages_required(&self) -> usize {
        pages_for(self.prompt_len.saturating_add(self.max_new), self.page_size)
    }

    /// 是否已结束。
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }

    /// 当前上下文长度（提示 + 已生成）。
    ///
    /// 不会溢出：[`Sequence::new`] 已经校验过 `prompt_len + max_new` 装得下，而 `produced`
    /// 永远小于 `max_new`。
    pub fn len(&self) -> usize {
        self.prompt_len + self.produced
    }

    /// 上下文是否为空（只有空提示且尚未生成时成立）。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 走到当前长度需要多少页：向上取整，至少一页（否则 [`crate::kv::Table`] 无处可放）。
    pub fn pages_needed(&self) -> usize {
        pages_for(self.len(), self.page_size)
    }

    /// KV 的逻辑形状：`[len, page_size]`。
    pub fn kv_shape(&self) -> Shape {
        Shape::new([self.len(), self.page_size])
    }
}

/// 批里一个正在跑的序列：它的页表就是它占的页。
///
/// `needs_page` 是"下一次 [`Scheduler::advance`] 要不要新页"的派生值，在 `admit` 与每次
/// `advance` 之后刷新。它让 [`Scheduler::blocked`] 不必再问调用方要 `Sequence`——批级信号
/// 只依赖批自己的状态。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Running {
    id: u64,
    table: crate::kv::Table,
    needs_page: bool,
}

/// 连续批调度器：准入、占用页、结束后归还。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheduler {
    cache: crate::kv::Cache,
    running: Vec<Running>,
}

impl Scheduler {
    /// 由页池构造。
    pub fn new(cache: crate::kv::Cache) -> Self {
        Self {
            cache,
            running: Vec::new(),
        }
    }

    /// 尝试准入：页够就占页并返回 `true`，不够返回 `false`（由调用方决定排队还是拒绝）。
    ///
    /// 预留的是**当前**（提示）长度的页：按 `max_new` 全量预留会把池子提前锁死，而请求可能
    /// 很早就结束；之后每一页由 [`Scheduler::advance`] 在真的需要时才占。
    ///
    /// 但"能开始"不等于"能收尾"：若 [`Sequence::pages_required`]（跑满 `max_new` 的上界）已经
    /// 超过整个页池，这条序列独占页池也跑不完，放进来只会让批卡在 [`Progress::Blocked`] 上、
    /// 而没有任何序列能结束，页也永远回不来。这种请求同样返回 `false`。
    pub fn admit(&mut self, sequence: &Sequence) -> Result<bool> {
        self.check_admissible(sequence)?;
        let needed = sequence.pages_needed();
        if !self.fits(sequence) || self.cache.free_pages() < needed {
            return Ok(false);
        }
        let table = self.reserve(needed, "engine::Scheduler::admit")?;
        let needs_page = self.needs_page(sequence, &table);
        self.running.push(Running {
            id: sequence.id(),
            table,
            needs_page,
        });
        Ok(true)
    }

    /// 准入的状态判据：重名、已结束、页宽不符都是调用方的状态错误；页够不够由调用点判。
    fn check_admissible(&self, sequence: &Sequence) -> Result<()> {
        if self.position(sequence.id()).is_some() {
            return Err(
                Error::new(Code::Invalid, format!("序列 {} 已经在批里", sequence.id()))
                    .with("engine::Scheduler::admit"),
            );
        }
        if sequence.is_done() {
            return Err(Error::new(
                Code::Invalid,
                format!("序列 {} 已结束，准入只会白占页", sequence.id()),
            )
            .with("engine::Scheduler::admit"));
        }
        self.check_width(sequence, "engine::Scheduler::admit")
    }

    /// 从页池里占 `needed` 页装成新页表；该不该占由调用点判。
    fn reserve(&mut self, needed: usize, site: &'static str) -> Result<crate::kv::Table> {
        let mut table = crate::kv::Table::new(needed)?;
        Self::fill(&mut self.cache, &mut table, needed, site)?;
        Ok(table)
    }

    /// 往页表里追加 `extra` 个新页：分配与入表成对发生，缺页说明空闲页数已被推翻。
    fn fill(
        cache: &mut crate::kv::Cache,
        table: &mut crate::kv::Table,
        extra: usize,
        site: &'static str,
    ) -> Result<()> {
        for _ in 0..extra {
            let page = cache
                .alloc()
                .ok_or_else(|| Error::new(Code::Internal, "空闲页数与实际分配不符").with(site))?;
            table.push(page)?;
        }
        Ok(())
    }

    /// 这条序列在这个页池里有没有可能跑完：最坏情况的页数不超过页池总数。
    ///
    /// 这是 [`Scheduler::admit`] 里"永远跑不完就拒绝"的判据本身，单独暴露是因为服务层要
    /// 据此区分两种拒绝：**现在满**（等别人结束，返回可重试的 503）与**永远装不下**
    /// （`max_new` 太大，重试多少次都一样，该返回 400）。两处共用一个定义，规则不会漂。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        sequence.pages_required() <= self.cache.total()
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
        let index = self.check_advance(sequence)?;
        if !self.extend(
            index,
            self.next_pages(sequence),
            "engine::Scheduler::advance",
        )? {
            return Ok(Progress::Blocked);
        }
        self.take_step(index, sequence)
    }

    /// 生成的前置判据：序列必须在批里、还没结束、页宽与页池一致。
    ///
    /// 返回它在批里的下标；三种不符都是调用方的状态错误，而不是「页不够」。
    fn check_advance(&self, sequence: &Sequence) -> Result<usize> {
        let index = self.position(sequence.id()).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("序列 {} 不在批里：生成必须先 `admit`", sequence.id()),
            )
            .with("engine::Scheduler::advance")
        })?;
        if sequence.is_done() {
            return Err(
                Error::new(Code::Invalid, format!("序列 {} 已结束", sequence.id()))
                    .with("engine::Scheduler::advance"),
            );
        }
        self.check_width(sequence, "engine::Scheduler::advance")?;
        Ok(index)
    }

    /// 把第 `index` 条页表扩到 `needed` 页；池不够时返回 `false`，三者都不动。
    ///
    /// 先看池够不够，再动容量与分配：不够时页池、页表、序列全不变，调用方可以原样重试。
    fn extend(&mut self, index: usize, needed: usize, site: &'static str) -> Result<bool> {
        let extra = needed.saturating_sub(self.running[index].table.len());
        if extra == 0 {
            return Ok(true);
        }
        if self.cache.free_pages() < extra {
            return Ok(false);
        }
        self.running[index].table.grow(extra)?;
        Self::fill(&mut self.cache, &mut self.running[index].table, extra, site)?;
        Ok(true)
    }

    /// 走完一步并刷新批级派生值：推进计数器、重算「下一次要不要新页」、给出结局。
    fn take_step(&mut self, index: usize, sequence: &mut Sequence) -> Result<Progress> {
        sequence.step()?;
        let needs_page = self.needs_page(sequence, &self.running[index].table);
        self.running[index].needs_page = needs_page;
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

    /// 现在拿不到页、必须先等别人结束的序列（按插入顺序）。
    ///
    /// 这是 [`Scheduler::advance`] 会返回 [`Progress::Blocked`] 的那些序列，所以"这一批卡在
    /// 谁身上"是一个批级问题、不需要再问调用方要 [`Sequence`]。判据是精确的而不是估算：
    /// 页表一直按 `len` 刚好够用，所以下一步最多多要一页，于是"要新页且空闲为零"就是全部
    /// 阻塞条件。
    pub fn blocked(&self) -> Vec<u64> {
        if self.cache.free_pages() > 0 {
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
        let Some(index) = self.position(id) else {
            return Ok(false);
        };
        let running = self.running.remove(index);
        for page in running.table.pages() {
            self.cache.release(*page)?;
        }
        Ok(true)
    }

    /// 批里正在跑的序列数。
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// 某个序列占的页表。
    pub fn pages_of(&self, id: u64) -> Option<&crate::kv::Table> {
        self.position(id).map(|index| &self.running[index].table)
    }

    /// 页池（只读）。
    pub fn cache(&self) -> &crate::kv::Cache {
        &self.cache
    }

    /// 下一次生成要不要新页：页表按 `len` 刚好够用，所以差数只可能是 0 或 1。
    fn needs_page(&self, sequence: &Sequence, table: &crate::kv::Table) -> bool {
        if sequence.is_done() {
            return false;
        }
        self.next_pages(sequence) > table.len()
    }

    /// 写到下一个 token 需要的页数：这一步落在 `len()` 这个位置，所以页表要盖住 `len + 1` 个位置。
    ///
    /// [`Scheduler::advance`] 与 [`Scheduler::needs_page`] 都从这里取：两处各写一遍
    /// `pages_for(len + 1)` 时，任一处改成 `len` 或 `>=`，[`Scheduler::blocked`] 就会把无辜的
    /// 序列报成受害者，而调用方会据此结束错的序列。
    fn next_pages(&self, sequence: &Sequence) -> usize {
        pages_for(sequence.len() + 1, self.cache.page_size())
    }

    /// 页宽必须与页池一致：页数是按它算的，差一档就会少占一页。
    fn check_width(&self, sequence: &Sequence, site: &'static str) -> Result<()> {
        if sequence.page_size() == self.cache.page_size() {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!(
                "序列 {} 的页宽 {} 与页池的 {} 不一致：页数会算错",
                sequence.id(),
                sequence.page_size(),
                self.cache.page_size()
            ),
        )
        .with(site))
    }
}

#[cfg(test)]
mod tests;
