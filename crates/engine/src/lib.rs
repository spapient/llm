//! 服务侧状态机：[`Sequence`]、[`State`]、[`Scheduler`]。
//!
//! **拥有**：`Sequence`、`Scheduler`、`State`。**不拥有**：KV 页的簿记（`kv`）、提交（`exec`）、
//! 采样策略（P1：它只决定 `advance` 取哪个 token，不改变状态机）。
//! **依赖**：`kv`、`exec`、`tensor`、`diagnostic`。**被依赖**：`serve`、`llm`。
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
//! use exec::Stream;
//! use kv::Cache;
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

use diagnostic::{Code, Error, Result};
use tensor::Shape;

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
    stream: exec::Stream,
}

impl Sequence {
    /// 由请求参数构造：`page_size` 来自页池，用于算需要多少页。
    pub fn new(
        id: u64,
        prompt_len: usize,
        max_new: usize,
        page_size: usize,
        stream: exec::Stream,
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
    pub fn stream(&self) -> exec::Stream {
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
        let len = self.prompt_len.saturating_add(self.max_new);
        len.div_ceil(self.page_size).max(1)
    }

    /// 是否已结束。
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }

    /// 当前上下文长度（提示 + 已生成）。
    pub fn len(&self) -> usize {
        self.prompt_len + self.produced
    }

    /// 上下文是否为空（只有空提示且尚未生成时成立）。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 走到当前长度需要多少页：向上取整，至少一页（否则 `kv::Table` 无处可放）。
    pub fn pages_needed(&self) -> usize {
        self.len().div_ceil(self.page_size).max(1)
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
    table: kv::Table,
    needs_page: bool,
}

/// 连续批调度器：准入、占用页、结束后归还。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheduler {
    cache: kv::Cache,
    running: Vec<Running>,
}

impl Scheduler {
    /// 由页池构造。
    pub fn new(cache: kv::Cache) -> Self {
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
        if self.running.iter().any(|item| item.id == sequence.id()) {
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
        self.check_width(sequence, "engine::Scheduler::admit")?;
        if !self.fits(sequence) {
            return Ok(false);
        }
        let needed = sequence.pages_needed();
        if self.cache.free_pages() < needed {
            return Ok(false);
        }
        let mut table = kv::Table::new(needed)?;
        for _ in 0..needed {
            let page = self.cache.alloc().ok_or_else(|| {
                Error::new(Code::Internal, "空闲页数与实际分配不符")
                    .with("engine::Scheduler::admit")
            })?;
            table.push(page)?;
        }
        let needs_page = self.needs_page(sequence, &table);
        self.running.push(Running {
            id: sequence.id(),
            table,
            needs_page,
        });
        Ok(true)
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
        let Some(index) = self
            .running
            .iter()
            .position(|item| item.id == sequence.id())
        else {
            return Err(Error::new(
                Code::Invalid,
                format!("序列 {} 不在批里：生成必须先 `admit`", sequence.id()),
            )
            .with("engine::Scheduler::advance"));
        };
        if sequence.is_done() {
            return Err(
                Error::new(Code::Invalid, format!("序列 {} 已结束", sequence.id()))
                    .with("engine::Scheduler::advance"),
            );
        }
        self.check_width(sequence, "engine::Scheduler::advance")?;
        // 这一步要写的 token 落在 `len()` 这个位置，所以页表要覆盖 `len + 1` 个位置。
        let needed = (sequence.len() + 1).div_ceil(self.cache.page_size());
        let table = &mut self.running[index].table;
        let extra = needed.saturating_sub(table.len());
        if extra > 0 {
            // 先看池够不够，再动容量与分配：不够时三者都不变。
            if self.cache.free_pages() < extra {
                return Ok(Progress::Blocked);
            }
            table.grow(extra)?;
            for _ in 0..extra {
                let page = self.cache.alloc().ok_or_else(|| {
                    Error::new(Code::Internal, "空闲页数与实际分配不符")
                        .with("engine::Scheduler::advance")
                })?;
                table.push(page)?;
            }
        }
        sequence.step()?;
        let table = &self.running[index].table;
        let needs_page = self.needs_page(sequence, table);
        self.running[index].needs_page = needs_page;
        Ok(if sequence.is_done() {
            Progress::Done
        } else {
            Progress::Running
        })
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
        let Some(index) = self.running.iter().position(|item| item.id == id) else {
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
    pub fn pages_of(&self, id: u64) -> Option<&kv::Table> {
        self.running
            .iter()
            .find(|item| item.id == id)
            .map(|item| &item.table)
    }

    /// 页池（只读）。
    pub fn cache(&self) -> &kv::Cache {
        &self.cache
    }

    /// 下一次生成要不要新页：页表按 `len` 刚好够用，所以差数只可能是 0 或 1。
    fn needs_page(&self, sequence: &Sequence, table: &kv::Table) -> bool {
        if sequence.is_done() {
            return false;
        }
        (sequence.len() + 1).div_ceil(self.cache.page_size()) > table.len()
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
mod tests {
    use super::*;
    use kv::Cache;

    fn sequence(id: u64, prompt: usize, max_new: usize) -> Sequence {
        Sequence::new(id, prompt, max_new, 16, exec::Stream::new(0)).expect("序列")
    }

    #[test]
    fn a_sequence_generates_until_its_limit_then_stops() {
        let mut scheduler = Scheduler::new(Cache::new(4, 16).expect("页池"));
        let mut sequence = sequence(1, 4, 2);
        assert_eq!(sequence.state(), State::Waiting);
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(
            scheduler.advance(&mut sequence).expect("第一步"),
            Progress::Running
        );
        assert_eq!((sequence.produced(), sequence.state()), (1, State::Running));
        assert_eq!(
            scheduler.advance(&mut sequence).expect("第二步"),
            Progress::Done
        );
        assert!(sequence.is_done());
        assert_eq!(sequence.len(), 6);
        assert_eq!(
            scheduler.advance(&mut sequence).unwrap_err().code(),
            Code::Invalid,
            "结束之后不能再推进"
        );
        assert_eq!(
            scheduler.cache().in_use(),
            1,
            "结束的序列仍然占着页，直到 `finish`"
        );
    }

    #[test]
    fn pages_are_rounded_up_and_never_zero() {
        let empty = sequence(0, 0, 1);
        assert!(empty.is_empty());
        assert_eq!(empty.pages_needed(), 1, "至少一页");
        let long = sequence(1, 17, 1);
        assert_eq!(long.pages_needed(), 2);
        assert_eq!(long.kv_shape().dims(), [17, 16]);
    }

    /// 最坏情况的上界与"当前需要"是两个数：准入要的是前者。
    #[test]
    fn pages_required_covers_the_whole_run_not_just_the_prompt() {
        let short = Sequence::new(1, 4, 4, 4, exec::Stream::new(0)).expect("序列");
        assert_eq!(short.pages_needed(), 1, "当前长度 4 刚好一页");
        assert_eq!(short.pages_required(), 2, "跑满后长度 8 要两页");
        let long = Sequence::new(2, 1, 39, 4, exec::Stream::new(0)).expect("序列");
        assert_eq!(long.pages_needed(), 1);
        assert_eq!(long.pages_required(), 10, "(1 + 39).div_ceil(4)");
    }

    /// 独占整个页池也跑不完的序列不进批：否则整批会卡在 `Blocked` 上，而没人能结束。
    #[test]
    fn a_sequence_that_can_never_finish_is_refused() {
        let mut scheduler = Scheduler::new(Cache::new(4, 4).expect("页池"));
        let endless = Sequence::new(1, 4, 40, 4, exec::Stream::new(0)).expect("序列");
        assert_eq!(endless.pages_needed(), 1, "当前只占一页");
        assert!(
            !scheduler.admit(&endless).expect("页池放不下整段生成"),
            "跑不完的序列必须被拒绝，而不是先放进来再卡住"
        );
        assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下任何分配");
        assert_eq!(scheduler.running(), 0);
    }

    #[test]
    fn admission_occupies_and_finish_returns_the_pages() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let sequence = sequence(1, 20, 4);
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(scheduler.running(), 1);
        assert_eq!(scheduler.cache().in_use(), 2);
        assert_eq!(scheduler.pages_of(1).map(kv::Table::len), Some(2));
        assert!(scheduler.finish(1).expect("结束"));
        assert_eq!(scheduler.cache().in_use(), 0);
        assert!(
            !scheduler.finish(1).expect("重复结束"),
            "重复结束返回 false"
        );
        assert_eq!(scheduler.pages_of(1), None);
    }

    #[test]
    fn a_sequence_that_does_not_fit_is_refused_not_partially_admitted() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let big = sequence(7, 64, 4);
        assert!(
            !scheduler.admit(&big).expect("页不够"),
            "页不够时应返回 false"
        );
        assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下半个分配");
        assert_eq!(scheduler.running(), 0);
    }

    #[test]
    fn admitting_the_same_sequence_twice_is_a_bug() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let sequence = sequence(3, 8, 4);
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(
            scheduler.admit(&sequence).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn impossible_sequence_parameters_are_rejected() {
        assert_eq!(
            Sequence::new(0, 1, 1, 0, exec::Stream::new(0))
                .unwrap_err()
                .code(),
            Code::Invalid
        );
        assert_eq!(
            Sequence::new(0, 1, 0, 16, exec::Stream::new(0))
                .unwrap_err()
                .code(),
            Code::Invalid
        );
    }

    /// 生成会跨页：页表按"要写的位置"增长，而不是停在准入那一刻。
    #[test]
    fn a_growing_sequence_takes_a_page_each_time_it_crosses_one() {
        let mut scheduler = Scheduler::new(Cache::new(4, 4).expect("页池"));
        let mut sequence = Sequence::new(1, 3, 10, 4, exec::Stream::new(0)).expect("序列");
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(scheduler.cache().in_use(), 1, "提示长度 3 只占一页");
        // 长度 4 仍在第一页，写到第 5 个位置才需要第二页，第 9 个位置才需要第三页。
        for (length, pages) in [(4, 1), (5, 2), (6, 2), (7, 2), (8, 2), (9, 3)] {
            assert_eq!(
                scheduler.advance(&mut sequence).expect("生成"),
                Progress::Running
            );
            assert_eq!(sequence.len(), length);
            assert_eq!(scheduler.cache().in_use(), pages, "长度 {length}");
        }
        assert_eq!(scheduler.pages_of(1).map(kv::Table::len), Some(3));
        assert!(scheduler.finish(1).expect("结束"));
        assert_eq!(scheduler.cache().in_use(), 0, "增长出来的页也要还");
    }

    /// 池被别人占满时这一步不生成：序列、页表、页池三者都不动；等对方结束后原样重试即可。
    #[test]
    fn a_step_that_cannot_get_a_page_is_blocked_without_side_effects() {
        let mut scheduler = Scheduler::new(Cache::new(3, 4).expect("页池"));
        let mut a = Sequence::new(1, 4, 8, 4, exec::Stream::new(0)).expect("序列");
        let b = Sequence::new(2, 4, 8, 4, exec::Stream::new(0)).expect("序列");
        assert!(scheduler.admit(&b).expect("准入"));
        assert!(scheduler.admit(&a).expect("准入"));
        assert_eq!(scheduler.cache().in_use(), 2, "各自只占提示那一页");
        for length in 5..=8 {
            assert_eq!(scheduler.advance(&mut a).expect("生成"), Progress::Running);
            assert_eq!(a.len(), length);
        }
        assert_eq!(scheduler.cache().in_use(), 3, "长度 8 用掉最后一张空闲页");
        assert_eq!(
            scheduler.advance(&mut a).expect("页不够"),
            Progress::Blocked
        );
        assert_eq!(a.len(), 8, "阻塞不能推进序列");
        assert_eq!(a.produced(), 4, "阻塞不能多算一个 token");
        assert_eq!(scheduler.cache().in_use(), 3, "阻塞不能动页池");
        assert_eq!(scheduler.pages_of(1).map(kv::Table::len), Some(2));
        assert_eq!(
            scheduler.blocked(),
            vec![2, 1],
            "按准入顺序列出：空闲为零时两张页表都还要长，所以两条都在等"
        );
        // 别人的页回来之后，同一步原样重试就能过。
        assert!(scheduler.finish(2).expect("结束"));
        assert!(scheduler.blocked().is_empty(), "页回来了，谁都不再等");
        assert_eq!(
            scheduler.advance(&mut a).expect("页回来了"),
            Progress::Running
        );
        assert_eq!((a.len(), scheduler.cache().in_use()), (9, 3));
    }

    /// 判据本身也是公开的：服务层要按它区分"现在满"与"永远装不下"。
    #[test]
    fn fits_separates_a_full_pool_from_a_never_finishable_request() {
        let scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        assert!(scheduler.fits(&sequence(1, 4, 2)), "两页里跑得完");
        let endless = Sequence::new(2, 4, 1024, 16, exec::Stream::new(0)).expect("序列");
        assert!(!scheduler.fits(&endless), "最坏情况远超两页");
        assert!(scheduler.cache().in_use() == 0, "判据不占页");
    }

    #[test]
    fn advancing_a_sequence_that_is_not_in_the_batch_is_a_bug() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let mut sequence = sequence(4, 4, 4);
        assert_eq!(
            scheduler.advance(&mut sequence).unwrap_err().code(),
            Code::Invalid
        );
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert!(scheduler.finish(4).expect("结束"));
        assert_eq!(
            scheduler.advance(&mut sequence).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(sequence.produced(), 0, "失败的推进不能留下痕迹");
    }

    /// 页宽是页数的唯一依据：序列与页池不一致时，准入必须拒绝而不是少占页。
    #[test]
    fn a_sequence_whose_page_width_disagrees_with_the_pool_is_rejected() {
        let mut scheduler = Scheduler::new(Cache::new(4, 16).expect("页池"));
        let narrow = Sequence::new(5, 16, 4, 8, exec::Stream::new(0)).expect("序列");
        assert_eq!(scheduler.admit(&narrow).unwrap_err().code(), Code::Invalid);
        assert_eq!(scheduler.cache().in_use(), 0, "拒绝不能留下分配");

        let mut admitted = sequence(6, 16, 4);
        assert!(scheduler.admit(&admitted).expect("准入"));
        assert_eq!(
            scheduler.advance(&mut admitted).expect("生成"),
            Progress::Running
        );
    }

    #[test]
    fn a_done_sequence_is_not_admitted() {
        let mut scheduler = Scheduler::new(Cache::new(2, 16).expect("页池"));
        let mut sequence = sequence(7, 0, 1);
        assert!(scheduler.admit(&sequence).expect("准入"));
        assert_eq!(
            scheduler.advance(&mut sequence).expect("唯一一步"),
            Progress::Done
        );
        assert!(scheduler.finish(7).expect("结束"));
        assert_eq!(
            scheduler.admit(&sequence).unwrap_err().code(),
            Code::Invalid,
            "结束的序列再准入只会白占页"
        );
    }

    /// 确定性 PRNG：不给依赖表加 `rand`，同时让失败可复现。
    struct Rng {
        state: u32,
    }

    impl Rng {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn below(&mut self, bound: u32) -> u32 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 17;
            self.state ^= self.state << 5;
            self.state % bound.max(1)
        }
    }

    /// 随机准入 / 推进 / 结束，任何一步之后账都要对得上：
    ///
    /// - 页池占用 = 各页表长度之和，空闲 + 占用恒等于页总数；
    /// - 每张页表覆盖当前长度，最多短一页（下个 token 那一页）；
    /// - 报错的推进不改变 `produced`，`Blocked` 也不改变页池。
    #[test]
    fn random_batches_keep_the_page_accounts_consistent() {
        let mut rng = Rng::new(0x5eed);
        let mut scheduler = Scheduler::new(Cache::new(6, 4).expect("页池"));
        let mut sequences: Vec<Sequence> = (0..4u64)
            .map(|id| {
                let prompt = rng.below(8) as usize;
                let max_new = 1 + rng.below(6) as usize;
                Sequence::new(id, prompt, max_new, 4, exec::Stream::new(0)).expect("序列")
            })
            .collect();

        for _ in 0..400 {
            let index = rng.below(sequences.len() as u32) as usize;
            let id = sequences[index].id();
            match rng.below(3) {
                0 => {
                    let admitted = scheduler.pages_of(id).is_some();
                    let done = sequences[index].is_done();
                    match scheduler.admit(&sequences[index]) {
                        Err(error) => {
                            assert_eq!(error.code(), Code::Invalid);
                            assert!(admitted || done, "只有重复准入与已结束的序列才是错误");
                        }
                        Ok(grown) => {
                            assert!(!admitted && !done, "不该准入的序列被准入");
                            if grown {
                                assert!(scheduler.pages_of(id).is_some(), "准入成功必须占页");
                            }
                        }
                    }
                }
                1 => {
                    let before = sequences[index].produced();
                    let admitted = scheduler.pages_of(id).is_some();
                    let outcome = scheduler.advance(&mut sequences[index]);
                    // 批级信号必须与实际结局同源：`blocked()` 就是刚刚返回 Blocked 的那个集合。
                    assert_eq!(
                        scheduler.blocked().contains(&id),
                        outcome == Ok(Progress::Blocked),
                        "序列 {id} 的批级阻塞信号与单步结局不一致"
                    );
                    match outcome {
                        Err(error) => {
                            assert_eq!(error.code(), Code::Invalid);
                            assert!(
                                !admitted || sequences[index].is_done(),
                                "已准入且未结束的推进不该报错"
                            );
                            assert_eq!(sequences[index].produced(), before, "报错不能推进");
                        }
                        Ok(Progress::Blocked) => {
                            assert!(admitted);
                            assert_eq!(sequences[index].produced(), before, "阻塞不能推进");
                        }
                        Ok(progress) => {
                            assert_eq!(sequences[index].produced(), before + 1);
                            let expected = if sequences[index].is_done() {
                                Progress::Done
                            } else {
                                Progress::Running
                            };
                            assert_eq!(progress, expected);
                        }
                    }
                }
                _ => {
                    let had = scheduler.pages_of(id).is_some();
                    assert_eq!(scheduler.finish(id).expect("归还"), had);
                }
            }

            let mut sum = 0;
            for sequence in &sequences {
                let Some(table) = scheduler.pages_of(sequence.id()) else {
                    continue;
                };
                sum += table.len();
                assert!(table.len() <= sequence.pages_needed());
                assert!(
                    sequence.pages_needed() - table.len() <= 1,
                    "页表落后超过一页：序列 {} 长度 {} 需要 {} 页，只有 {} 页",
                    sequence.id(),
                    sequence.len(),
                    sequence.pages_needed(),
                    table.len()
                );
            }
            assert_eq!(scheduler.cache().in_use(), sum, "占用与页表不一致");
            assert_eq!(
                scheduler.cache().free_pages() + sum,
                scheduler.cache().total()
            );
        }
    }
}
