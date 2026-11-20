//! 服务侧状态机：[`Sequence`]、[`State`]、[`Scheduler`] 与连续批 [`Batch`]。
//!
//! **拥有**：`Sequence`、`Scheduler`、`Batch`、`Admission`、`State`，以及页账（[`kv`]）、
//! 提交与执行（[`exec`]）、设备后端 FFI（[`metal`]，macOS 越界岛，Metal `Executor` 落点，不单独成 crate）。
//! **不拥有**：采样策略（它只决定 `advance` 取哪个 token，不改变状态机）、
//! kernel 编译（`cutile`）、缓冲分配（`storage`）。**依赖**：`tensor`、`diagnostic`、`storage`。
//! **被依赖**：`serve`、`llm`。
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
//! # KV 数据只经页表进入
//!
//! 页里的数在 [`kv::Pager`] 里，而拿到一行只经 [`Scheduler::slot`]：先 `admit`、再按位置问页。
//! 于是"往一个还没有页的位置写 KV"写不出来（那个位置根本不在页表里），而"往已经归还的页写"
//! 会在 [`kv::Pager`] 的存活校验上失败——失效的 [`kv::Slot`] 不会被当成合法的写入地址。
//! [`Scheduler::write_kv`] 与 [`Scheduler::kv`] 只是把这条路径说完整：模型不必自己拿页表再算
//! 一次偏移，而页表与数据的口径只有一个。
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
//! use llm_engine::kv::{Cache, Pager};
//!
//! let mut scheduler = Scheduler::new(Pager::new(Cache::new(4, 16).expect("页池"), 4).expect("行宽"));
//! let mut sequence = Sequence::new(0, &[0; 8], 4, 16, Stream::new(0)).expect("序列");
//! assert_eq!(
//!     scheduler.admit(&sequence).expect("准入"),
//!     Some(0),
//!     "页池空着，没有可接走的前缀：提示段一个位置都没算过",
//! );
//! assert_eq!(
//!     scheduler.advance(&mut sequence).expect("生成一个 token"),
//!     Progress::Running,
//! );
//! assert_eq!(sequence.produced(), 1);
//! assert_eq!(scheduler.cache().in_use(), 1, "长度 9 仍在第一页里");
//! assert!(scheduler.finish(sequence.id()).expect("释放页"));
//! ```

#![deny(unsafe_code)]

pub mod backends;
pub mod batch;
pub mod codegen;
pub mod exec;
pub mod ifb;
pub mod kv;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod norm;
pub mod radix;
pub mod scan;
pub mod tiled;
pub mod wmma;

pub use batch::{Admission, Batch, SequenceOutcome};
pub use ifb::{Budget, Candidate, Phase, Verdict, chunk, pack};

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
    /// 提示段已经算进页账的位置数（`0..=prompt_len`）；到了 `prompt_len` 才是「可以解码」。
    prefilled: usize,
    produced: usize,
    max_new: usize,
    page_size: usize,
    state: State,
    stream: crate::exec::Stream,
    /// 完整上下文：提示段在前、生成段在后；[`tokens`](Self::tokens) 切出后半段。
    all: Vec<u32>,
}

impl Sequence {
    /// 由请求参数构造：`prompt` 是提示段的 token id（整段留存，供嵌入查表与历史回看），
    /// `page_size` 来自页池，用于算需要多少页。
    pub fn new(
        id: u64,
        prompt: &[u32],
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
        let prompt_len = prompt.len();
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
            prefilled: 0,
            produced: 0,
            max_new,
            page_size,
            state: State::Waiting,
            stream,
            all: prompt.to_vec(),
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

    /// 提示段已经算进页账的位置数。
    ///
    /// 提示段的 K/V 必须由预填充逐位写进页：解码只算当前位置、历史从页里读，从不回补提示段。
    /// 所以「能解码了没」看的是这个数，不是 `len()`——`len()` 从准入起就等于 `prompt_len`。
    pub fn prefilled(&self) -> usize {
        self.prefilled
    }

    /// 提示段是否已算完（可以直接解码）。
    pub fn prefilled_ready(&self) -> bool {
        self.prefilled >= self.prompt_len
    }

    /// 最多生成多少 token。
    pub fn max_new(&self) -> usize {
        self.max_new
    }

    /// 已经采样出的生成 token，按生成顺序（不含提示段）。
    ///
    /// 序列是这条历史的唯一存放处：端点把它作为结果返回时只该带生成段，提示段由请求侧
    /// 自己持有。需要完整上下文（嵌入查表、注意力回看历史）用 [`context`](Self::context)。
    pub fn tokens(&self) -> &[u32] {
        &self.all[self.prompt_len..]
    }

    /// 完整上下文：提示段在前、生成段在后。
    ///
    /// 模型前向只看得到 [`Sequence`]：整段 token 存在这里，模型不必让调用方再喂一遍历史。
    pub fn context(&self) -> &[u32] {
        &self.all
    }

    /// 记录一个已经采样出来的 token；**不**推进计数、也不碰页账。
    ///
    /// 只对同 crate 可见：[`Batch::emit`] 是"记账 + 推进"成对的唯一入口，绕过它单独记一笔
    /// 会让 `tokens` 与 `produced` 各说一套。上限也只有一处：`max_new`。
    ///
    /// # Errors
    ///
    /// `max_new` 已经记满时返回 `invalid`。
    pub(crate) fn push_token(&mut self, token: u32) -> Result<()> {
        if self.all.len() - self.prompt_len >= self.max_new {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "序列 {} 已经记满 {} 个生成 token：再多一个就超出上限",
                    self.id, self.max_new
                ),
            )
            .with("engine::Sequence::push_token"));
        }
        self.all.push(token);
        Ok(())
    }

    /// 当前状态。
    pub fn state(&self) -> State {
        self.state
    }

    /// 提交所在的流。
    pub fn stream(&self) -> crate::exec::Stream {
        self.stream
    }

    /// 提前结束（采到停止符）：只改状态，不生成 token、也不碰页账。
    ///
    /// 与 `max_new` 到顶的区别只在计数：两种结束都不增不删页，页由 [`Scheduler::finish`] 归还。
    /// 已经结束的序列再 `stop` 是调用方的 bug——重复结束会让"为什么结束"变成两种说法。
    ///
    /// # Errors
    ///
    /// 序列已经结束时返回 `invalid`。
    pub fn stop(&mut self) -> Result<()> {
        if self.state == State::Done {
            return Err(
                Error::new(Code::Invalid, format!("序列 {} 已结束", self.id))
                    .with("engine::Sequence::stop"),
            );
        }
        self.state = State::Done;
        Ok(())
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

    /// 把提示段已算位置推进到 `to`；**不碰页**。
    ///
    /// 只对同 crate 可见：[`Batch::mark_prefilled`] 是"推进 + 登记块"成对的唯一入口，绕过它
    /// 单独推进会让前缀树少登记一个本可复用的块。
    ///
    /// # Errors
    ///
    /// `to` 超出 `0..=prompt_len` 或比当前已算位置更浅时返回 `invalid`。
    pub(crate) fn prefill_to(&mut self, to: usize) -> Result<()> {
        if to > self.prompt_len || to < self.prefilled {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "预填充位置 {to} 不在 {}..={} 之内（不许倒退、不许越过提示段）",
                    self.prefilled, self.prompt_len
                ),
            )
            .with("engine::Sequence::prefill_to"));
        }
        self.prefilled = to;
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
    ///
    /// 这是**准入**的紧凑预算（`pages_for(len)`）：行 0 的空槽在用之前不占页，写入时由
    /// [`Scheduler::write_kv`](crate::Scheduler) 按需补齐；上界看 [`Sequence::pages_required`]。
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

/// 一次准入落定的页：建好的页表与被前缀树接走的 token 数（共享页 × 页宽）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
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

impl Scheduler {
    /// 由 KV 页池构造：页号簿记与页里的数据一起进来。
    ///
    /// 行宽由装载模型的一侧决定（`2 · kv_heads · head_dim` 这类数），页池不替它猜；
    /// 调度器只认"第几个 token 落在哪一页哪一格"，与行里装了什么无关。
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
    ///
    /// 这一步只产出计划、不改外面可见的状态（页池与页表在 [`Scheduler::plan`] 内部已经落定），
    /// 于是「装不下」永远是一条 `None`，而不是调用方要回的半截状态。
    fn plan(&mut self, sequence: &Sequence) -> Result<Option<Plan>> {
        let needed = sequence.pages_needed();
        let width = self.pager.page_size();
        let prefix = self.radix.longest(sequence.context(), width);
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

    /// 建页表：先按块序共享命中页，再补 `fresh` 个新页。共享与新建的账必须同一份——
    /// 只共享页不改位置，下一个请求会把同一段提示重算一遍，而重算的结果正好是它要去读的行。
    fn take_pages(
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
    fn link_prefix(
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

    /// 往页表里追加 `extra` 个新页：分配与入表成对发生，缺页说明空闲页数已被推翻。
    ///
    /// 分配走 [`kv::Pager::alloc`] 而不是页号分配器：页号与页背后的数据是同一个池，
    /// "借到了页却没有那块数据"因此没有可能。
    fn fill(
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
    ///
    /// 这是 [`Scheduler::admit`] 里"永远跑不完就拒绝"的判据本身，单独暴露是因为服务层要
    /// 据此区分两种拒绝：**现在满**（等别人结束，返回可重试的 503）与**永远装不下**
    /// （`max_new` 太大，重试多少次都一样，该返回 400）。两处共用一个定义，规则不会漂。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        sequence.pages_required() <= self.pager.total()
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
        if self.pager.free_pages() < extra {
            return Ok(false);
        }
        self.running[index].table.grow(extra)?;
        Self::fill(&mut self.pager, &mut self.running[index].table, extra, site)?;
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
    ///
    /// 归还走[`kv::Table::truncate`]：摘页与还页成对发生，因此"最后一页"不会因为循环写到
    /// `len - 1` 而被漏掉。
    pub fn finish(&mut self, id: u64) -> Result<bool> {
        let Some(index) = self.position(id) else {
            return Ok(false);
        };
        let mut running = self.running.remove(index);
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
    pub fn pages_of(&self, id: u64) -> Option<&crate::kv::Table> {
        self.position(id).map(|index| &self.running[index].table)
    }

    /// 页号分配器（只读）：服务层报"占了多少页"时用它，页里的数据不在这里。
    pub fn cache(&self) -> &crate::kv::Cache {
        self.pager.cache()
    }

    /// KV 页池（只读）：页号与页里的数据是同一个对象。
    pub fn pager(&self) -> &crate::kv::Pager {
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
    pub fn slot(&self, id: u64, token: usize) -> Result<crate::kv::Slot> {
        let table = self.pages_of(id).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("序列 {id} 不在批里：先 `admit` 再问页位置"),
            )
            .with("engine::Scheduler::slot")
        })?;
        table.slot(token, self.pager.page_size())
    }

    /// 把第 `token` 个位置的 KV 行写进页池（长度必须等于页池的行宽）。
    ///
    /// 位置必须已经有页：页表只盖住"已经生成到的地方"，所以这条路径写不出超前于生成的位置；
    /// 也不会写进一张已经归还的页（[`kv::Pager`] 会拒绝一个失效的 [`kv::Slot`]）。
    ///
    /// # Errors
    ///
    /// 序列不在批里、位置还没有页、行宽不符、或那页已归还。
    pub fn write_kv(&mut self, id: u64, token: usize, values: &[f32]) -> Result<()> {
        let slot = self.slot(id, token)?;
        self.pager.write(slot, values)
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
        let Some(index) = self.position(id) else {
            return Err(Error::new(
                Code::Invalid,
                format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
            )
            .with("engine::Scheduler::reserve"));
        };
        let width = self.pager.page_size();
        if width == 0 {
            return Err(Error::new(Code::Invalid, "页宽为零：算不出要几页")
                .with("engine::Scheduler::reserve"));
        }
        let needed = pages_for(rows, width);
        self.extend(index, needed, "engine::Scheduler::reserve")
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
        pages_for(sequence.len() + 1, self.pager.page_size())
    }

    /// 页宽必须与页池一致：页数是按它算的，差一档就会少占一页。
    fn check_width(&self, sequence: &Sequence, site: &'static str) -> Result<()> {
        if sequence.page_size() == self.pager.page_size() {
            return Ok(());
        }
        Err(Error::new(
            Code::Invalid,
            format!(
                "序列 {} 的页宽 {} 与页池的 {} 不一致：页数会算错",
                sequence.id(),
                sequence.page_size(),
                self.pager.page_size()
            ),
        )
        .with(site))
    }
}

#[cfg(test)]
mod tests;
