//! 连续批：[`Batch`] 把页账（[`Scheduler`]）与它正在跑的 [`Sequence`] 收在一处。
//!
//! 两者分开的代价是"批里有哪些序列"有两个来源：调用方自己维护的列表，与调度器里按 id 记的
//! `running`。两者一漂，[`Scheduler::advance`] 就会对一个不在批里的序列报错，或者序列已经结束
//! 而它的页还占着。[`Batch`] 让这件事只有一个来源：准入时把序列交出去，推进按 id 找，
//! 结束时页与序列一起释放。
//!
//! 采样不在这里：每一步该采哪个 token 由调用方决定（`advance` 只回答"这一步的页有没有着落"），
//! 所以本层不需要认识词表，也不持有 logits。

use crate::kv::Pager;
use crate::{Progress, Scheduler, Sequence};
use diagnostic::{Code, Error, Result};

/// 准入的结局：进去了，或者页不够而原样退回。
///
/// 退回的是**序列本身**而不是一个错误：`max_new` 与页池都合法、只是现在没有空闲页，这正是
/// 服务层要报成"可重试的 503"的情形。做成 `Err` 会让"现在满"与"请求不成立"混成一类，
/// 客户端也就分不清该等一会儿还是该改请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// 已进批：序列由批持有，之后用 [`Sequence::id`] 找它。
    Admitted,
    /// 没进批：页不够，序列原样退回。
    Full(Sequence),
}

/// 一个批：页账 + 它正在跑的序列。
///
/// `sequences` 与 [`Scheduler`] 里的 `running` 一一对应，且顺序相同：同一批的两次 `step`
/// 得到的结果按同一个顺序排列，调用方不必自己对齐。
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    scheduler: Scheduler,
    sequences: Vec<Sequence>,
}

/// 批量结果视图：批内一条序列在本轮 `step` 后的结局。
///
/// 只读借片，生命周期跟随 [`Batch`]：序列事实（历史、计数、状态）与本步 [`Progress`] 成对持有，
/// 两条序列的视图互不重叠，历史不会串到别的条上。批量前向（多序列同一步）的结果按这个视图
/// 逐条对齐；单序列旧路径仍直接从 [`Batch::sequence`] 取事实，语义不变。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceOutcome<'a> {
    sequence: &'a Sequence,
    progress: Progress,
}

impl<'a> SequenceOutcome<'a> {
    /// 这条序列在批里的编号。
    pub fn id(&self) -> u64 {
        self.sequence.id()
    }

    /// 本步结局：生成了一步、已结束，还是没拿到页。
    pub fn progress(&self) -> Progress {
        self.progress
    }

    /// 序列本身（只读事实）：历史、计数与状态从这里取，不经过本视图另存一份。
    pub fn sequence(&self) -> &'a Sequence {
        self.sequence
    }

    /// 已生成 token，按生成顺序（不含提示段）；与 [`Sequence::tokens`] 同一口径。
    pub fn tokens(&self) -> &'a [u32] {
        self.sequence.tokens()
    }

    /// 已生成步数；与 [`Sequence::produced`] 同一口径。
    pub fn produced(&self) -> usize {
        self.sequence.produced()
    }

    /// 这条序列是否已结束。
    pub fn is_done(&self) -> bool {
        self.sequence.is_done()
    }
}

impl Batch {
    /// 由 KV 页池构造一个空批。
    pub fn new(pager: Pager) -> Self {
        Self {
            scheduler: Scheduler::new(pager),
            sequences: Vec::new(),
        }
    }

    /// 一条序列在本轮 `step` 后的批量结局视图：它自己的序列事实与本步进度成对持有。
    ///
    /// 视图是只读借片：历史（[`Sequence::tokens`]）与进度分属两条序列，不会跨条合并；批量结果
    /// 只负责把本轮 [`Batch::step`] 的结局按准入顺序对齐到每条序列。
    pub fn outcome(&self, index: usize, progress: Progress) -> Option<SequenceOutcome<'_>> {
        self.sequences
            .get(index)
            .map(|sequence| SequenceOutcome { sequence, progress })
    }

    /// 尝试准入：页够就把序列交进批里，不够就原样退回。
    ///
    /// # Errors
    ///
    /// 序列自身不合法（重名、已结束、`max_new` 为零）时返回 `invalid`——那是调用方的状态
    /// 错误，与"页不够"不是一回事。
    pub fn admit(&mut self, sequence: Sequence) -> Result<Admission> {
        let share = self.prefix_share(sequence.context());
        if !self.scheduler.admit_shared(&sequence, share)? {
            return Ok(Admission::Full(sequence));
        }
        self.sequences.push(sequence);
        Ok(Admission::Admitted)
    }

    /// 前缀匹配（D12）：在批内找公共前缀最长的序列，返回 `(源序列 id, 可整页共享的页数)`。
    ///
    /// 只按页边界整页共享：不满一页的前缀不拆页（那一页混着非前缀 token），落在新页里；
    /// 共享页数不超过源序列当前页表长度。比较 token 在这里做（序列上下文归批持有），页采纳与
    /// 引用计数归调度器。
    fn prefix_share(&self, context: &[u32]) -> Option<(u64, usize)> {
        let page_size = self.scheduler.page_size();
        let mut best: Option<(usize, u64, usize)> = None;
        for existing in &self.sequences {
            let prefix = common_prefix(context, existing.context());
            let shared_pages = (prefix / page_size).min(
                self.scheduler
                    .pages_of(existing.id())
                    .map(|table| table.len())
                    .unwrap_or(0),
            );
            if shared_pages == 0 {
                continue;
            }
            let better = best.is_none_or(|(best_prefix, _, _)| prefix > best_prefix);
            if better {
                best = Some((prefix, existing.id(), shared_pages));
            }
        }
        best.map(|(_, id, pages)| (id, pages))
    }

    /// 这条序列在这个页池里有没有可能跑完（与 [`Scheduler::fits`] 同源）。
    ///
    /// 服务层据此区分"现在满"（可重试）与"永远装不下"（改请求才有用）；判据本身只有一份。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        self.scheduler.fits(sequence)
    }

    /// 推进批里的一条序列一步（页不够时返回 [`Progress::Blocked`] 且什么都不改）。
    ///
    /// # Errors
    ///
    /// `id` 不在批里时返回 `invalid`。
    pub fn advance(&mut self, id: u64) -> Result<Progress> {
        let index = self.position(id).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
            )
            .with("engine::Batch::advance")
        })?;
        let Self {
            scheduler,
            sequences,
        } = self;
        scheduler.advance(&mut sequences[index])
    }

    /// 记录一个 token 并推进一步：两者成对，所以"生成了却没记账"写不出来。
    ///
    /// `Blocked` 时不记账：这一步没有生成 token，历史里也就不能多一个。返回 `Blocked` 时
    /// 序列与页表都与调用前一致，调用方可以稍后原样重试——重试时要重新采样，因此上一步的
    /// token 不能提前留在历史里。
    ///
    /// # Errors
    ///
    /// `id` 不在批里、序列已结束、页宽不符，或 `max_new` 已经记满。
    pub fn emit(&mut self, id: u64, token: u32) -> Result<Progress> {
        let progress = self.advance(id)?;
        if progress != Progress::Blocked {
            // 这一格的页已经占到，`produced` 也因此不超过 `max_new`：记账必然放得下。
            self.record(id, token)?;
        }
        Ok(progress)
    }

    /// 把这一步的 token 记进序列；只在真的生成了一步之后调用。
    fn record(&mut self, id: u64, token: u32) -> Result<()> {
        self.sequence_mut(id)
            .ok_or_else(|| {
                Error::new(
                    Code::Internal,
                    format!("序列 {id} 在推进之后离开了批：页账与序列不一致"),
                )
                .with("engine::Batch::emit")
            })?
            .push_token(token)
    }

    /// 把第 `token` 个位置的 KV 行写进这条序列的页（长度必须等于页池的行宽）。
    ///
    /// 批是"序列 + 页账"的唯一来源，KV 也经它写：调用方只需要 id 与位置，不必自己拿页表
    /// 再算一次偏移，也不会写到一条已经 `finish` 的序列上。
    ///
    /// # Errors
    ///
    /// `id` 不在批里、位置还没有页、行宽不符，或那页已归还。
    pub fn write_kv(&mut self, id: u64, token: usize, values: &[f32]) -> Result<()> {
        self.scheduler.write_kv(id, token, values)
    }

    /// 读回批里某条序列第 `token` 个位置的 KV 行。
    ///
    /// # Errors
    ///
    /// 同 [`Batch::write_kv`] 的前两种情形。
    pub fn kv(&self, id: u64, token: usize) -> Result<&[f32]> {
        self.scheduler.kv(id, token)
    }

    /// 批里每条序列各推进一步，结果按 [`Batch::sequences`] 的顺序排列。
    ///
    /// 单条序列的 [`Progress::Blocked`] 不会中断整批：它只是"这条现在没页"，其余序列照样推进。
    /// 出错则立刻停下——`advance` 的错误只来自调用方的状态错误，不是页不够。
    ///
    /// # Errors
    ///
    /// 任一序列的推进返回 `invalid`（例如它的页宽与页池不一致）。
    pub fn step(&mut self) -> Result<Vec<Progress>> {
        let Self {
            scheduler,
            sequences,
        } = self;
        let mut outcomes = Vec::with_capacity(sequences.len());
        for sequence in sequences.iter_mut() {
            outcomes.push(scheduler.advance(sequence)?);
        }
        Ok(outcomes)
    }

    /// 结束一条序列：归还它的页，并把它从批里摘掉。
    ///
    /// 返回它是否在批里——不在批里不是错误（重复 `finish` 不该把调用方打成 `Err`）。
    ///
    /// # Errors
    ///
    /// 页池拒绝归还（页不在池里或重复归还）：那是内部账目出错。
    pub fn finish(&mut self, id: u64) -> Result<bool> {
        let removed = self.scheduler.finish(id)?;
        if let Some(index) = self.position(id) {
            self.sequences.remove(index);
        }
        Ok(removed)
    }

    /// 批里的某条序列（只读）。
    pub fn sequence(&self, id: u64) -> Option<&Sequence> {
        self.sequences.iter().find(|sequence| sequence.id() == id)
    }

    /// 批里的某条序列（可写）：调用方在推进之前把采样结果记进自己的缓冲时用得上。
    pub fn sequence_mut(&mut self, id: u64) -> Option<&mut Sequence> {
        self.sequences
            .iter_mut()
            .find(|sequence| sequence.id() == id)
    }

    /// 批内第 `index` 条序列（只读）：位置即 [`Batch::sequences`] 的准入顺序。
    ///
    /// 批量视图按序号取视图：模型侧按批内几何逐条前向时，用序号对齐本轮结果，不必自己再按
    /// id 重排。`index` 越界返回 `None`，批内顺序在同一轮 `step` 内稳定。
    pub fn sequence_at(&self, index: usize) -> Option<&Sequence> {
        self.sequences.get(index)
    }

    /// 本步批量结局：每条在跑序列一条进度，按准入顺序排列，与 [`Batch::sequences`] 一一对应。
    ///
    /// 批量结果视图不另存历史：进度只回答"这一步生成没生成"，历史仍在各自的 [`Sequence`] 里，
    /// 两条序列的 `tokens` 不会在这个视图里串台。调用方按序号与 [`Batch::sequence_at`] 对齐。
    pub fn outcomes(&self, step: &[Progress]) -> Vec<SequenceOutcome<'_>> {
        self.sequences
            .iter()
            .zip(step)
            .map(|(sequence, progress)| SequenceOutcome {
                sequence,
                progress: *progress,
            })
            .collect()
    }

    /// 全部序列，按准入顺序。
    pub fn sequences(&self) -> &[Sequence] {
        &self.sequences
    }

    /// 全部序列号，按准入顺序。
    pub fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.sequences.iter().map(Sequence::id)
    }

    /// 现在拿不到页、必须先等别人结束的序列（与 [`Scheduler::blocked`] 同源）。
    pub fn blocked(&self) -> Vec<u64> {
        self.scheduler.blocked()
    }

    /// 批里正在跑的序列数。
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// 批是否为空。
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// 页账（只读）。
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// 页账（可写）：服务层要单独准入、释放或问页位置时用。
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    fn position(&self, id: u64) -> Option<usize> {
        self.sequences
            .iter()
            .position(|sequence| sequence.id() == id)
    }
}

/// 两段上下文的公共前缀长度；逐位比较有串行依赖，保留 zip 折叠。
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
#[path = "../tests/batch_tests.rs"]
mod tests;
