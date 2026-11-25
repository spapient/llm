//! 连续批：[`Batch`] 把页账（[`Scheduler`]）与它正在跑的 [`Sequence`] 收在一处。
//!
//! 批是"序列 + 页账"的唯一来源：KV 经它写、token 经它记、结束经它还页。两半因此不会各记一份
//! ——`Sequence` 不知道页，`Scheduler` 不知道历史，谁也写不出"生成了却没页"或"占了页却没序列"。

use std::ops::Range;

use crate::ifb::{Budget, Candidate, Verdict, chunk, pack};
use crate::kv::Pager;
use crate::{Progress, Scheduler, Sequence, invalid};
use diagnostic::{Code, Error, Result};

#[cfg(test)]
#[path = "../tests/batch.rs"]
mod tests;

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
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    scheduler: Scheduler,
    sequences: Vec<Sequence>,
}

/// 批量结果视图：批内一条序列在本轮 `step` 后的结局。
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

    /// 尝试准入：页够就把序列交进批里，不够就原样退回。
    ///
    /// 前缀页由调度器向满块前缀树要（跨请求）：批不再自己扫活序列找公共前缀——那样源序列一
    /// 结束、页一归还，前缀就跟着没了，而"同样的系统提示词再来一个请求"恰恰是最常见的复用。
    ///
    /// 接走的共享前缀同时把「已算位置」推上去：那些 token 的 K/V 就在接走的那几页里，重算一遍
    /// 只会写出同样的行。位置账与页账因此永远是同一份——只共享页不改位置，端点会把整段提示再算
    /// 一遍，而它要写的正好是它刚打算去读的那些行。
    ///
    /// # Errors
    ///
    /// 序列自身不合法（重名、已结束、`max_new` 为零）时返回 `invalid`——那是调用方的状态
    /// 错误，与"页不够"不是一回事。
    pub fn admit(&mut self, mut sequence: Sequence) -> Result<Admission> {
        let Some(prefilled) = self.scheduler.admit(&sequence)? else {
            return Ok(Admission::Full(sequence));
        };
        sequence.prefill_to(prefilled)?;
        self.sequences.push(sequence);
        Ok(Admission::Admitted)
    }

    /// 推进批里的一条序列一步（页不够时返回 [`Progress::Blocked`] 且什么都不改）。
    ///
    /// # Errors
    ///
    /// `id` 不在批里时返回 `invalid`。
    pub fn advance(&mut self, id: u64) -> Result<Progress> {
        let idx = self.position(id).ok_or_else(|| {
            invalid(
                "engine::Batch::advance",
                format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
            )
        })?;
        self.scheduler.advance(&mut self.sequences[idx])
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
            self.record(id, token)?;
        }
        Ok(progress)
    }

    /// 记下一个已产出的 token：这一格的页已经占到，`produced` 也因此不超过 `max_new`。
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

    /// 把第 `token` 个位置的 KV 行写进这条序列的页（长度必须等于页池的行宽）。
    ///
    /// 批是"序列 + 页账"的唯一来源，KV 也经它写：调用方只需要 id 与位置，不必自己拿页表
    /// 再算一次偏移，也不会写到一条已经 `finish` 的序列上。
    ///
    /// 写满一页时顺手把这一块登记进前缀树：前缀复用的单位是"算好的块"，而只有写完那一刻才
    /// 知道它算完了。登记只加一份页引用，不动这条序列的页表——之后这条序列结束归还自己那份，
    /// 页仍由树持着，下一个同前缀请求照样命中。
    ///
    /// 预填充走同一条路：每个提示位置写回它自己那一行，写满一页就登记那一块，于是提示段的前缀
    /// 在预填充过程中就已经可复用，不必等解码。
    ///
    /// # Errors
    ///
    /// `id` 不在批里、位置还没有页、行宽不符，或那页已归还。
    pub fn write_kv(&mut self, id: u64, token: usize, values: &[f32]) -> Result<()> {
        self.scheduler.write_kv(id, token, values)?;
        self.settle(id, token + 1)
    }

    /// 把提示段已算位置推进到 `to`，并把这段路上新写满的块登记进前缀树。
    ///
    /// 为什么推进与登记必须成对：块只有"行已经落定"之后才可复用。预填充写的是提示段的行，
    /// 登记却按 [`Sequence::prefilled`] 而不是按写到的行——写提前一格（第 `at` 步写第
    /// `at + 1` 行），登记若跟着写走就会把还没算的块当成算好的。
    ///
    /// `to` 不超过提示长度、也不倒退，由 `Sequence::prefill_to` 校验。
    ///
    /// # Errors
    ///
    /// `id` 不在批里、`to` 越界或倒退、或登记时页池拒绝加引用。
    pub fn mark_prefilled(&mut self, id: u64, to: usize) -> Result<()> {
        self.sequence_mut(id)
            .ok_or_else(|| {
                invalid(
                    "engine::Batch::mark_prefilled",
                    format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
                )
            })?
            .prefill_to(to)?;
        self.settle(id, to)
    }

    /// 把这条序列的页表扩到盖住前 `rows` 行：预填充写多行之前先向页池要页。
    ///
    /// 与 [`Batch::write_kv`] 分开的理由是错误要说真话：一次写多行时缺的不是「位置无页」，而是
    /// 「页没预留」，两件事报同一句话就把调用点指错了。
    ///
    /// 返回「要到了没有」：`false` 是这一轮页不够——调用方把这块留到下一轮，而不是把请求判错。
    ///
    /// # Errors
    ///
    /// `id` 不在批里、页宽为零。
    pub fn reserve(&mut self, id: u64, rows: usize) -> Result<bool> {
        self.scheduler.reserve(id, rows)
    }

    /// 读回批里某条序列第 `token` 个位置的 KV 行。
    ///
    /// # Errors
    ///
    /// 同 [`Batch::write_kv`] 的前两种情形。
    pub fn kv(&self, id: u64, token: usize) -> Result<&[f32]> {
        self.scheduler.kv(id, token)
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
        if let Some(idx) = self.position(id) {
            self.sequences.remove(idx);
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

    /// 批内第 `idx` 条序列（只读）：位置即 [`Batch::sequences`] 的准入顺序。
    pub fn sequence_at(&self, idx: usize) -> Option<&Sequence> {
        self.sequences.get(idx)
    }

    /// 本步批量结局：每条在跑序列一条进度，按准入顺序排列，与 [`Batch::sequences`] 一一对应。
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

    /// 一条序列在本轮 `step` 后的批量结局视图：它自己的序列事实与本步进度成对持有。
    pub fn outcome(&self, idx: usize, progress: Progress) -> Option<SequenceOutcome<'_>> {
        self.sequences
            .get(idx)
            .map(|sequence| SequenceOutcome { sequence, progress })
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

    /// 这条序列在这个页池里有没有可能跑完（与 [`Scheduler::fits`] 同源）。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        self.scheduler.fits(sequence)
    }

    /// 这一轮按 IFB 预算该跑哪些行：`Packed.rows` 是**本轮的行与序**（生成在前、上下文在后）。
    #[must_use]
    pub fn round(&self, budget: Budget) -> Verdict {
        let candidates: Vec<Candidate> = self
            .sequences
            .iter()
            .map(|sequence| {
                let remaining = sequence.prompt_len().saturating_sub(sequence.prefilled());
                if remaining == 0 {
                    return Candidate::generation(sequence.id());
                }
                let rows = chunk(remaining, sequence.page_size(), budget.tokens());
                Candidate::prefill(sequence.id(), rows)
            })
            .filter(|candidate| candidate.rows() > 0)
            .collect();
        pack(budget, &candidates)
    }

    /// 某个序列在批里的下标（插入顺序即批内顺序）。
    fn position(&self, id: u64) -> Option<usize> {
        self.sequences
            .iter()
            .position(|sequence| sequence.id() == id)
    }

    /// 把前 `to` 行里**走完的块**登记进前缀树。
    ///
    /// 块 `b` 落定的条件是 `(b + 1) · 页宽 <= to`，而每写一行都从 0 扫到 `to` 会让一次预填充
    /// 付出 `O(提示长² / 页宽)` 次前缀树行走、然后听它逐块回答"已在"。所以起点取序列自己的块账：
    /// 它只往前推，走过的块不必再问。跳着写的调用方也不用特殊照顾——缺的块正落在没走过的区间里。
    ///
    /// 登记要的是「上下文 + 一页」：上下文只借不克隆（它在序列里），页号是 `Copy` 的
    /// [`crate::kv::Page`]，因此页池的可变借用与序列的只读借用可以先后各取一次，不必先复制一份快照。
    fn settle(&mut self, id: u64, to: usize) -> Result<()> {
        let Self {
            scheduler,
            sequences,
        } = self;
        let width = scheduler.page_size();
        if width == 0 {
            return Ok(());
        }
        let Some(at) = sequences.iter().position(|sequence| sequence.id() == id) else {
            return Ok(());
        };
        let blocks = to.min(sequences[at].ctx().len()) / width;
        let range = sequences[at].settle(blocks);
        let history = sequences[at].ctx();
        publish_range(scheduler, history, id, range)
    }
}

/// 把一轮里走完的块逐块登记进前缀树：页表里没有的块跳过（序列自己已判它不可发布）。
fn publish_range(
    scheduler: &mut Scheduler,
    history: &[u32],
    id: u64,
    blocks: Range<usize>,
) -> Result<()> {
    for block in blocks {
        let Some(page) = scheduler
            .pages_of(id)
            .and_then(|table| table.pages().get(block).copied())
        else {
            continue;
        };
        scheduler.publish(history, block, page)?;
    }
    Ok(())
}
