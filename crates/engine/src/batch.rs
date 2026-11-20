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

    /// 这条序列在这个页池里有没有可能跑完（与 [`Scheduler::fits`] 同源）。
    ///
    /// 服务层据此区分"现在满"（可重试）与"永远装不下"（改请求才有用）；判据本身只有一份。
    pub fn fits(&self, sequence: &Sequence) -> bool {
        self.scheduler.fits(sequence)
    }

    /// 这一轮按 IFB 预算该跑哪些行：`Packed.rows` 是**本轮的行与序**（生成在前、上下文在后）。
    ///
    /// 相位由序列自身的事实决定，不由调用方挑：提示段没算完就是上下文（块大小走
    /// [`crate::ifb::chunk`] 的页对齐算式），算完了就是生成（一轮一行）。于是"先跑谁"不是调度参数，
    /// 而是 [`crate::ifb::pack`] 在预算下的一条算式——挤压时只挤上下文，生成行一条不丢。
    ///
    /// 零行的候选不进结果：`chunk` 返回 0 是"这一轮不做上下文"（剩余不足一整页），把它当成
    /// 一条 0 行候选排进来，调用方就得自己再判一次"这行要不要跑"。
    #[must_use]
    pub fn round(&self, budget: crate::ifb::Budget) -> crate::ifb::Verdict {
        let candidates: Vec<crate::ifb::Candidate> = self
            .sequences
            .iter()
            .map(|sequence| {
                let remaining = sequence.prompt_len().saturating_sub(sequence.prefilled());
                if remaining == 0 {
                    return crate::ifb::Candidate::generation(sequence.id());
                }
                let rows = crate::ifb::chunk(remaining, sequence.page_size(), budget.tokens());
                crate::ifb::Candidate::context(sequence.id(), rows)
            })
            .filter(|candidate| candidate.rows() > 0)
            .collect();
        crate::ifb::pack(budget, &candidates)
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
        // 写了第 `token` 行 ⇒ 行 0..=token 落定，共 `token + 1` 行。
        self.settle_through(id, token + 1)
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

    /// 把提示段已算位置推进到 `to`，并把沿途新写满的块登记进前缀树。
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
                Error::new(
                    Code::Invalid,
                    format!("序列 {id} 不在批里：先 `admit`（或它已经 `finish`）"),
                )
                .with("engine::Batch::mark_prefilled")
            })?
            .prefill_to(to)?;
        // 已算 `to` 个位置 ⇒ 行 0..to 落定，与写路径同一口径。
        self.settle_through(id, to)
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

    /// 写到第 `token` 行之后登记：已落定的行数就是这一行之后的行数。
    /// 把已落定 `rows` 行里凑满的块登记进前缀树。
    ///
    /// 只登记**整块且整块落在已落定行数之内**的前缀：块内 token 从序列上下文切（前缀树比对的
    /// 就是这些数），页号从页表取——两者都只在"这一块已经算进页里"之后才有意义。哪些块算数只由
    /// `rows` 决定，调用方负责让它等于已落定的行数，因此这里不再自己推偏移。
    ///
    /// 扫描从第 0 块开始而不是从"新落定的第一块"开始：登记对已登记的块返回 `false`、不加第二份
    /// 引用，重扫只花一次比对；代价远小于"从哪开始"这个参数写错一次就漏掉一块。
    ///
    /// 不登记的部分不是错误：不足一块的尾巴、或已被别人登记过的块，都只是"这次没有新前缀"。
    fn settle_through(&mut self, id: u64, rows: usize) -> Result<()> {
        let Self {
            scheduler,
            sequences,
        } = self;
        let width = scheduler.page_size();
        let Some((context, pages)) = settleable(sequences, scheduler, id) else {
            return Ok(());
        };
        for block in settled_blocks(rows, width, context.len(), pages.len()) {
            scheduler.publish(&context, block, pages[block])?;
        }
        Ok(())
    }
}

/// 登记块要先只读地把上下文与页号收齐：`publish` 要独占页池，而这两样还得从只读的序列与页表里
/// 取，两个借用不能同时活着。整段上下文只克隆一次，逐块登记时复用。
///
/// `None` 是「现在没有可登记的块」——序列已不在批里、页表没了、或页宽为零（零页不是算好的块）。
fn settleable(
    sequences: &[Sequence],
    scheduler: &Scheduler,
    id: u64,
) -> Option<(Vec<u32>, Vec<crate::kv::Page>)> {
    let sequence = sequences.iter().find(|sequence| sequence.id() == id)?;
    if scheduler.page_size() == 0 {
        return None;
    }
    let table = scheduler.pages_of(id)?;
    Some((sequence.context().to_vec(), table.pages().to_vec()))
}

/// 前 `rows` 行覆盖到的满块：块尾已落定（`(block + 1) · width <= 上下文长`）且页表里真有这一页。
fn settled_blocks(rows: usize, width: usize, context_len: usize, pages: usize) -> Vec<usize> {
    if width == 0 {
        return Vec::new();
    }
    (0..rows / width)
        .filter(|block| (block + 1) * width <= context_len && *block < pages)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::Stream;
    use crate::kv::{Cache, Page};

    fn pool(total: usize, page_size: usize) -> Pager {
        Pager::new(Cache::new(total, page_size).expect("页池"), 2).expect("行宽")
    }

    fn sequence(id: u64, prompt: usize, max_new: usize, page_size: usize) -> Sequence {
        let prompt = vec![0u32; prompt];
        Sequence::new(id, &prompt, max_new, page_size, Stream::new(0)).expect("序列")
    }

    /// 指定提示 token 的序列：用于需要避开前缀整页共享的夹具（两条提示不同就不会采纳同一页）。
    fn sequence_with(id: u64, tokens: &[u32], max_new: usize, page_size: usize) -> Sequence {
        Sequence::new(id, tokens, max_new, page_size, Stream::new(0)).expect("序列")
    }

    #[test]
    fn an_admitted_sequence_is_owned_by_the_batch() {
        let mut batch = Batch::new(pool(4, 4));
        assert!(batch.is_empty());
        assert_eq!(
            batch.admit(sequence(7, 4, 4, 4)).expect("准入"),
            Admission::Admitted
        );
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.ids().collect::<Vec<u64>>(), [7]);
        let admitted = batch.sequence(7).expect("在批里");
        assert_eq!(admitted.state(), crate::State::Waiting);
        assert_eq!(batch.scheduler().cache().in_use(), 1);
    }

    /// "现在满"必须原样退回序列：调用方要能把它排队，而不是丢掉一条合法请求。
    #[test]
    fn a_full_batch_hands_the_sequence_back() {
        let mut batch = Batch::new(pool(1, 4));
        let rejected = match batch.admit(sequence(1, 4, 4, 4)).expect("准入判定") {
            Admission::Full(sequence) => sequence,
            Admission::Admitted => panic!("页池只有 1 页，装不下 2 页的最坏情况"),
        };
        assert_eq!(rejected.id(), 1);
        assert!(batch.is_empty());
        assert_eq!(batch.scheduler().cache().in_use(), 0, "退回不能占页");
    }

    #[test]
    fn step_advances_every_sequence_in_admission_order() {
        let mut batch = Batch::new(pool(8, 4));
        batch.admit(sequence(1, 0, 4, 4)).expect("准入");
        batch.admit(sequence(2, 0, 4, 4)).expect("准入");
        assert_eq!(
            batch.step().expect("一步"),
            [Progress::Running, Progress::Running]
        );
        assert_eq!(batch.sequence(1).expect("在批里").produced(), 1);
        assert_eq!(batch.sequence(2).expect("在批里").produced(), 1);
        assert_eq!(batch.ids().collect::<Vec<u64>>(), [1, 2]);
    }

    /// 页用尽时只有"下一步确实要新页"的那条被挡住：其余序列照常推进，批不会整体停摆。
    #[test]
    fn a_blocked_sequence_does_not_stop_the_batch() {
        let mut batch = Batch::new(pool(2, 4));
        batch.admit(sequence(1, 0, 8, 4)).expect("准入");
        batch.admit(sequence(2, 4, 4, 4)).expect("准入");
        assert_eq!(batch.scheduler().cache().free_pages(), 0);
        assert_eq!(batch.blocked(), [2], "只有第 2 条需要新页");

        assert_eq!(
            batch.step().expect("一步"),
            [Progress::Running, Progress::Blocked]
        );
        assert_eq!(batch.sequence(1).expect("在批里").produced(), 1);
        assert_eq!(
            batch.sequence(2).expect("在批里").produced(),
            0,
            "被挡住的那条什么都不该改变"
        );
    }

    #[test]
    fn finishing_returns_pages_and_removes_the_sequence() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence(3, 8, 4, 4)).expect("准入");
        assert_eq!(batch.scheduler().cache().in_use(), 2);
        assert!(batch.finish(3).expect("结束"));
        assert!(batch.is_empty());
        assert_eq!(batch.scheduler().cache().in_use(), 0, "页全部归还");
        assert!(!batch.finish(3).expect("重复结束是恒等"), "重复结束不算错");
        assert_eq!(
            batch.advance(3).unwrap_err().code(),
            Code::Invalid,
            "已经离开批的序列不能再推进"
        );
    }

    #[test]
    fn a_stopped_sequence_cannot_be_advanced() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence(5, 0, 8, 4)).expect("准入");
        batch.sequence_mut(5).expect("在批里").stop().expect("停止");
        assert!(batch.sequence(5).expect("在批里").is_done());
        assert_eq!(batch.advance(5).unwrap_err().code(), Code::Invalid);
        // 页仍然占着，直到 `finish`：结束本身不碰页账。
        assert_eq!(batch.scheduler().cache().in_use(), 1);
    }

    #[test]
    fn a_duplicate_id_is_refused_rather_than_silently_merged() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence(1, 0, 4, 4)).expect("准入");
        assert_eq!(
            batch.admit(sequence(1, 0, 4, 4)).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(batch.len(), 1);
    }

    /// 记账与推进成对：`emit` 之后计数与历史各长一格，顺序与采样顺序一致。
    #[test]
    fn emit_records_the_token_in_the_sequence() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence(1, 0, 3, 4)).expect("准入");
        assert!(batch.sequence(1).expect("在批里").tokens().is_empty());

        assert_eq!(batch.emit(1, 7).expect("生成"), Progress::Running);
        assert_eq!(batch.emit(1, 8).expect("生成"), Progress::Running);
        let admitted = batch.sequence(1).expect("在批里");
        assert_eq!(admitted.tokens(), [7, 8], "历史按生成顺序");
        assert_eq!(admitted.produced(), 2, "计数与历史同步");

        assert_eq!(batch.emit(1, 9).expect("最后一步"), Progress::Done);
        assert_eq!(
            batch.sequence(1).expect("在批里").tokens(),
            [7, 8, 9],
            "走到上限的那一步也要记账"
        );
    }

    /// 被挡的那一步没有生成：历史里不能多一个 token，否则重试会看见一个不存在的输出。
    #[test]
    fn a_blocked_step_does_not_record_a_token() {
        // 两页、每页 2 个 token：两条序列各占一页，各自下一步都要新页。提示刻意不同，
        // 不触发前缀整页共享，才能复现"两条各独占一页、池被占满"的阻塞账。
        let mut batch = Batch::new(pool(2, 2));
        batch.admit(sequence_with(1, &[0, 0], 1, 2)).expect("准入");
        batch.admit(sequence_with(2, &[1, 1], 1, 2)).expect("准入");
        assert_eq!(batch.blocked(), [1, 2], "两条都在等页");

        assert_eq!(batch.emit(1, 5).expect("被挡不是错误"), Progress::Blocked);
        let blocked = batch.sequence(1).expect("在批里");
        assert!(blocked.tokens().is_empty(), "没有生成就没有历史");
        assert_eq!(blocked.produced(), 0, "被挡不推进计数");

        // 腾出一页之后重试成功：同一个 token 这时才被记住。
        assert!(batch.finish(2).expect("腾页"));
        assert_eq!(batch.emit(1, 5).expect("重试"), Progress::Done);
        assert_eq!(batch.sequence(1).expect("在批里").tokens(), [5]);
    }

    /// 把一条序列的前若干满块算完：逐位置写 KV，写满一页就登记进前缀树。
    fn prefill(batch: &mut Batch, id: u64, blocks: usize, width: usize) -> Vec<Page> {
        let row = vec![0.0f32; width];
        for token in 0..blocks * width {
            batch.write_kv(id, token, &row).expect("写 KV");
        }
        batch
            .scheduler()
            .pages_of(id)
            .expect("页表")
            .pages()
            .to_vec()
    }

    /// 已算好的块被下一个同前缀请求采纳：同一页号进新表，池里只占一个页位。
    #[test]
    fn a_written_block_is_adopted_by_the_next_identical_request() {
        let mut batch = Batch::new(pool(4, 2));
        batch.admit(sequence_with(1, &[0, 0], 1, 2)).expect("准入");
        let shared = prefill(&mut batch, 1, 1, 2)[0];

        batch.admit(sequence_with(2, &[0, 0], 1, 2)).expect("准入");
        assert_eq!(
            batch.scheduler().pages_of(2).expect("新表").pages()[0],
            shared,
            "已算好的块被采纳进新表"
        );
        assert_eq!(
            batch.scheduler().cache().reference_count(shared),
            3,
            "源表、树、新表各持一份"
        );
        assert_eq!(
            batch.scheduler().cache().in_use(),
            1,
            "三份引用只占一个池页"
        );
    }

    /// 源序列结束后前缀仍命中：树持着那一页，新请求照样接手，页不会回池清零。
    #[test]
    fn a_written_block_outlives_the_sequence_that_computed_it() {
        let mut batch = Batch::new(pool(4, 2));
        batch.admit(sequence_with(1, &[3, 3], 1, 2)).expect("准入");
        let shared = prefill(&mut batch, 1, 1, 2)[0];
        batch.write_kv(1, 0, &[9.0, 9.0]).expect("改写前缀行");

        assert!(batch.finish(1).expect("源序列结束"));
        assert_eq!(
            batch.scheduler().cache().reference_count(shared),
            1,
            "只剩树这一份"
        );
        assert!(batch.scheduler().cache().is_live(shared), "页不在池里");

        batch.admit(sequence_with(2, &[3, 3], 1, 2)).expect("准入");
        assert_eq!(
            batch.scheduler().pages_of(2).expect("新表").pages()[0],
            shared,
            "跨请求命中：源序列已经不在了"
        );
        assert_eq!(batch.kv(2, 0).expect("读共享前缀行"), [9.0, 9.0]);

        assert!(batch.finish(2).expect("共享方结束"));
        assert!(batch.scheduler().cache().is_live(shared), "树仍持着前缀页");
    }

    /// 引用计数跨结束顺序正确：每张表一份、树一份，谁先结束都不提前回池。
    #[test]
    fn a_shared_block_is_released_only_when_the_last_holder_goes() {
        let mut batch = Batch::new(pool(4, 2));
        batch
            .admit(sequence_with(1, &[5, 5], 1, 2))
            .expect("准入源序列");
        let shared = prefill(&mut batch, 1, 1, 2)[0];
        batch
            .admit(sequence_with(2, &[5, 5], 1, 2))
            .expect("准入共享序列");

        assert_eq!(
            batch.scheduler().cache().reference_count(shared),
            3,
            "源表、树、共享表各一份"
        );
        assert!(batch.finish(1).expect("源序列先结束"));
        assert_eq!(batch.scheduler().cache().reference_count(shared), 2);
        assert!(batch.scheduler().cache().is_live(shared));
        assert_eq!(batch.scheduler().cache().in_use(), 1, "共享页只占一个池页");

        assert!(batch.finish(2).expect("共享方结束"));
        assert_eq!(
            batch.scheduler().cache().reference_count(shared),
            1,
            "只剩树这一份"
        );
        assert!(
            !batch.finish(2).expect("重复结束恒等"),
            "重复结束不再次归还"
        );
    }

    /// 三方可共享同一前缀，谁先结束都只减自己那一份。
    #[test]
    fn three_requests_share_one_block() {
        let mut batch = Batch::new(pool(6, 2));
        batch
            .admit(sequence_with(1, &[4, 4], 1, 2))
            .expect("第一条");
        let shared = prefill(&mut batch, 1, 1, 2)[0];
        batch
            .admit(sequence_with(2, &[4, 4], 1, 2))
            .expect("第二条");
        batch
            .admit(sequence_with(3, &[4, 4], 1, 2))
            .expect("第三条");

        assert_eq!(batch.scheduler().cache().reference_count(shared), 4);
        assert_eq!(batch.scheduler().cache().in_use(), 1, "四方共享一个池页");

        assert!(batch.finish(2).expect("第二条先结束"));
        assert_eq!(batch.scheduler().cache().reference_count(shared), 3);
        assert!(batch.finish(1).expect("第一条结束"));
        assert_eq!(batch.scheduler().cache().reference_count(shared), 2);
        assert!(batch.finish(3).expect("第三条结束"));
        assert_eq!(
            batch.scheduler().cache().reference_count(shared),
            1,
            "只剩树这一份"
        );
    }

    /// 首页内发散的前缀不共享：新序列另取新页，两条各占一个池页位。
    #[test]
    fn diverging_prefix_takes_fresh_pages() {
        let mut batch = Batch::new(pool(4, 2));
        batch.admit(sequence_with(1, &[0, 0], 1, 2)).expect("准入");
        batch.admit(sequence_with(2, &[0, 1], 1, 2)).expect("准入");

        let first = batch.scheduler().pages_of(1).expect("源表");
        let second = batch.scheduler().pages_of(2).expect("新表");
        assert_ne!(second.pages()[0], first.pages()[0], "首页内发散，不共享");
        assert_eq!(batch.scheduler().cache().in_use(), 2, "两条各独占一页");
    }

    /// 公共前缀不满一页时不拆页：整页共享只对齐到页边界。
    #[test]
    fn prefix_shorter_than_one_page_is_not_split() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence_with(1, &[0, 0], 1, 4)).expect("准入");
        batch.admit(sequence_with(2, &[0, 0], 1, 4)).expect("准入");

        let first = batch.scheduler().pages_of(1).expect("源表");
        let second = batch.scheduler().pages_of(2).expect("新表");
        assert_ne!(
            second.pages()[0],
            first.pages()[0],
            "前缀 2 token < 页宽 4，落在新页"
        );
        assert_eq!(batch.scheduler().cache().in_use(), 2);
    }

    /// 记满之后再记是调用方多记了一次：上限只有 `max_new` 一处。
    #[test]
    fn recording_past_the_limit_is_refused() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence(1, 0, 2, 4)).expect("准入");
        assert_eq!(batch.emit(1, 1).expect("第一步"), Progress::Running);
        assert_eq!(batch.emit(1, 2).expect("第二步"), Progress::Done);
        let sequence = batch.sequence_mut(1).expect("在批里");
        assert_eq!(
            sequence.push_token(3).unwrap_err().code(),
            Code::Invalid,
            "第三个 token 超出 max_new"
        );
        assert_eq!(sequence.tokens(), [1, 2], "失败的记账不留痕");
    }

    /// 批内两条以上、不同长度与不同前缀的序列同批跑多步：各自历史与进度对拍，不串台。
    ///
    /// 三条序列：一条短提示、一条长提示、一条与首条共享整页前缀；每轮同批 `step`，再按 id
    /// 把采样结果 `emit` 记进各自历史。轮次结束后逐条核对生成数、生成历史与上下文长度，
    /// 共享前缀那条的提示段不能被另两条的生成 token 顶替。已跑满的序列不再参与记账。
    #[test]
    fn multi_sequence_batch_runs_their_own_history_and_progress() {
        // 页宽 2：前缀整页共享；页池 12 页，三条各自提示与跨页增长都放得下，不互相挡页。
        let mut batch = Batch::new(pool(12, 2));
        batch
            .admit(sequence_with(1, &[7, 7], 3, 2))
            .expect("准入第一条");
        batch
            .admit(sequence_with(2, &[8, 8, 9, 9], 2, 2))
            .expect("准入第二条");
        batch
            .admit(sequence_with(3, &[7, 7], 2, 2))
            .expect("准入第三条（与第一条共享整页前缀）");

        // 每轮三条各走一步；采样 token 按序列各自编号，记账与推进成对（`emit` 内部先预留位再记账）。
        for round in 0..3 {
            let ids: Vec<u64> = batch.ids().collect();
            let mut recorded: Vec<Progress> = Vec::with_capacity(ids.len());
            for id in ids {
                if batch.sequence(id).expect("在批里").is_done() {
                    recorded.push(Progress::Done);
                    continue;
                }
                let token = id as u32 * 10 + round as u32;
                let progress = batch.emit(id, token).expect("记账推进");
                recorded.push(progress);
            }
            // 批量视图把本轮结局按准入顺序对齐到每条序列：不串台。
            let view = batch.outcomes(&recorded);
            assert_eq!(view.len(), 3, "批量视图与准入序列一一对应");
        }

        let first = batch.sequence(1).expect("第一条在批里");
        let second = batch.sequence(2).expect("第二条在批里");
        let third = batch.sequence(3).expect("第三条在批里");

        assert_eq!(first.tokens(), [10, 11, 12], "第一条历史按自己的生成顺序");
        assert_eq!(second.tokens(), [20, 21], "第二条先跑满 max_new");
        assert_eq!(third.tokens(), [30, 31], "第三条历史不与第一条串");
        assert_eq!(first.produced(), 3);
        assert_eq!(second.produced(), 2);
        assert_eq!(third.produced(), 2);
        assert!(first.is_done(), "第一条跑满上限");
        assert!(second.is_done(), "第二条跑满上限");
        assert!(third.is_done(), "第三条跑满上限");

        // 上下文 = 提示段 + 各自生成段：共享前缀那条的提示仍是 [7, 7]，没被生成历史冲掉。
        assert_eq!(first.context(), [7, 7, 10, 11, 12]);
        assert_eq!(second.context(), [8, 8, 9, 9, 20, 21]);
        assert_eq!(third.context(), [7, 7, 30, 31]);
    }

    /// 批量结果视图：`outcomes` 按准入顺序把每条序列的历史与本步进度对齐，视图间互不重叠。
    #[test]
    fn batch_outcomes_pair_each_progress_with_its_own_sequence() {
        let mut batch = Batch::new(pool(4, 4));
        batch.admit(sequence(1, 0, 2, 4)).expect("准入第一条");
        batch.admit(sequence(2, 0, 1, 4)).expect("准入第二条");

        // 第一步走 `emit`（预留位 + 记账成对）；结局按准入顺序收进批量视图。
        let first = batch.emit(1, 5).expect("第一条第一步");
        let second = batch.emit(2, 6).expect("第二条第一步");
        let outcomes = [first, second];
        let view = batch.outcomes(&outcomes);
        assert_eq!(view.len(), 2);

        assert_eq!(view[0].id(), 1);
        assert_eq!(view[0].progress(), Progress::Running);
        assert_eq!(view[0].produced(), 1);
        assert_eq!(view[0].tokens(), [5], "第一条历史挂在自己序列上");
        assert_eq!(view[1].id(), 2);
        // 第二条 `max_new = 1`，这一步即跑满：结局是 `Done`，不是 `Running`。
        assert_eq!(view[1].progress(), Progress::Done);
        assert_eq!(view[1].tokens(), [6], "第二条历史挂在自己序列上");
        assert!(view[1].is_done());

        // 第二条已结束不再推进；第一条再走一步，视图仍按准入位置对齐到各自序列。
        let third = batch.emit(1, 7).expect("第一条第二步");
        let view = batch.outcomes(&[third, Progress::Done]);
        assert_eq!(view[0].tokens(), [5, 7], "第一条两步历史不串");
        assert_eq!(view[1].tokens(), [6], "第二条历史仍是第一步那一笔");

        // 按序号取单条视图，越界返回 `None`。
        assert_eq!(batch.outcome(0, third).expect("首条").id(), 1);
        assert!(batch.outcome(2, third).is_none(), "越界序号无视图");
        assert_eq!(batch.sequence_at(1).expect("按序号取序列").id(), 2);
        assert!(batch.sequence_at(2).is_none());
    }

    /// 准入满批退回：页不够时序列原样退回、不占页，重试前页账与退回前一致。
    #[test]
    fn admission_full_handles_back_without_touching_the_page_accounts() {
        // 池 2 页、页宽 2：第一条提示 2 占 1 页；第二条提示 2、`max_new = 4`，最坏情况要 3 页，
        // 既超过池总页数、空闲也只剩 1 页，准入补不出差额，整页退回，不留半个分配。
        let mut batch = Batch::new(pool(2, 2));
        batch
            .admit(sequence_with(1, &[7, 7], 1, 2))
            .expect("准入第一条");
        let rejected = match batch
            .admit(sequence_with(2, &[8, 8], 4, 2))
            .expect("准入判定")
        {
            Admission::Full(sequence) => sequence,
            Admission::Admitted => panic!("空闲页不够，第二条应原样退回"),
        };
        assert_eq!(rejected.id(), 2);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.scheduler().cache().in_use(), 1, "退回不占页");
        assert_eq!(batch.scheduler().cache().free_pages(), 1);

        // 第一条结束腾出页位后：第二条最坏情况仍要 3 页，池只有 2 页，`fits` 不通过，
        // 继续按"现在装不下"退回，而不是被部分准入；它的页账原样不动。
        assert!(batch.finish(1).expect("结束第一条"));
        assert_eq!(batch.scheduler().cache().in_use(), 0);
        match batch.admit(rejected).expect("腾页后重试仍判容量") {
            Admission::Full(sequence) => {
                assert_eq!(sequence.id(), 2, "装不下的序列原样退回");
                assert_eq!(batch.scheduler().cache().in_use(), 0, "重试失败不留分配");
            }
            Admission::Admitted => panic!("池只有 2 页，第二条最坏情况要 3 页，不该准入"),
        }
    }

    /// 结束序列的页随序列回收：独占新页全部回池，增长出来的页也一并归还，不残留占用。
    #[test]
    fn finishing_a_sequence_releases_all_of_its_pages() {
        let mut batch = Batch::new(pool(4, 2));
        batch.admit(sequence_with(1, &[3, 3], 3, 2)).expect("准入");
        // 提示长度 2 占 1 页；再生成三步：长度 4 跨进第二页，长度 6 跨进第三页，末步跑满上限。
        for (index, token) in (10..13).enumerate() {
            let expected = if index == 2 {
                Progress::Done
            } else {
                Progress::Running
            };
            assert_eq!(batch.emit(1, token).expect("生成"), expected);
        }
        assert_eq!(batch.scheduler().cache().in_use(), 3, "增长后占三页");

        assert!(batch.finish(1).expect("结束"));
        assert!(batch.is_empty());
        assert_eq!(batch.scheduler().cache().in_use(), 0, "全部页归还");
        assert_eq!(
            batch.scheduler().cache().free_pages(),
            batch.scheduler().cache().total(),
            "归还后空闲等于池总页"
        );
    }

    /// 预填充只登记"行已落定"的整块：推进到 3 个位置只登记第 0 块，满 4 个位置才有第 1 块。
    #[test]
    fn marking_a_prompt_prefilled_publishes_only_whole_blocks() {
        let mut batch = Batch::new(pool(8, 2));
        assert_eq!(
            batch
                .admit(sequence_with(1, &[1, 2, 3, 4, 5], 2, 2))
                .expect("准入"),
            Admission::Admitted
        );
        assert!(!batch.sequence(1).expect("在批里").prefilled_ready());
        batch.mark_prefilled(1, 3).expect("算完三个位置");
        assert_eq!(batch.scheduler().cached_blocks(), 1, "只有第 0 块写满了");
        assert!(!batch.sequence(1).expect("在批里").prefilled_ready());
        batch.mark_prefilled(1, 5).expect("算完提示段");
        assert!(batch.sequence(1).expect("在批里").prefilled_ready());
        assert_eq!(
            batch.scheduler().cached_blocks(),
            2,
            "第 2 块只有一行，不够一块，不许登记"
        );
    }

    /// 预填充位置只许往前：倒退会把已登记的块当成没算过，越过提示段则索引不存在的行。
    #[test]
    fn a_prefill_position_cannot_go_backwards_or_past_the_prompt() {
        let mut batch = Batch::new(pool(4, 2));
        batch
            .admit(sequence_with(1, &[1, 2, 3, 4], 2, 2))
            .expect("准入");
        batch.mark_prefilled(1, 3).expect("算完三个位置");
        assert_eq!(
            batch.mark_prefilled(1, 2).unwrap_err().code(),
            Code::Invalid,
            "倒退不许"
        );
        assert_eq!(
            batch.mark_prefilled(1, 5).unwrap_err().code(),
            Code::Invalid,
            "越过提示段不许"
        );
        assert_eq!(
            batch.mark_prefilled(9, 1).unwrap_err().code(),
            Code::Invalid,
            "不在批里的序列不许"
        );
    }

    /// 解码写满一块也登记：两条路径（预填充、解码）走的是同一份"落定即登记"。
    #[test]
    fn writing_the_last_row_of_a_block_publishes_it() {
        let mut batch = Batch::new(pool(8, 2));
        batch.admit(sequence_with(1, &[1, 2], 4, 2)).expect("准入");
        assert_eq!(batch.scheduler().cached_blocks(), 0);
        batch.step().expect("推进一步：页表随长度长到两页");
        batch.write_kv(1, 2, &[0.5; 2]).expect("写第 2 行");
        assert_eq!(
            batch.scheduler().cached_blocks(),
            1,
            "第 2 行落定即第 0 块写完"
        );
    }

    /// 预填充过的一次提示，下一个同前缀请求只补差额页：共享的页就是同一批页号。
    #[test]
    fn a_prefilled_prompt_is_shared_with_the_next_request() {
        let mut batch = Batch::new(pool(8, 2));
        let prompt = [1, 2, 3, 4];
        batch.admit(sequence_with(1, &prompt, 1, 2)).expect("准入");
        let first: Vec<Page> = batch
            .scheduler()
            .pages_of(1)
            .expect("页表")
            .pages()
            .to_vec();
        assert_eq!(first.len(), 2, "提示 4 个 token、页宽 2：正好两页");
        batch.mark_prefilled(1, prompt.len()).expect("算完提示段");
        assert_eq!(batch.scheduler().cached_blocks(), 2);
        batch.finish(1).expect("结束");
        assert_eq!(
            batch.scheduler().cache().in_use(),
            2,
            "块由前缀树持有，请求结束不还"
        );

        batch.admit(sequence_with(2, &prompt, 1, 2)).expect("准入");
        let second: Vec<Page> = batch
            .scheduler()
            .pages_of(2)
            .expect("页表")
            .pages()
            .to_vec();
        assert_eq!(&second[..2], &first[..2], "前两页整块接走，页号必须同源");
        assert_eq!(second.len(), 2, "整段提示都接走，一页也不用新开");
    }

    /// 召回按整块走：块登记进去后，页已被别的请求共享也不会被当作空闲页摘走。
    #[test]
    fn a_published_block_keeps_its_page_out_of_the_free_list() {
        let mut batch = Batch::new(pool(2, 2));
        batch.admit(sequence_with(1, &[1, 2], 1, 2)).expect("准入");
        batch.mark_prefilled(1, 2).expect("算完提示段");
        let page = batch.scheduler().pages_of(1).expect("页表").pages()[0];
        batch.finish(1).expect("结束");
        assert!(
            batch.scheduler().cache().is_live(page),
            "块还在树上，页不能回池"
        );
        assert_eq!(
            batch.scheduler().cache().free_pages(),
            1,
            "两页池只空闲一页"
        );
    }

    /// 这一轮的行由相位定序：提示段还没算完的在后面，能解码的在前。
    ///
    /// 相位不是调用方给的标签，而是序列自己的事实（`prefilled < prompt_len`），所以"谁先进批"
    /// 与"谁先跑"是两件事。
    #[test]
    fn a_round_runs_generation_before_context() {
        use crate::ifb::{Budget, Verdict};

        let mut batch = Batch::new(pool(8, 4));
        batch.admit(sequence(1, 8, 4, 4)).expect("准入");
        batch.admit(sequence(2, 0, 4, 4)).expect("准入");

        let Verdict::Packed { rows, tokens } = batch.round(Budget::unlimited()) else {
            panic!("不限预算必须装得下");
        };
        assert_eq!(
            rows.iter().map(|row| row.id()).collect::<Vec<_>>(),
            [2, 1],
            "能解码的在前，提示段在后"
        );
        assert_eq!(rows[1].rows(), 8, "8 个 token 的提示段一轮算完");
        assert_eq!(tokens, 9);
    }

    /// 挤压时只挤上下文：生成行留着，上下文整块推到下一轮。
    ///
    /// 块大小由 [`crate::ifb::chunk`] 按页对齐给出，所以"少跑一点"只能是整页少跑，不会出现
    /// 半页的块——那种块既不进前缀树，也换不来一次干净的对齐。
    #[test]
    fn a_tight_budget_pushes_the_context_block_to_the_next_round() {
        use crate::ifb::{Budget, Verdict};

        let mut batch = Batch::new(pool(8, 4));
        batch.admit(sequence(1, 8, 4, 4)).expect("准入");
        batch.admit(sequence(2, 0, 4, 4)).expect("准入");

        let Verdict::Packed { rows, tokens } = batch.round(Budget::new(2, 2)) else {
            panic!("生成行只要 1 行，装得下");
        };
        assert_eq!(
            rows.iter().map(|row| row.id()).collect::<Vec<_>>(),
            [2],
            "2 个 token 的预算连一整页上下文都放不下，只跑生成"
        );
        assert_eq!(tokens, 1);

        let Verdict::Packed { rows, tokens } = batch.round(Budget::new(2, 5)) else {
            panic!("1 生成 + 4 上下文正好 5 行");
        };
        assert_eq!(
            rows.iter().map(|row| (row.id(), row.rows())).collect::<Vec<_>>(),
            [(2, 1), (1, 4)],
            "预算够了就按整页推进一块"
        );
        assert_eq!(tokens, 5);
    }

    /// 两类"装不下"各报各的：序列数超了可等一条结束，生成行自己超预算重试无用。
    #[test]
    fn a_round_separates_the_two_over_budget_cases() {
        use crate::ifb::{Budget, Verdict};

        let mut batch = Batch::new(pool(8, 4));
        batch.admit(sequence(1, 0, 4, 4)).expect("准入");
        batch.admit(sequence(2, 0, 4, 4)).expect("准入");

        assert_eq!(
            batch.round(Budget::new(1, 4)),
            Verdict::OverSeqs {
                seqs: 2,
                max: 1
            }
        );
        assert_eq!(
            batch.round(Budget::new(2, 1)),
            Verdict::OverTokens {
                tokens: 2,
                max: 1
            }
        );
    }

    /// 空批不排任何行：`Idle` 与"被预算挤掉"是两件事，调用方据此决定要不要继续等。
    #[test]
    fn an_empty_batch_has_nothing_to_round() {
        use crate::ifb::{Budget, Verdict};

        let batch = Batch::new(pool(4, 4));
        assert_eq!(batch.round(Budget::unlimited()), Verdict::Idle);
    }
}
