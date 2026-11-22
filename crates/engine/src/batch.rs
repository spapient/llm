//! 连续批：[`Batch`] 把页账（[`Scheduler`]）与它正在跑的 [`Sequence`] 收在一处。

mod step;

#[cfg(test)]
#[path = "../tests/batch.rs"]
mod tests;

use crate::kv::Pager;
use crate::{Progress, Scheduler, Sequence};
use diagnostic::{Code, Error, Result};

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

// ---------------------------------------------------------------------------
// 登记块：只读收集上下文与页号，再交出页池
// ---------------------------------------------------------------------------

/// 登记块要先只读地把上下文与页号收齐：`publish` 要独占页池，而这两样还得从只读的序列与页表里
/// 取，两个借用不能同时活着。整段上下文只克隆一次，逐块登记时复用。
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
    Some((sequence.ctx().to_vec(), table.pages().to_vec()))
}

/// 前 `rows` 行覆盖到的满块：块尾已落定（`(block + 1) · width <= 上下文长`）且页表里真有这一页。
fn settled_blocks(rows: usize, width: usize, ctx_len: usize, pages: usize) -> Vec<usize> {
    if width == 0 {
        return Vec::new();
    }
    (0..rows / width)
        .filter(|block| (block + 1) * width <= ctx_len && *block < pages)
        .collect()
}
