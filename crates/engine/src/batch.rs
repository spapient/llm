//! 连续批：[`Batch`] 把页账（[`Scheduler`]）与它正在跑的 [`Sequence`] 收在一处。
//!
//! 两者分开的代价是"批里有哪些序列"有两个来源：调用方自己维护的列表，与调度器里按 id 记的
//! `running`。两者一漂，[`Scheduler::advance`] 就会对一个不在批里的序列报错，或者序列已经结束
//! 而它的页还占着。[`Batch`] 让这件事只有一个来源：准入时把序列交出去，推进按 id 找，
//! 结束时页与序列一起释放。
//!
//! 采样不在这里：每一步该采哪个 token 由调用方决定（`advance` 只回答"这一步的页有没有着落"），
//! 所以本层不需要认识词表，也不持有 logits。
//!
//! `impl Batch` 按主题分成三个文件：[`admit`]（准入与轮次）、[`step`]（推进与记账）、
//! [`account`]（KV 记账与只读视图）——三组方法之间没有共享的中间量，只是同一块状态的不同侧面。

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

// ---------------------------------------------------------------------------
// 登记块：只读收集上下文与页号，再交出页池
// ---------------------------------------------------------------------------

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
