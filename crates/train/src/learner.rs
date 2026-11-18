//! 独立训练编排器：把数据集遍历、前向评估、指标入账与优化步收拢到一个驱动对象。
//!
//! [`Session`](crate::Session) 只持有会话状态与单步接缝，不关心数据集如何分批；本模块补上
//! 「遍历一批 → 前向 → 记 loss/accuracy → 装梯度 → 优化步」的循环编排。模型拓扑不在这里硬编码：
//! 前向装配与梯度来源由两个闭包注入——
//!
//! - `forward`：装载本批输入并算出 `logits` 与 `loss` 两个终点变量；
//! - `backward`：把本批梯度装入/累加进会话缓冲（反传数值属 P1 未接入范围，当前由调用方提供）。
//!
//! 标签按标准 next-token 设定由序列右移派生：位置 `(r, c)` 的标签是 `tokens[r, c+1]`；每行末列
//! 没有后继，剔除出评估集。logits 必须按同一行主序位置展开成 `[positions, vocab]`。

use diagnostic::{Code, Error, Result};

use autograd::Var;
use data::Dataset;

use crate::loss::{argmax_hits, cross_entropy};
use crate::metrics::{MetricPoint, MetricsBoard};
use crate::optimizer::Optimizer;
use crate::{Kernel, Session};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Learner";

/// 一次前向的两个评估终点：`logits` 用于算 NLL 与命中率，`loss` 用于反传计划对账。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardOut {
    /// 每个有效位置的词表分布，展平成 `[positions, vocab]`。
    pub logits: Var,
    /// 标量/向量 loss 终点，供会话规划梯度可达性。
    pub loss: Var,
}

/// 前向一步取出的评估数据：标签与同序位置的 logits；反传终点不在此处留存。
struct Prepared {
    labels: Vec<f32>,
    logits: Vec<f32>,
}

/// 训练编排器：包住一个会话与一块指标看板，按 epoch 遍历数据集推进。
pub struct Learner<K: Kernel, O: Optimizer> {
    session: Session<K, O>,
    board: MetricsBoard,
    vocab: usize,
}

impl<K: Kernel, O: Optimizer> Learner<K, O> {
    /// 由已装配会话与词表宽度构造；词表为 0 是 [`Code::Invalid`]。
    pub fn new(session: Session<K, O>, vocab: usize) -> Result<Self> {
        if vocab == 0 {
            return Err(Error::new(Code::Invalid, "词表宽度不能为 0").with(SITE));
        }
        Ok(Self {
            session,
            board: MetricsBoard::new(),
            vocab,
        })
    }

    /// 会话状态（只读）。
    pub fn session(&self) -> &Session<K, O> {
        &self.session
    }

    /// 会话状态（可变：登记参数、装权重、回载快照等装配动作走这里）。
    pub fn session_mut(&mut self) -> &mut Session<K, O> {
        &mut self.session
    }

    /// 指标看板（只读）。
    pub fn board(&self) -> &MetricsBoard {
        &self.board
    }

    /// 拆解回收会话：快照导出/回载等需要裸会话的场景取走会话本体。
    pub fn into_session(self) -> Session<K, O> {
        self.session
    }

    /// 跑指定 epoch 数；每个完整批走一次前向评估、入账与优化步。
    ///
    /// `forward` 借会话与本批 token 装配前向，返回两个终点；`backward` 在入账后把梯度装进缓冲。
    /// 数据集残批（不足一行）由 [`Dataset`] 自身丢弃，编排器不补齐。
    ///
    /// # Errors
    ///
    /// 批几何、前向、计划、梯度入账或优化步任一步失败都会透传；评估几何不自洽也是
    /// [`Code::Invalid`]。
    pub fn run<'a, F, G>(
        &mut self,
        dataset: &mut dyn Dataset<'a>,
        epochs: u64,
        mut forward: F,
        mut backward: G,
    ) -> Result<()>
    where
        F: FnMut(&mut Session<K, O>, &[u32]) -> Result<ForwardOut>,
        G: FnMut(&mut Session<K, O>) -> Result<()>,
    {
        for _epoch in 0..epochs {
            while let Some(batch) = dataset.next_batch() {
                self.train_batch(
                    batch.tokens(),
                    batch.rows(),
                    batch.cols(),
                    &mut forward,
                    &mut backward,
                )?;
            }
        }
        Ok(())
    }

    /// 单个 micro-batch 的编排闭环：接批 → 前向 → 评估入账 → 装梯度 → 优化步。
    fn train_batch<F, G>(
        &mut self,
        tokens: &[u32],
        rows: usize,
        cols: usize,
        forward: &mut F,
        backward: &mut G,
    ) -> Result<()>
    where
        F: FnMut(&mut Session<K, O>, &[u32]) -> Result<ForwardOut>,
        G: FnMut(&mut Session<K, O>) -> Result<()>,
    {
        let prepared = self.forward_pass(rows, cols, tokens, forward)?;
        self.finish(prepared, backward)?;
        Ok(())
    }

    /// 入账评估量、装入本批梯度并走优化步。
    fn finish<G>(&mut self, prepared: Prepared, backward: &mut G) -> Result<()>
    where
        G: FnMut(&mut Session<K, O>) -> Result<()>,
    {
        self.record(&prepared.logits, &prepared.labels)?;
        backward(&mut self.session)?;
        self.session.optimize()?;
        Ok(())
    }

    /// 接批、跑前向并取出评估所需的标签与 logits；反传终点已由前向闭包登记进计划。
    fn forward_pass<F>(
        &mut self,
        rows: usize,
        cols: usize,
        tokens: &[u32],
        forward: &mut F,
    ) -> Result<Prepared>
    where
        F: FnMut(&mut Session<K, O>, &[u32]) -> Result<ForwardOut>,
    {
        self.session.accept_batch(rows, cols, tokens.len())?;
        let out = forward(&mut self.session, tokens)?;
        let labels = next_token_labels(tokens, rows, cols);
        let logits = self.read_logits(out.logits)?;
        Ok(Prepared { labels, logits })
    }

    /// 读出 logits 终点的 f32 切片；不在场是 [`Code::Missing`]，lane 不符是 [`Code::Invalid`]。
    fn read_logits(&mut self, logits: Var) -> Result<Vec<f32>> {
        let view = self.session.buffers().value(logits).ok_or_else(|| {
            Error::new(Code::Missing, format!("logits 变量 {} 不在场", logits.id())).with(SITE)
        })?;
        view.f32()
            .map(<[f32]>::to_vec)
            .ok_or_else(|| Error::new(Code::Invalid, "logits 不是 f32 lane").with(SITE))
    }

    /// 算 NLL 与命中率并记进看板；logits 长度必须等于有效位置数 × 词表宽度。
    fn record(&mut self, logits: &[f32], labels: &[f32]) -> Result<()> {
        let positions = labels.len();
        let expected = positions * self.vocab;
        if logits.len() != expected {
            let message = format!(
                "logits 元素数 {} 与 {positions} 个有效位置 × {} 词表 = {expected} 不符",
                logits.len(),
                self.vocab
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        let loss = cross_entropy(logits, labels, self.vocab)?;
        let (correct, total) = argmax_hits(logits, labels, self.vocab)?;
        let accuracy = correct as f32 / total as f32;
        let point = MetricPoint {
            step: self.session.step(),
            samples: total,
            loss,
            accuracy,
        };
        self.board.record(point).map_err(|error| error.with(SITE))
    }
}

/// next-token 标签：`(r, c)` 的标签取 `tokens[r, c+1]`；行末列无后继，剔除。
///
/// 位置顺序与前向闭包产出的 logits 行主序一一对应；一列宽（`cols == 1`）没有可预测位置，
/// 返回空表，由下游损失函数拒绝。
fn next_token_labels(tokens: &[u32], rows: usize, cols: usize) -> Vec<f32> {
    let positions = rows.saturating_mul(cols.saturating_sub(1));
    let mut labels = Vec::with_capacity(positions);
    // 行边界切断右移：外层按行推进；行内后继是一段连续 slice，整块映射进标签表。
    for row in 0..rows {
        let base = row * cols;
        labels.extend(
            tokens[base + 1..base + cols]
                .iter()
                .map(|&token| token as f32),
        );
    }
    labels
}

#[cfg(test)]
#[path = "../tests/learner_tests.rs"]
mod tests;
