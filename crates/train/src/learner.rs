//! 独立训练编排器：把数据集遍历、前向评估、指标入账与优化步收拢到一个驱动对象。
//!
//! [`Session`](crate::Session) 只持有会话状态与单步接缝，不关心数据集如何分批；本模块补上
//! 「遍历一批 → 前向 → 记 loss/accuracy → 回传梯度 → 优化步」的循环编排。模型拓扑不在这里硬编码：
//! 前向装配由闭包注入，梯度先由 [`Session::backward`] 按反向计划自动回算——
//!
//! - `forward`：装载本批输入并算出 `logits` 与 `loss` 两个终点变量；
//! - `backward`：自动回算之后的覆盖/追加口，把额外梯度装入或累加进会话缓冲。
//!
//! 标签按标准 next-token 设定由序列右移派生：位置 `(r, c)` 的标签是 `tokens[r, c+1]`；每行末列
//! 没有后继，剔除出评估集。logits 必须按同一行主序位置展开成 `[positions, vocab]`。

use diagnostic::{Code, Error, Result};

use autograd::Var;
use shard::Collective;

use crate::Dataset;
use crate::distill::Distill;
use crate::metrics::{MetricPoint, MetricsBoard};
use crate::objective::{Outcome, classify, distill, grpo};
use crate::optimizer::Optimizer;
use crate::rl::Grpo;
use crate::{Kernel, Session};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Learner";

/// 一次前向的两个评估终点：`logits` 用于算 NLL 与命中率，`loss` 用于反传计划对账。
///
/// 默认度量是分类 NLL；`metric` 为 `Some` 时按蒸馏/RL 度量入账（见 [`Metric`]）。
#[derive(Debug, Clone, Copy)]
pub struct ForwardOut {
    /// 每个有效位置的词表分布，展平成 `[positions, vocab]`。
    pub logits: Var,
    /// 标量/向量 loss 终点，供会话规划梯度可达性。
    pub loss: Var,
    /// 可选的非分类度量来源：`None` 走分类 NLL + 命中率。
    pub metric: Option<Metric>,
}

/// 非分类度量来源：蒸馏/RL 度量所需的额外变量与配置。
///
/// 蒸馏只多一个教师 logits 变量；RL 额外带当前/行为策略 log-prob、advantage 与参考 KL 四个变量
/// （逐 token 对齐），配置随度量一起由前向闭包给出。
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    /// Hinton 蒸馏：软 KL + 硬 CE。
    Distill {
        /// 教师 logits（软标签来源），与 `logits` 同形状。
        teacher: Var,
        /// 蒸馏配置（温度 `T` 与软损失权重 `α`）。
        config: Distill,
    },
    /// GRPO：裁剪策略梯度 + 参考 KL 惩罚。
    Grpo {
        /// 当前策略逐 token log-prob。
        new_logp: Var,
        /// 行为策略逐 token log-prob（冻结）。
        old_logp: Var,
        /// 逐 token advantage（同 completion 共享）。
        advantages: Var,
        /// 逐 token 参考 KL。
        ref_kl: Var,
        /// GRPO 配置（裁剪 `ε` 与 KL 惩罚 `β`）。
        config: Grpo,
    },
}

/// 前向一步取出的评估数据：标签与同序位置的 logits；反传终点不在此处留存。
struct Prepared {
    labels: Vec<f32>,
    logits: Vec<f32>,
    metric: Option<Metric>,
}

/// 训练编排器：包住一个会话与一块指标看板，按 epoch 遍历数据集推进。
///
/// `C` 是通信后端，默认 [`Remote`](crate::Remote) 走单进程恒等回路；接分布式后端时显式传入，
/// 编排器本体对后端透明——梯度同步发生在 [`Session::optimize`] 内部，本层不感知。
pub struct Learner<K: Kernel, O: Optimizer, C: Collective = shard::Remote> {
    session: Session<K, O, C>,
    board: MetricsBoard,
    vocab: usize,
}

impl<K: Kernel, O: Optimizer, C: Collective> Learner<K, O, C> {
    /// 由已装配会话与词表宽度构造；词表为 0 是 [`Code::Invalid`]。
    pub fn new(session: Session<K, O, C>, vocab: usize) -> Result<Self> {
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
    pub fn session(&self) -> &Session<K, O, C> {
        &self.session
    }

    /// 会话状态（可变：登记参数、装权重、回载快照等装配动作走这里）。
    pub fn session_mut(&mut self) -> &mut Session<K, O, C> {
        &mut self.session
    }

    /// 指标看板（只读）。
    pub fn board(&self) -> &MetricsBoard {
        &self.board
    }

    /// 拆解回收会话：快照导出/回载等需要裸会话的场景取走会话本体。
    pub fn into_session(self) -> Session<K, O, C> {
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
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        for _epoch in 0..epochs {
            while let Some(batch) = dataset.next_batch() {
                self.train_batch(
                    batch.rows(),
                    batch.cols(),
                    batch.tokens(),
                    &mut forward,
                    &mut backward,
                )?;
            }
        }
        Ok(())
    }

    /// 在独立验证集上跑一轮：只做前向与评估入账，不回传梯度、不步进优化器、不并入训练序列。
    ///
    /// 评估指标进看板的独立账表（[`MetricsBoard::eval_points`]），训练样本数/加权均值/最优
    /// loss 都不受验证批影响；跑完清空梯度缓冲，避免残留验证梯度混进下一个训练窗口。
    ///
    /// # Errors
    ///
    /// 批几何、前向或评估入账任一步失败都会透传；评估几何不自洽是 [`Code::Invalid`]。
    pub fn evaluate<'a, F>(&mut self, dataset: &mut dyn Dataset<'a>, mut forward: F) -> Result<()>
    where
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
    {
        while let Some(batch) = dataset.next_batch() {
            let prepared =
                self.forward_pass(batch.rows(), batch.cols(), batch.tokens(), &mut forward)?;
            self.record_eval(&prepared)?;
        }
        // 验证批不回传：清空可能残留的激活梯度，下一个训练窗口从空缓冲重新累加。
        self.session.zero_gradients();
        Ok(())
    }

    /// 单个 micro-batch 的编排闭环：前向装配 → 评估入账 → 装梯度 → 优化步。
    fn train_batch<F, G>(
        &mut self,
        rows: usize,
        cols: usize,
        tokens: &[u32],
        forward: &mut F,
        backward: &mut G,
    ) -> Result<()>
    where
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        let prepared = self.forward_pass(rows, cols, tokens, forward)?;
        self.finish(prepared, backward)?;
        Ok(())
    }

    /// 入账评估量、回传梯度、跑闭包覆盖口并走优化步。
    ///
    /// 回传段见 [`Self::backward_with`]；`backward` 闭包在自动回算之后保留为覆盖/追加口。
    fn finish<G>(&mut self, prepared: Prepared, backward: &mut G) -> Result<()>
    where
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        self.record(&prepared)?;
        self.backward_with(backward)?;
        self.session.optimize()?;
        Ok(())
    }

    /// 自动回算参数梯度，再跑调用方闭包覆盖/追加口。
    fn backward_with<G>(&mut self, backward: &mut G) -> Result<()>
    where
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        self.session.backward()?;
        backward(&mut self.session)?;
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
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
    {
        self.session.accept_batch(rows, cols, tokens.len())?;
        let out = forward(&mut self.session, tokens)?;
        let labels = next_token_labels(tokens, rows, cols);
        let logits = self.read_var(out.logits)?;
        Ok(Prepared {
            labels,
            logits,
            metric: out.metric,
        })
    }

    /// 读出变量终点的 f32 切片；不在场是 [`Code::Missing`]，lane 不符是 [`Code::Invalid`]。
    fn read_var(&mut self, var: Var) -> Result<Vec<f32>> {
        let view = self.session.buffers().value(var).ok_or_else(|| {
            Error::new(Code::Missing, format!("变量 {} 不在场", var.id())).with(SITE)
        })?;
        view.f32()
            .map(<[f32]>::to_vec)
            .ok_or_else(|| Error::new(Code::Invalid, "变量不是 f32 lane").with(SITE))
    }

    /// 按度量来源计算本步 (loss, 命中)；`None` 走分类，`Some` 走蒸馏/RL。
    fn compute_metric(&mut self, prepared: &Prepared) -> Result<Outcome> {
        match &prepared.metric {
            None => classify(&prepared.logits, &prepared.labels, self.vocab),
            Some(Metric::Distill { teacher, config }) => {
                let teacher = self.read_var(*teacher)?;
                distill(
                    &prepared.logits,
                    &teacher,
                    &prepared.labels,
                    self.vocab,
                    *config,
                )
            }
            Some(Metric::Grpo {
                new_logp,
                old_logp,
                advantages,
                ref_kl,
                config,
            }) => self.compute_grpo(*new_logp, *old_logp, *advantages, *ref_kl, *config),
        }
    }

    /// RL 度量：读出四个逐 token 变量后算 GRPO 组合损失。
    fn compute_grpo(
        &mut self,
        new_logp: Var,
        old_logp: Var,
        advantages: Var,
        ref_kl: Var,
        config: Grpo,
    ) -> Result<Outcome> {
        let new_logp = self.read_var(new_logp)?;
        let old_logp = self.read_var(old_logp)?;
        let advantages = self.read_var(advantages)?;
        let ref_kl = self.read_var(ref_kl)?;
        grpo(&new_logp, &old_logp, &advantages, &ref_kl, config)
    }

    /// 按度量计算 loss 与命中并记进训练看板。
    fn record(&mut self, prepared: &Prepared) -> Result<()> {
        let outcome = self.compute_metric(prepared)?;
        let point = MetricPoint {
            step: self.session.step(),
            samples: prepared.labels.len(),
            loss: outcome.loss,
            correct: outcome.correct,
        };
        self.board.record(point).map_err(|error| error.with(SITE))
    }

    /// 按度量计算 loss 与命中并记进评估独立账表；步号沿用当前训练步，仅作对账标记。
    fn record_eval(&mut self, prepared: &Prepared) -> Result<()> {
        let outcome = self.compute_metric(prepared)?;
        let point = MetricPoint {
            step: self.session.step(),
            samples: prepared.labels.len(),
            loss: outcome.loss,
            correct: outcome.correct,
        };
        self.board
            .record_eval(point)
            .map_err(|error| error.with(SITE))
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
