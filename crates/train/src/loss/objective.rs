//! 训练目标（损失）选择：把三种损失族的正向度量统一到一个配置化入口。

use diagnostic::Result;

use crate::distill::{Distill, distillation_loss};
use crate::loss::evaluate;
use crate::rl::{Grpo, grpo_loss};

/// 一次损失计算的结果：平均 loss 与命中条数（RL 恒 0）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// 本步平均损失。
    pub loss: f32,
    /// 本步命中条数（分类/蒸馏为 argmax 命中，RL 为 0）。
    pub correct: u64,
}

/// 训练/评估损失选择：统一分类、蒸馏与 RL 三种损失族的正向度量。
pub enum Loss<'a> {
    /// 硬标签交叉熵。
    Classify,
    /// Hinton 蒸馏：软 KL + 硬 CE。
    Distill {
        /// 教师 logits（软标签来源），与 `logits` 同形状。
        teacher: &'a [f32],
        /// 蒸馏配置（温度 `T` 与软损失权重 `α`）。
        config: Distill,
    },
    /// GRPO：裁剪策略梯度 + 参考 KL 惩罚。
    Grpo {
        /// 当前策略逐 token log-prob（由 [`crate::rl::log_prob`] 预计算）。
        new_logp: &'a [f32],
        /// 行为策略逐 token log-prob（冻结）。
        old_logp: &'a [f32],
        /// 逐 token advantage（GRPO 中同 completion 共享）。
        advantages: &'a [f32],
        /// 逐 token 参考 KL（由 [`crate::distill::kl_per_row`] 预计算）。
        ref_kl: &'a [f32],
        /// GRPO 配置（裁剪 `ε` 与 KL 惩罚 `β`）。
        config: Grpo,
    },
}

impl<'a> Loss<'a> {
    /// 由 logits 与 labels 计算 [`Outcome`]；几何/参数校验透传。RL 的 `logits`/`labels` 不参与
    /// （`new_logp` 已预计算）。
    pub fn compute(&self, logits: &[f32], labels: &[f32], vocab: usize) -> Result<Outcome> {
        match self {
            Self::Classify => classify(logits, labels, vocab),
            Self::Distill { teacher, config } => distill(logits, teacher, labels, vocab, *config),
            Self::Grpo {
                new_logp,
                old_logp,
                advantages,
                ref_kl,
                config,
            } => grpo(new_logp, old_logp, advantages, ref_kl, *config),
        }
    }
}

/// 分类：CE loss + argmax 命中。
pub(crate) fn classify(logits: &[f32], labels: &[f32], vocab: usize) -> Result<Outcome> {
    let (nll_sum, correct, total) = evaluate(logits, labels, vocab)?;
    Ok(Outcome {
        loss: (nll_sum / total as f64) as f32,
        correct: correct as u64,
    })
}

/// 蒸馏：软 KL + 硬 CE loss + argmax 命中。
pub(crate) fn distill(
    logits: &[f32],
    teacher: &[f32],
    labels: &[f32],
    vocab: usize,
    config: Distill,
) -> Result<Outcome> {
    let loss = distillation_loss(logits, teacher, labels, vocab, config)?;
    let (_, correct, _) = evaluate(logits, labels, vocab)?;
    Ok(Outcome {
        loss,
        correct: correct as u64,
    })
}

/// RL：GRPO 组合 loss，命中恒 0。
pub(crate) fn grpo(
    new_logp: &[f32],
    old_logp: &[f32],
    advantages: &[f32],
    ref_kl: &[f32],
    config: Grpo,
) -> Result<Outcome> {
    let loss = grpo_loss(new_logp, old_logp, advantages, ref_kl, config)?;
    Ok(Outcome { loss, correct: 0 })
}

#[cfg(test)]
#[path = "../../tests/objective_tests.rs"]
mod tests;
