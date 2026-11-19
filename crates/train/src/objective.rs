//! 训练目标（损失）选择：把三种损失族的正向度量统一到一个配置化入口。
//!
//! [`crate::loss`]（分类）、[`crate::distill`]（蒸馏）、[`crate::rl`]（RL）各自提供纯函数；本模块
//! 用 [`Loss`] 枚举按配置分发，`compute` 返回统一的 [`Outcome`]（loss + 命中条数），供编排器与
//! 看板用同一种口径取评估量。三种损失族仍可直接调用纯函数，[`Loss`] 只是把「选哪个」也做成数据。
//!
//! RL 的逐 token log-prob 由调用方经 [`crate::rl::log_prob`] 预计算后传入，`compute` 不做分配；
//! RL 无分类命中，`correct` 恒为 0（蒸馏仍有硬标签命中，取 argmax(student) 对硬标签）。

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
///
/// `logits` 恒为学生/策略 logits，`labels` 恒为硬标签（分类/蒸馏）或采样动作（RL）。蒸馏额外
/// 携带教师 logits，RL 额外携带预计算的当前策略 log-prob、行为策略 log-prob、advantage 与
/// 参考 KL。
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
mod tests {
use crate::{Distill, Grpo, Loss, Outcome, cross_entropy, distillation_loss, grpo_loss};
use diagnostic::{Code, Result};

/// 分类损失选择与 cross_entropy + argmax 命中一致。
#[test]
fn classify_matches_cross_entropy_and_hits() -> Result<()> {
    let logits = [0.0, 10.0, 0.0, 10.0, 0.0, 0.0];
    let labels = [1.0, 0.0];
    let out = Loss::Classify.compute(&logits, &labels, 3)?;
    assert!(
        (out.loss - cross_entropy(&logits, &labels, 3)?).abs() < 1e-6,
        "loss 应与 CE 一致"
    );
    assert_eq!(out.correct, 2, "两行都命中");
    Ok(())
}

/// 蒸馏损失选择与 distillation_loss + argmax 命中一致。
#[test]
fn distill_matches_distillation_loss_and_hits() -> Result<()> {
    let student = [0.0, 10.0, 0.0];
    let teacher = [0.0, 0.0, 0.0];
    let labels = [1.0];
    let config = Distill::new(2.0, 0.5);
    let out = Loss::Distill {
        teacher: &teacher,
        config,
    }
    .compute(&student, &labels, 3)?;
    assert!(
        (out.loss - distillation_loss(&student, &teacher, &labels, 3, config)?).abs() < 1e-5,
        "loss 应与蒸馏损失一致"
    );
    assert_eq!(out.correct, 1, "argmax(student)=1 命中标签 1");
    Ok(())
}

/// RL 损失选择与 grpo_loss 一致，命中恒 0。
#[test]
fn grpo_matches_grpo_loss_and_zero_hits() -> Result<()> {
    let new_logp = [-0.5, -1.0];
    let old_logp = [-0.5, -1.0];
    let advantages = [2.0, 1.0];
    let ref_kl = [0.5, 0.5];
    let config = Grpo::new(0.2, 1.0);
    let out = Loss::Grpo {
        new_logp: &new_logp,
        old_logp: &old_logp,
        advantages: &advantages,
        ref_kl: &ref_kl,
        config,
    }
    .compute(&[0.0], &[0.0], 1)?;
    let expected = grpo_loss(&new_logp, &old_logp, &advantages, &ref_kl, config)?;
    assert!(
        (out.loss - expected).abs() < 1e-6,
        "loss 应与 GRPO 损失一致，得到 {} vs {expected}",
        out.loss
    );
    assert_eq!(out.correct, 0, "RL 无分类命中");
    Ok(())
}

/// RL 损失的 logits/labels 不参与计算，空输入也可用（new_logp 已预计算）。
#[test]
fn grpo_ignores_logits_and_labels() -> Result<()> {
    let new_logp = [0.0];
    let old_logp = [0.0];
    let advantages = [1.0];
    let ref_kl = [0.0];
    let out = Loss::Grpo {
        new_logp: &new_logp,
        old_logp: &old_logp,
        advantages: &advantages,
        ref_kl: &ref_kl,
        config: Grpo::new(0.2, 0.0),
    }
    .compute(&[], &[], 1)?;
    assert!(out.loss.is_finite(), "空 logits/labels 不应影响 RL 损失");
    Ok(())
}

/// Outcome 是可比较的值类型。
#[test]
fn outcome_is_a_comparable_value() {
    let a = Outcome {
        loss: 1.0,
        correct: 2,
    };
    let b = Outcome {
        loss: 1.0,
        correct: 2,
    };
    assert_eq!(a, b);
}

/// 非法几何透传：空 labels 的分类损失返回 Invalid。
#[test]
fn classify_rejects_empty_labels() {
    assert!(matches!(
        Loss::Classify.compute(&[0.0], &[], 2),
        Err(e) if e.code() == Code::Invalid
    ));
}
}

