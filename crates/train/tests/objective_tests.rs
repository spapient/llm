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
