use crate::{Grpo, advantage, clipped_surrogate, entropy, grpo_loss, log_prob, policy_loss};
use diagnostic::{Code, Result};

/// 对齐类别顶到最高时，log-prob 应趋近 0。
#[test]
fn log_prob_is_near_zero_for_the_peaked_action() -> Result<()> {
    let logits = [0.0, 10.0, 0.0];
    let actions = [1.0];
    let mut out = [0.0];
    log_prob(&logits, &actions, 3, &mut out)?;
    assert!(
        out[0].abs() < 1e-3,
        "对齐类别 log-prob 应趋近 0，得到 {}",
        out[0]
    );
    Ok(())
}

/// 均匀分布下 log-prob 应等于 −ln(vocab)。
#[test]
fn log_prob_of_uniform_distribution_is_negative_ln_vocab() -> Result<()> {
    let logits = [0.0, 0.0];
    let actions = [0.0];
    let mut out = [0.0];
    log_prob(&logits, &actions, 2, &mut out)?;
    assert!(
        (out[0] + 2.0f32.ln()).abs() < 1e-6,
        "均匀 log-prob 应趋近 −ln2，得到 {}",
        out[0]
    );
    Ok(())
}

/// 越界动作被拒绝。
#[test]
fn log_prob_rejects_out_of_range_action() {
    assert!(matches!(
        log_prob(&[0.0, 0.0], &[3.0], 2, &mut [0.0]),
        Err(e) if e.code() == Code::Invalid
    ));
}

/// 均匀分布熵 = ln(vocab)，是所有分布里的最大值。
#[test]
fn entropy_of_uniform_distribution_is_ln_vocab() -> Result<()> {
    let h = entropy(&[0.0, 0.0, 0.0], 3)?;
    assert!((h - 3.0f32.ln()).abs() < 1e-6, "均匀熵应趋近 ln3，得到 {h}");
    Ok(())
}

/// 独热分布熵趋近 0。
#[test]
fn entropy_of_a_peaked_distribution_is_zero() -> Result<()> {
    let h = entropy(&[0.0, 100.0, 0.0], 3)?;
    assert!(h.abs() < 1e-3, "独热熵应趋近 0，得到 {h}");
    Ok(())
}

/// advantage 是零均值、单位标准差的 z-score。
#[test]
fn advantage_is_a_zero_mean_unit_std_zscore() -> Result<()> {
    let rewards = [1.0, 2.0, 3.0];
    let mut out = [0.0; 3];
    advantage(&rewards, &mut out)?;
    let mean = out.iter().sum::<f32>() / 3.0;
    let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
    assert!(mean.abs() < 1e-5, "z-score 均值应趋近 0，得到 {mean}");
    assert!(
        (var.sqrt() - 1.0).abs() < 1e-5,
        "z-score 标准差应趋近 1，得到 {}",
        var.sqrt()
    );
    Ok(())
}

/// 全体 reward 相等时 advantage 恒 0。
#[test]
fn advantage_of_equal_rewards_is_zero() -> Result<()> {
    let rewards = [5.0, 5.0, 5.0];
    let mut out = [0.0; 3];
    advantage(&rewards, &mut out)?;
    assert!(
        out.iter().all(|v| v.abs() < 1e-6),
        "相等 reward 的 advantage 应为 0，得到 {out:?}"
    );
    Ok(())
}

/// 策略损失是 −mean(advantage · log_prob)。
#[test]
fn policy_loss_negates_the_mean_of_advantage_times_log_prob() -> Result<()> {
    let log_probs = [-0.5, -1.0];
    let advantages = [2.0, 1.0];
    let loss = policy_loss(&log_probs, &advantages)?;
    // 2·(−0.5) + 1·(−1.0) = −2，均值 −1，取负 = 1。
    assert!((loss - 1.0).abs() < 1e-6, "策略损失应等于 1，得到 {loss}");
    Ok(())
}

/// 优势与 log-prob 不等长被拒绝。
#[test]
fn policy_loss_rejects_length_mismatch() {
    assert!(matches!(
        policy_loss(&[0.0], &[0.0, 0.0]),
        Err(e) if e.code() == Code::Invalid
    ));
}

/// ratio=1（新旧 log-prob 相等）时裁剪代理退化为 advantage 本身。
#[test]
fn clipped_surrogate_equals_advantage_when_ratio_is_one() -> Result<()> {
    let mut out = [0.0; 2];
    clipped_surrogate(&[0.0, 0.0], &[0.0, 0.0], &[2.0, 1.0], 0.2, &mut out)?;
    assert!((out[0] - 2.0).abs() < 1e-6 && (out[1] - 1.0).abs() < 1e-6, "ratio=1 时应为 A，得到 {out:?}");
    Ok(())
}

/// ratio 过高被裁剪到 1+ε：代理 = (1+ε)·A。
#[test]
fn clipped_surrogate_caps_high_ratio() -> Result<()> {
    let mut out = [0.0];
    clipped_surrogate(&[10.0], &[0.0], &[1.0], 0.2, &mut out)?;
    assert!((out[0] - 1.2).abs() < 1e-6, "高 ratio 应裁剪到 1.2·A，得到 {}", out[0]);
    Ok(())
}

/// β=0 时 GRPO 损失退化为 −mean(裁剪代理)。
#[test]
fn grpo_loss_with_zero_beta_is_negative_surrogate_mean() -> Result<()> {
    let loss = grpo_loss(
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[2.0, 1.0],
        &[0.5, 0.5],
        Grpo::new(0.2, 0.0),
    )?;
    // ratio=1，代理 = [2,1]，均值 1.5，取负 = −1.5。
    assert!((loss + 1.5).abs() < 1e-6, "β=0 损失应为 −1.5，得到 {loss}");
    Ok(())
}

/// β>0 时 GRPO 损失额外加上 β·mean(KL_ref)。
#[test]
fn grpo_loss_adds_beta_times_mean_kl() -> Result<()> {
    let loss = grpo_loss(
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[2.0, 1.0],
        &[0.5, 0.5],
        Grpo::new(0.2, 1.0),
    )?;
    // −1.5 + 1.0·0.5 = −1.0。
    assert!((loss + 1.0).abs() < 1e-6, "β=1 损失应为 −1.0，得到 {loss}");
    Ok(())
}

/// 负 ε 被拒绝。
#[test]
fn grpo_loss_rejects_negative_epsilon() {
    assert!(matches!(
        grpo_loss(&[0.0], &[0.0], &[0.0], &[0.0], Grpo::new(-0.1, 0.0)),
        Err(e) if e.code() == Code::Invalid
    ));
}

/// 负 β 被拒绝。
#[test]
fn grpo_loss_rejects_negative_beta() {
    assert!(matches!(
        grpo_loss(&[0.0], &[0.0], &[0.0], &[0.0], Grpo::new(0.2, -0.1)),
        Err(e) if e.code() == Code::Invalid
    ));
}
