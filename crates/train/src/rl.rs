//! RL 损失族：策略梯度的前向纯计算原语（GRPO / PPO 共享地基）。
//!
//! 与 [`crate::loss`]（分类 NLL）、[`crate::distill`]（蒸馏软标签）互补：RL 采样出动作后，需要
//! token 的 log-prob、组内 advantage 归一化、熵正则与 advantage 加权策略损失。策略到参考模型
//! 的 KL 直接复用 [`crate::distill::kl_div`]（temperature=1），本模块不重复。与 loss / distill
//! 一致：只算前向纯计算，不进 op 设备核注册表，反向公式与 ratio 裁剪属 P1 / 后续接入。
//!
//! 数值稳定沿用 loss 的口径：log-prob 与熵都先减行内最大值再取指数，避免 `exp` 上溢。

use diagnostic::{Code, Error, Result};

use crate::loss::{check_geometry, check_len, check_logits_len, check_rows, check_vocab};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::rl";

/// 校验 logits 几何（词表/行数/长度），返回行数。
fn check_logits_shape(logits_len: usize, vocab: usize) -> Result<usize> {
    check_vocab(vocab)?;
    let rows = logits_len / vocab;
    check_rows(rows)?;
    check_logits_len(logits_len, rows, vocab)?;
    Ok(rows)
}

/// 单行 log-prob：`logits[action] − logsumexp(logits)`。
fn row_log_prob(slice: &[f32], action: usize) -> f32 {
    let max = slice.iter().copied().fold(f32::MIN, f32::max);
    let logsumexp = max + slice.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    slice[action] - logsumexp
}

/// 单行熵：`−Σ softmax·log softmax = logsumexp − Σ softmax·logits`。
fn row_entropy(row: &[f32]) -> f32 {
    let max = row.iter().copied().fold(f32::MIN, f32::max);
    let sum_exp: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let logsumexp = max + sum_exp.ln();
    let mut h = 0.0f32;
    for &v in row {
        let p = (v - max).exp() / sum_exp;
        h -= p * (v - logsumexp);
    }
    h
}

/// 采样动作的 log-prob：`logits[action] − logsumexp(logits)`，逐样本写回 `out`。
///
/// `logits` 按行展平成 `[rows, vocab]`，`actions` 长 `rows`（f32 承载类别索引，与 `embed_f32`
/// 同一索引惯例），`out` 长 `rows`。
///
/// # Errors
///
/// 几何不自洽、动作越界/非整数、或 `out` 与 `actions` 不等长时返回 [`Code::Invalid`]。
pub fn log_prob(logits: &[f32], actions: &[f32], vocab: usize, out: &mut [f32]) -> Result<()> {
    let rows = check_geometry(logits, actions, vocab)?;
    check_len(out.len(), rows, "out")?;
    for (row, o) in out.iter_mut().enumerate() {
        let slice = &logits[row * vocab..(row + 1) * vocab];
        *o = row_log_prob(slice, actions[row] as usize);
    }
    Ok(())
}

/// 熵均值：`−Σ softmax·log softmax`，逐行计算后取算术平均（探索正则项）。
///
/// # Errors
///
/// 词表为 0、无行、或 logits 长度不是 `vocab` 整数倍时返回 [`Code::Invalid`]。
pub fn entropy(logits: &[f32], vocab: usize) -> Result<f32> {
    let rows = check_logits_shape(logits.len(), vocab)?;
    let mut sum = 0.0f64;
    for row in logits.chunks_exact(vocab) {
        sum += row_entropy(row) as f64;
    }
    Ok((sum / rows as f64) as f32)
}

/// 组内 advantage：`(reward − mean) / std`（z-score），写回 `out`。
///
/// 全体 reward 相等时 std 为 0，advantage 恒 0（无相对优势）。
///
/// # Errors
///
/// reward 为空、或 `out` 与 `reward` 不等长时返回 [`Code::Invalid`]。
pub fn advantage(rewards: &[f32], out: &mut [f32]) -> Result<()> {
    check_rows(rewards.len())?;
    check_len(out.len(), rewards.len(), "out")?;
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32;
    let std = variance.sqrt();
    let inv_std = if std > 0.0 { 1.0 / std } else { 0.0 };
    for (o, &r) in out.iter_mut().zip(rewards) {
        *o = (r - mean) * inv_std;
    }
    Ok(())
}

/// REINFORCE 策略损失：`−mean(advantage · log_prob)`。
///
/// 最大化期望回报等价于最小化该损失；PPO / GRPO 的 ratio 裁剪在其上叠加（后续接入）。
///
/// # Errors
///
/// 任一为空、或两者不等长时返回 [`Code::Invalid`]。
pub fn policy_loss(log_probs: &[f32], advantages: &[f32]) -> Result<f32> {
    check_rows(log_probs.len())?;
    check_len(advantages.len(), log_probs.len(), "advantage")?;
    let sum: f64 = log_probs
        .iter()
        .zip(advantages)
        .map(|(&p, &a)| (a * p) as f64)
        .sum();
    Ok(-(sum / log_probs.len() as f64) as f32)
}

/// GRPO / PPO 训练损失配置：裁剪范围 `ε` 与参考 KL 惩罚系数 `β`。
///
/// 损失 = `−mean(min(ratio·A, clamp(ratio, 1−ε, 1+ε)·A)) + β·mean(KL_ref)`，其中
/// `ratio = exp(new_logp − old_logp)`，`A` 是 advantage，`KL_ref` 是逐 token 的参考 KL。
#[derive(Clone, Copy, Debug)]
pub struct Grpo {
    epsilon: f32,
    beta: f32,
}

impl Grpo {
    /// 构造 GRPO 配置；`epsilon`/`beta` 的合法性在损失入口校验（保持 `const`）。
    pub const fn new(epsilon: f32, beta: f32) -> Self {
        Self { epsilon, beta }
    }

    /// 裁剪范围 `ε`（非负）。
    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }

    /// 参考 KL 惩罚系数 `β`（非负）。
    pub const fn beta(self) -> f32 {
        self.beta
    }
}

/// 裁剪范围 ε 必须是非负有限数。
fn check_epsilon(epsilon: f32) -> Result<()> {
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("裁剪范围 ε={epsilon} 必须非负有限")).with(SITE),
        );
    }
    Ok(())
}

/// KL 惩罚系数 β 必须是非负有限数。
fn check_beta(beta: f32) -> Result<()> {
    if beta < 0.0 || !beta.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("KL 惩罚系数 β={beta} 必须非负有限")).with(SITE),
        );
    }
    Ok(())
}

/// 校验四个逐元素数组等长且非空，返回元素数。
fn check_quad_len(a: usize, b: usize, c: usize, d: usize) -> Result<usize> {
    check_rows(a)?;
    check_len(b, a, "old_logp")?;
    check_len(c, a, "advantage")?;
    check_len(d, a, "out")?;
    Ok(a)
}

/// 单 token 裁剪代理：`min(ratio·a, clamp(ratio, low, high)·a)`，`ratio = exp(n − old)`。
fn clipped_term(n: f32, old: f32, a: f32, low: f32, high: f32) -> f32 {
    let ratio = (n - old).exp();
    let clipped = ratio.clamp(low, high);
    (ratio * a).min(clipped * a)
}

/// 逐 token 裁剪代理：`min(ratio·A, clamp(ratio, 1−ε, 1+ε)·A)`，写回 `out`。
///
/// `ratio = exp(new_logp − old_logp)`。逐元素独立，advantage 由调用方按 token 广播
/// （GRPO 中同一 completion 的所有 token 共享同一 advantage）。
///
/// # Errors
///
/// `ε` 为负、四个数组任一为空或不等长时返回 [`Code::Invalid`]。
pub fn clipped_surrogate(
    new_logp: &[f32],
    old_logp: &[f32],
    advantages: &[f32],
    epsilon: f32,
    out: &mut [f32],
) -> Result<()> {
    check_epsilon(epsilon)?;
    check_quad_len(new_logp.len(), old_logp.len(), advantages.len(), out.len())?;
    let low = 1.0 - epsilon;
    let high = 1.0 + epsilon;
    for (((o, &n), &old), &a) in out.iter_mut().zip(new_logp).zip(old_logp).zip(advantages) {
        *o = clipped_term(n, old, a, low, high);
    }
    Ok(())
}

/// GRPO 组合损失：`−mean(min(ratio·A, clamp(ratio)·A)) + β·mean(KL_ref)`。
///
/// `new_logp`/`old_logp`/`advantages`/`ref_kl` 四数组等长（逐 token 对齐），`ref_kl` 由
/// [`crate::distill::kl_per_row`] 预计算（每 token 的 `KL(π ‖ π_ref)`）。`β=0` 退化为纯
/// 裁剪策略梯度，`ε=0` 退化为无裁剪的 REINFORCE + KL 惩罚。
///
/// # Errors
///
/// `ε` 或 `β` 为负、四数组任一为空或不等长时返回 [`Code::Invalid`]。
pub fn grpo_loss(
    new_logp: &[f32],
    old_logp: &[f32],
    advantages: &[f32],
    ref_kl: &[f32],
    grpo: Grpo,
) -> Result<f32> {
    check_epsilon(grpo.epsilon)?;
    check_beta(grpo.beta)?;
    let rows = check_quad_len(
        new_logp.len(),
        old_logp.len(),
        advantages.len(),
        ref_kl.len(),
    )?;
    let low = 1.0 - grpo.epsilon;
    let high = 1.0 + grpo.epsilon;
    let mut surrogate_sum = 0.0f64;
    let mut kl_sum = 0.0f64;
    for (((&n, &old), &a), &kl) in new_logp.iter().zip(old_logp).zip(advantages).zip(ref_kl) {
        surrogate_sum += clipped_term(n, old, a, low, high) as f64;
        kl_sum += kl as f64;
    }
    let surrogate_mean = surrogate_sum / rows as f64;
    let kl_mean = kl_sum / rows as f64;
    Ok((-surrogate_mean + grpo.beta as f64 * kl_mean) as f32)
}

#[cfg(test)]
#[path = "../tests/rl_tests.rs"]
mod tests;
