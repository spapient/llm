//! 目标函数：分类/回归/排序损失、蒸馏族与策略族共用一份几何校验与数值助手。
//!
//! 三族同住一处是因为它们校验的是同一个契约（`[rows, vocab]` 展平、标签在词表内、行数为正），
//! 拆开就得让每族各带一份长度/词表/行数检查或互相 `pub(crate)` 借用；这里只有一份。

use crate::rl::{Grpo, grpo_loss};
use diagnostic::{Code, Error, Result};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::loss";

/// 蒸馏目标的诊断站点：本文件承载"目标函数"这一轴，站名沿用蒸馏自己的名字，
/// 使几何校验失败与蒸馏超参非法在诊断里仍可区分。
pub(crate) const DISTILL_SITE: &str = "train::distill";

/// 校验标签是闭区间 `[0, vocab)` 内的非负整数；返回标签行数。
///
/// # Errors
///
/// 词表为 0、标签为空、logits 长度不是 `rows · vocab`、或标签越界/非整数/为负时返回
/// [`Code::Invalid`]。
pub(crate) fn check_geometry(logits: &[f32], labels: &[f32], vocab: usize) -> Result<usize> {
    let rows = labels.len();
    check_vocab(vocab)?;
    check_rows(rows)?;
    check_logits_len(logits.len(), rows, vocab)?;
    check_labels(labels, vocab)?;
    Ok(rows)
}

/// 逐行校验标签索引；循环单独收口，避免几何校验函数把分支堆在一处。
fn check_labels(labels: &[f32], vocab: usize) -> Result<()> {
    for &label in labels {
        check_label(label, vocab)?;
    }
    Ok(())
}

pub(crate) fn check_vocab(vocab: usize) -> Result<()> {
    if vocab == 0 {
        return Err(Error::new(Code::Invalid, "词表宽度不能为 0").with(SITE));
    }
    Ok(())
}

pub(crate) fn check_rows(rows: usize) -> Result<()> {
    if rows == 0 {
        return Err(Error::new(Code::Invalid, "标签为空：没有可评估的样本行").with(SITE));
    }
    Ok(())
}

pub(crate) fn check_logits_len(logits_len: usize, rows: usize, vocab: usize) -> Result<()> {
    let expected = rows * vocab;
    if logits_len != expected {
        let message =
            format!("logits 元素数 {logits_len} 与 {rows} 行 × {vocab} 词表 = {expected} 不符");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok(())
}

/// 校验两个长度相等；不等返回 [`Code::Invalid`]。`what` 标识被校验的一侧。
pub(crate) fn check_len(actual: usize, expected: usize, what: &str) -> Result<()> {
    if actual != expected {
        let message = format!("{what} 长度 {actual} 与期望 {expected} 不符");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok(())
}

fn check_label(label: f32, vocab: usize) -> Result<()> {
    if label < 0.0 || label.fract() != 0.0 {
        let message = format!("标签 {label} 不是非负整数索引");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    let idx = label as usize;
    if idx >= vocab {
        let message = format!("标签索引 {idx} 越界（词表 {vocab}）");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok(())
}

/// 一行的数值稳定归约：返回 `(行内最大值 · scale, Σ exp(v·scale − max))`。
pub(crate) fn exp_sum(row: &[f32], scale: f32) -> (f32, f32) {
    let max = row.iter().copied().fold(f32::MIN, f32::max) * scale;
    let sum = row.iter().map(|value| (value * scale - max).exp()).sum();
    (max, sum)
}

/// 一行的 `logsumexp`：`max + ln Σ exp(v − max)`。
pub(crate) fn logsumexp(row: &[f32]) -> f32 {
    let (max, sum) = exp_sum(row, 1.0);
    max + sum.ln()
}

/// 一行 logits 的 NLL：`-log softmax(label) = logsumexp − logits[label]`。
fn row_nll(row: &[f32], label: usize) -> f32 {
    logsumexp(row) - row[label]
}

/// 一行 argmax；平局取首个命中的下标。
fn row_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::MIN;
    for (idx, &value) in row.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = idx;
        }
    }
    best
}

/// 单次行扫描同时算出 NLL 总和（f64）、命中条数与总行数；编排器一次遍历取两个评估量，
/// 不必把 logits 行扫两遍。
///
/// # Errors
///
/// 几何不自洽或标签越界时返回 [`Code::Invalid`]。
pub(crate) fn evaluate(
    logits: &[f32],
    labels: &[f32],
    vocab: usize,
) -> Result<(f64, usize, usize)> {
    let rows = check_geometry(logits, labels, vocab)?;
    let mut nll_sum = 0.0f64;
    let mut correct = 0usize;
    for (row, &label) in labels.iter().enumerate() {
        let start = row * vocab;
        let slice = &logits[start..start + vocab];
        let label = label as usize;
        nll_sum += row_nll(slice, label) as f64;
        if row_argmax(slice) == label {
            correct += 1;
        }
    }
    Ok((nll_sum, correct, rows))
}

/// 交叉熵均值：所有行 NLL 的算术平均。
///
/// # Errors
///
/// 几何不自洽或标签越界时返回 [`Code::Invalid`]。
pub fn cross_entropy(logits: &[f32], labels: &[f32], vocab: usize) -> Result<f32> {
    let (nll_sum, _, rows) = evaluate(logits, labels, vocab)?;
    Ok((nll_sum / rows as f64) as f32)
}

/// 分类命中率：`(正确条数, 总条数)`。
///
/// # Errors
///
/// 几何不自洽或标签越界时返回 [`Code::Invalid`]。
pub fn argmax_hits(logits: &[f32], labels: &[f32], vocab: usize) -> Result<(usize, usize)> {
    let (_, correct, rows) = evaluate(logits, labels, vocab)?;
    Ok((correct, rows))
}

/// 均方误差：`mean((pred − target)²)`。
///
/// # Errors
///
/// 两缓冲等长且非空，否则返回 [`Code::Invalid`]。
pub fn mse(pred: &[f32], target: &[f32]) -> Result<f32> {
    if pred.len() != target.len() || pred.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!("mse: 形状不一致 {} vs {}", pred.len(), target.len()),
        ));
    }
    let sum: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let d = (p - t) as f64;
            d * d
        })
        .sum();
    Ok((sum / pred.len() as f64) as f32)
}

/// 二分类交叉熵（带 logits 的数值稳定式）。
///
/// 逐元素 `max(x,0) − x·y + ln(1+exp(−|x|))`：`x` 为 logit，`y ∈ {0,1}`。
/// 直接对概率取 log 会在 `softmax` 饱和处溢出，这里不先过 sigmoid。
///
/// # Errors
///
/// 两缓冲等长且非空，否则返回 [`Code::Invalid`]。
pub fn bce_with_logits(logits: &[f32], target: &[f32]) -> Result<f32> {
    if logits.len() != target.len() || logits.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!("bce: 形状不一致 {} vs {}", logits.len(), target.len()),
        ));
    }
    let sum: f64 = logits
        .iter()
        .zip(target.iter())
        .map(|(x, y)| {
            let x = *x as f64;
            let y = *y as f64;
            x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln()
        })
        .sum();
    Ok((sum / logits.len() as f64) as f32)
}

/// Huber（smooth L1）：`|d| ≤ δ` 走平方段，超出后线性接续，取均值。
///
/// # Errors
///
/// 形状不一致、空输入或 `delta ≤ 0` 时返回说明。
pub fn huber(pred: &[f32], target: &[f32], delta: f32) -> Result<f32> {
    if pred.len() != target.len() || pred.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!("huber: 形状不一致 {} vs {}", pred.len(), target.len()),
        ));
    }
    if delta <= 0.0 {
        return Err(Error::new(Code::Invalid, "huber: delta 必须为正"));
    }
    let d = delta as f64;
    let sum: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| {
            let e = (*p as f64 - *t as f64).abs();
            if e <= d {
                0.5 * e * e
            } else {
                d * (e - 0.5 * d)
            }
        })
        .sum();
    Ok((sum / pred.len() as f64) as f32)
}

/// Hinge（`margin = 1`）：`mean(max(0, 1 − y·p))`，`y` 取 `−1/+1`。
///
/// # Errors
///
/// 形状不一致或空输入时返回说明。
pub fn hinge(pred: &[f32], target: &[f32]) -> Result<f32> {
    if pred.len() != target.len() || pred.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!("hinge: 形状不一致 {} vs {}", pred.len(), target.len()),
        ));
    }
    let sum: f64 = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (1.0 - *t as f64 * *p as f64).max(0.0))
        .sum();
    Ok((sum / pred.len() as f64) as f32)
}

#[cfg(test)]
#[path = "../tests/loss_tests.rs"]
mod tests;

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
        /// 逐 token 参考 KL（由 [`kl_per_row`] 预计算）。
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

// ---- 蒸馏目标：温度、α 混合与逐 token 参考 KL ----
/// 蒸馏损失配置：软目标温度 `T` 与软损失权重 `α`（Hinton 2015 约定）。
#[derive(Clone, Copy, Debug)]
pub struct Distill {
    temperature: f32,
    alpha: f32,
}

impl Distill {
    /// 构造蒸馏配置；`temperature`/`alpha` 的合法性在损失入口校验（保持 `const`）。
    pub const fn new(temperature: f32, alpha: f32) -> Self {
        Self { temperature, alpha }
    }

    /// 软目标温度 `T`（正数）。
    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    /// 软损失权重 `α`，取值 `[0,1]`。
    pub const fn alpha(self) -> f32 {
        self.alpha
    }
}

/// 温度必须是正的有限数。
fn check_temperature(temperature: f32) -> Result<()> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("温度 {temperature} 必须是正有限数"))
                .with(DISTILL_SITE),
        );
    }
    Ok(())
}

/// 软损失权重 α 必须在闭区间 `[0, 1]`。
fn check_alpha(alpha: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&alpha) || !alpha.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("软损失权重 α={alpha} 必须在 [0,1]"))
                .with(DISTILL_SITE),
        );
    }
    Ok(())
}

/// 校验 logits 几何（词表/行数/长度/`out` 等长），返回行数。
fn check_logits_shape(logits_len: usize, out_len: usize, vocab: usize) -> Result<usize> {
    check_vocab(vocab)?;
    let rows = logits_len / vocab;
    check_rows(rows)?;
    check_logits_len(logits_len, rows, vocab)?;
    check_len(out_len, logits_len, "out")?;
    Ok(rows)
}

/// 校验一对 logits 的几何（词表/等长/行数/长度），返回行数。
fn check_pair_shape(student_len: usize, teacher_len: usize, vocab: usize) -> Result<usize> {
    check_vocab(vocab)?;
    check_len(student_len, teacher_len, "教师 logits")?;
    let rows = student_len / vocab;
    check_rows(rows)?;
    check_logits_len(student_len, rows, vocab)?;
    Ok(rows)
}

/// 单行 log-softmax：`o = slice − logsumexp(slice)`。
fn row_log_softmax(slice: &[f32], o: &mut [f32]) {
    let lse = logsumexp(slice);
    for (o, &v) in o.iter_mut().zip(slice) {
        *o = v - lse;
    }
}

/// 单行 KL：`sum_v softmax(s/T)[v] · (log softmax(s/T)[v] − log softmax(t/T)[v])`。
fn row_kl(s: &[f32], t: &[f32], temperature: f32) -> f32 {
    let inv = 1.0 / temperature;
    let (s_max, s_sum) = exp_sum(s, inv);
    let (t_max, t_sum) = exp_sum(t, inv);
    let (s_lse, t_lse) = (s_max + s_sum.ln(), t_max + t_sum.ln());

    let mut kl = 0.0f32;
    for (&sv, &tv) in s.iter().zip(t) {
        let log_ps = sv * inv - s_lse;
        let log_pt = tv * inv - t_lse;
        kl += (log_ps - log_pt) * log_ps.exp();
    }
    kl
}

/// 行内 log-softmax，数值稳定：`out = logits − logsumexp(logits)`。
///
/// `logits` 按行展平成 `[rows, vocab]`，`out` 与 `logits` 等长、逐元素写回；`exp(out)` 每行和为 1。
///
/// # Errors
///
/// 词表为 0、无行、logits 长度不是 `vocab` 的整数倍、或 `out` 与 `logits` 不等长时返回
/// [`Code::Invalid`]。
pub fn log_softmax(logits: &[f32], out: &mut [f32], vocab: usize) -> Result<()> {
    check_logits_shape(logits.len(), out.len(), vocab)?;
    for (o_row, s_row) in out.chunks_exact_mut(vocab).zip(logits.chunks_exact(vocab)) {
        row_log_softmax(s_row, o_row);
    }
    Ok(())
}

/// 温度缩放的 KL 散度均值：`KL(softmax(student/T) ‖ softmax(teacher/T))`。
///
/// 逐行计算后取算术平均（与 [`crate::loss::cross_entropy`] 的均值约定一致）。KL ≥ 0，
/// 学生与教师同分布时为 0。
///
/// # Errors
///
/// 词表为 0、温度非正、两个 logits 不等长、长度不是 `vocab` 整数倍、或无行时返回 [`Code::Invalid`]。
pub fn kl_div(student: &[f32], teacher: &[f32], vocab: usize, temperature: f32) -> Result<f32> {
    check_temperature(temperature)?;
    let rows = check_pair_shape(student.len(), teacher.len(), vocab)?;
    let mut kl_sum = 0.0f64;
    for (s, t) in student.chunks_exact(vocab).zip(teacher.chunks_exact(vocab)) {
        kl_sum += row_kl(s, t, temperature) as f64;
    }
    Ok((kl_sum / rows as f64) as f32)
}

/// 每行 KL 散度：`KL(softmax(student/T) ‖ softmax(teacher/T))`，逐行写回 `out`（长 `rows`）。
///
/// [`kl_div`] 是它的行均值；GRPO 需要每 token 的参考 KL，此处逐行输出，供逐 token 的
/// KL 惩罚（`β·KL`）使用。
///
/// # Errors
///
/// 词表为 0、温度非正、两个 logits 不等长、长度不是 `vocab` 整数倍、无行、或 `out` 与行数
/// 不等长时返回 [`Code::Invalid`]。
pub fn kl_per_row(
    student: &[f32],
    teacher: &[f32],
    vocab: usize,
    temperature: f32,
    out: &mut [f32],
) -> Result<()> {
    check_temperature(temperature)?;
    let rows = check_pair_shape(student.len(), teacher.len(), vocab)?;
    check_len(out.len(), rows, "out")?;
    for (o, (s, t)) in out
        .iter_mut()
        .zip(student.chunks_exact(vocab).zip(teacher.chunks_exact(vocab)))
    {
        *o = row_kl(s, t, temperature);
    }
    Ok(())
}

/// Hinton 蒸馏损失：`(1−α)·CE(student, labels) + α·T²·KL(student/T ‖ teacher/T)`。
///
/// 硬标签交叉熵沿用 [`crate::loss::cross_entropy`]，软目标用温度缩放的 KL。`α=0` 退化为纯
/// 交叉熵，`α=1` 退化为纯软蒸馏。
///
/// # Errors
///
/// 几何不自洽、标签越界、温度非正或 `α` 不在 `[0,1]` 时返回 [`Code::Invalid`]。
pub fn distillation_loss(
    student: &[f32],
    teacher: &[f32],
    labels: &[f32],
    vocab: usize,
    dist: Distill,
) -> Result<f32> {
    check_alpha(dist.alpha)?;
    let hard = crate::loss::cross_entropy(student, labels, vocab)?;
    let soft = kl_div(student, teacher, vocab, dist.temperature)?;
    let scale = dist.temperature * dist.temperature;
    Ok((1.0 - dist.alpha) * hard + dist.alpha * scale * soft)
}
