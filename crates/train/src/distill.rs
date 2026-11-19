//! 蒸馏损失族：log-softmax、KL 散度与 Hinton 蒸馏损失（软标签）。
//!
//! 与 [`crate::loss`]（硬标签 NLL）互补。蒸馏用教师分布作软标签，KL 度量学生与教师在温度
//! 缩放下两个 softmax 的距离；RL（GRPO / RLHF）复用同一套 log-softmax（token log-prob）与
//! KL（策略到参考模型）原语。本模块与 loss 一致：只算前向纯计算，不进 op 设备核注册表，
//! 反向公式属 P1。
//!
//! 数值稳定沿用 loss 的口径：log-softmax 每行先减行内最大值再取指数，KL 由
//! `sum softmax(s/T) · (log softmax(s/T) − log softmax(t/T))` 一行算出，不先归一化再取对数。
//!
//! 校验沿用 loss 的收口模式：每个入口只做一次 `?`，把几何/参数校验推给纯校验函数，避免分支
//! 堆在计算函数里抬高复杂度。

use diagnostic::{Code, Error, Result};

use crate::loss::{check_len, check_logits_len, check_rows, check_vocab};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::distill";

/// 蒸馏损失配置：软目标温度 `T` 与软损失权重 `α`（Hinton 2015 约定）。
///
/// 损失 = `(1−α)·CE(student, labels) + α·T²·KL(student/T ‖ teacher/T)`。
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
            Error::new(Code::Invalid, format!("温度 {temperature} 必须是正有限数")).with(SITE),
        );
    }
    Ok(())
}

/// 软损失权重 α 必须在闭区间 `[0, 1]`。
fn check_alpha(alpha: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&alpha) || !alpha.is_finite() {
        return Err(
            Error::new(Code::Invalid, format!("软损失权重 α={alpha} 必须在 [0,1]")).with(SITE),
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
    let max = slice.iter().copied().fold(f32::MIN, f32::max);
    let logsumexp = max + slice.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    for (o, &v) in o.iter_mut().zip(slice) {
        *o = v - logsumexp;
    }
}

/// 单行 KL：`sum_v softmax(s/T)[v] · (log softmax(s/T)[v] − log softmax(t/T)[v])`。
fn row_kl(s: &[f32], t: &[f32], temperature: f32) -> f32 {
    let inv = 1.0 / temperature;
    let s_max = s.iter().copied().fold(f32::MIN, f32::max) * inv;
    let s_lse = s_max + s.iter().map(|v| (v * inv - s_max).exp()).sum::<f32>().ln();
    let t_max = t.iter().copied().fold(f32::MIN, f32::max) * inv;
    let t_lse = t_max + t.iter().map(|v| (v * inv - t_max).exp()).sum::<f32>().ln();

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

#[cfg(test)]
mod tests {
use crate::{Distill, distillation_loss, kl_div, kl_per_row, log_softmax};
use diagnostic::{Code, Result};

/// log-softmax 的指数每行和为 1（数值稳定的 softmax 归一化）。
#[test]
fn log_softmax_exponentiates_to_a_distribution() -> Result<()> {
    let logits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut out = [0.0; 6];
    log_softmax(&logits, &mut out, 3)?;
    for row in out.as_chunks::<3>().0 {
        let sum: f32 = row.iter().map(|v| v.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax 每行和应趋近 1，得到 {sum}"
        );
    }
    Ok(())
}

/// 大 logits 下不溢出：每行先减最大值，10^5 量级的 logits 也稳定。
#[test]
fn log_softmax_is_stable_for_large_logits() -> Result<()> {
    let logits = [0.0, 100_000.0, 0.0];
    let mut out = [0.0; 3];
    log_softmax(&logits, &mut out, 3)?;
    assert!(
        out.iter().all(|v| v.is_finite()),
        "大 logits 不应溢出，得到 {out:?}"
    );
    // 类别 1 顶到最高且高出 10^5，log-softmax(1) 应趋近 0。
    assert!(
        out[1].abs() < 1e-3,
        "最大类 log-prob 应趋近 0，得到 {}",
        out[1]
    );
    Ok(())
}

/// 学生与教师同分布时 KL 为 0。
#[test]
fn kl_div_is_zero_when_distributions_match() -> Result<()> {
    let logits = [1.0, 2.0, 3.0];
    let kl = kl_div(&logits, &logits, 3, 1.0)?;
    assert!(kl.abs() < 1e-6, "同分布 KL 应为 0，得到 {kl}");
    Ok(())
}

/// 学生趋近独热、教师均匀时，KL 趋近 ln(词表宽)。
#[test]
fn kl_div_of_a_peaked_student_against_uniform_teacher() -> Result<()> {
    let kl = kl_div(&[10.0, 0.0], &[0.0, 0.0], 2, 1.0)?;
    assert!(kl > 0.0, "KL 应严格为正，得到 {kl}");
    assert!((kl - 2.0f32.ln()).abs() < 1e-2, "KL 应趋近 ln 2，得到 {kl}");
    Ok(())
}

/// α=0 时蒸馏损失退化为硬标签交叉熵。
#[test]
fn distillation_loss_with_alpha_zero_is_cross_entropy() -> Result<()> {
    let student = [0.0, 10.0, 0.0];
    let teacher = [0.0, 0.0, 0.0];
    let labels = [1.0];
    let loss = distillation_loss(&student, &teacher, &labels, 3, Distill::new(2.0, 0.0))?;
    let ce = crate::cross_entropy(&student, &labels, 3)?;
    assert!(
        (loss - ce).abs() < 1e-6,
        "α=0 应退化为 CE，得到 {loss} vs {ce}"
    );
    Ok(())
}

/// α=1 时蒸馏损失退化为纯软蒸馏（T² 缩放 KL）。
#[test]
fn distillation_loss_with_alpha_one_is_scaled_kl() -> Result<()> {
    let student = [1.0, 2.0, 3.0];
    let teacher = [0.0, 0.0, 0.0];
    let labels = [0.0];
    let temperature = 3.0;
    let loss = distillation_loss(
        &student,
        &teacher,
        &labels,
        3,
        Distill::new(temperature, 1.0),
    )?;
    let expected = temperature * temperature * kl_div(&student, &teacher, 3, temperature)?;
    assert!(
        (loss - expected).abs() < 1e-5,
        "α=1 应退化为 T²·KL，得到 {loss} vs {expected}"
    );
    Ok(())
}

/// 温度非正被拒绝。
#[test]
fn rejects_nonpositive_temperature() {
    assert!(matches!(
        kl_div(&[0.0, 0.0], &[0.0, 0.0], 2, 0.0),
        Err(e) if e.code() == Code::Invalid
    ));
}

/// α 越界被拒绝。
#[test]
fn rejects_out_of_range_alpha() {
    assert!(matches!(
        distillation_loss(&[0.0, 0.0], &[0.0, 0.0], &[0.0], 2, Distill::new(1.0, 2.0)),
        Err(e) if e.code() == Code::Invalid
    ));
}

/// 学生与教师 logits 不等长被拒绝。
#[test]
fn rejects_length_mismatch() {
    assert!(matches!(
        kl_div(&[0.0, 0.0], &[0.0, 0.0, 0.0], 2, 1.0),
        Err(e) if e.code() == Code::Invalid
    ));
}

/// 每行 KL 与均值 KL 一致：同分布每行 KL 为 0。
#[test]
fn kl_per_row_is_zero_when_distributions_match() -> Result<()> {
    let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = [0.0; 2];
    kl_per_row(&logits, &logits, 3, 1.0, &mut out)?;
    assert!(
        out.iter().all(|v| v.abs() < 1e-6),
        "同分布每行 KL 应为 0，得到 {out:?}"
    );
    Ok(())
}

/// 每行 KL 的行均值等于 kl_div 的返回值。
#[test]
fn kl_per_row_averages_to_kl_div() -> Result<()> {
    let student = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
    let teacher = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut out = [0.0; 2];
    kl_per_row(&student, &teacher, 3, 1.0, &mut out)?;
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    let kl = kl_div(&student, &teacher, 3, 1.0)?;
    assert!(
        (mean - kl).abs() < 1e-5,
        "每行 KL 均值应等于 kl_div，得到 {mean} vs {kl}"
    );
    Ok(())
}

/// 每行 KL 的 out 与行数不等长被拒绝。
#[test]
fn kl_per_row_rejects_out_length_mismatch() {
    assert!(matches!(
        kl_per_row(&[0.0, 0.0], &[0.0, 0.0], 2, 1.0, &mut [0.0; 3]),
        Err(e) if e.code() == Code::Invalid
    ));
}
}

