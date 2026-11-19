//! 分类损失与命中统计：前向评估量的纯计算函数。
//!
//! logits 按行展平成 `[rows, vocab]`，标签以 f32 承载类别索引（与 `embed_f32` 同一条索引惯例：
//! 内核取下标时转整数）。本模块不进 `op` 设备核注册表——反向公式属 P1 未接入范围，这里只算
//! 前向的 NLL 与命中率，供会话与编排器记进指标看板。
//!
//! 数值稳定：每行先减行内最大值再取指数，避免 `exp` 上溢；NLL 用 log-sum-exp 一行算出，不先
//! 归一化 softmax 再取对数（少一次除法，也避开零概率下的 `-inf`）。

use diagnostic::{Code, Error, Result};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::loss";

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
    let index = label as usize;
    if index >= vocab {
        let message = format!("标签索引 {index} 越界（词表 {vocab}）");
        return Err(Error::new(Code::Invalid, message).with(SITE));
    }
    Ok(())
}

/// 一行 logits 的 NLL：`-log softmax(label) = logsumexp − logits[label]`。
fn row_nll(row: &[f32], label: usize) -> f32 {
    let max = row.iter().copied().fold(f32::MIN, f32::max);
    let sum_exp: f32 = row.iter().map(|value| (value - max).exp()).sum();
    let logsumexp = max + sum_exp.ln();
    logsumexp - row[label]
}

/// 一行 argmax；平局取首个命中的下标。
fn row_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::MIN;
    for (index, &value) in row.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = index;
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

#[cfg(test)]
mod tests {
use crate::{argmax_hits, cross_entropy};
use diagnostic::{Code, Result};

/// 对齐类别的 NLL 应趋近 0：logits 把类别 1 顶到 10，softmax(1)≈1。
#[test]
fn cross_entropy_is_near_zero_when_the_label_matches_argmax() -> Result<()> {
    let logits = [0.0, 10.0, 0.0];
    let labels = [1.0];
    let loss = cross_entropy(&logits, &labels, 3)?;
    assert!(loss < 1e-3, "对齐类别 NLL 应趋近 0，得到 {loss}");
    Ok(())
}

/// 错配类别被惩罚：同一分布指向类别 0，NLL 约等于两logit 差（10）。
#[test]
fn cross_entropy_penalizes_the_misaligned_label() -> Result<()> {
    let logits = [0.0, 10.0, 0.0];
    let labels = [0.0];
    let loss = cross_entropy(&logits, &labels, 3)?;
    assert!(loss > 9.0 && loss < 11.0, "错配 NLL 应接近 10，得到 {loss}");
    Ok(())
}

/// 多行 NLL 取算术平均。
#[test]
fn cross_entropy_averages_over_rows() -> Result<()> {
    let logits = [0.0, 10.0, 0.0, 10.0, 0.0, 0.0];
    let labels = [1.0, 0.0];
    let loss = cross_entropy(&logits, &labels, 3)?;
    assert!(loss < 1e-3, "两行都对齐，均值应趋近 0，得到 {loss}");
    Ok(())
}

/// argmax 命中统计：正确一行、错误一行分别对账。
#[test]
fn argmax_hits_counts_correct_and_total() -> Result<()> {
    let logits = [0.0, 10.0, 0.0, 10.0, 0.0, 0.0];
    let labels = [1.0, 1.0];
    let (correct, total) = argmax_hits(&logits, &labels, 3)?;
    assert_eq!(total, 2);
    assert_eq!(correct, 1, "第二行 argmax 是 0，标签是 1，不计命中");
    Ok(())
}

/// 几何与标签校验：词表为 0、空标签、长度不符、索引越界、负索引、非整数索引都拒绝。
#[test]
fn rejects_invalid_geometry_and_labels() {
    let logits = [0.0, 10.0, 0.0];

    let empty = cross_entropy(&logits, &[], 3);
    assert_eq!(empty.expect_err("空标签").code(), Code::Invalid);

    let bad_vocab = cross_entropy(&logits, &[1.0], 0);
    assert_eq!(bad_vocab.expect_err("词表为 0").code(), Code::Invalid);

    let bad_len = cross_entropy(&[0.0, 1.0], &[1.0], 3);
    assert_eq!(bad_len.expect_err("长度不符").code(), Code::Invalid);

    let out_of_range = argmax_hits(&logits, &[3.0], 3);
    assert_eq!(out_of_range.expect_err("索引越界").code(), Code::Invalid);

    let negative = cross_entropy(&logits, &[-1.0], 3);
    assert_eq!(negative.expect_err("负索引").code(), Code::Invalid);

    let fractional = cross_entropy(&logits, &[1.5], 3);
    assert_eq!(fractional.expect_err("非整数索引").code(), Code::Invalid);
}

/// 极端大 logit 不应让 exp 上溢：减行内最大值后指数都 ≤ 1，NLL 仍有限且对齐类别趋近 0。
#[test]
fn cross_entropy_stays_finite_on_extreme_logits() -> Result<()> {
    let logits = [1e30, 0.0, -1e30];
    let labels = [0.0];
    let loss = cross_entropy(&logits, &labels, 3)?;
    assert!(loss.is_finite(), "极端 logit 下 NLL 必须有限，得到 {loss}");
    assert!(
        loss < 1e-3,
        "对齐最大 logit 的类别 NLL 应趋近 0，得到 {loss}"
    );
    Ok(())
}

/// argmax 平局取首个命中的下标：相等时按下标最小一侧计数。
#[test]
fn argmax_breaks_ties_to_the_first_index() -> Result<()> {
    let logits = [5.0, 5.0, 0.0];
    let first_hit = argmax_hits(&logits, &[0.0], 3)?;
    assert_eq!(first_hit, (1, 1), "平局取下标 0");
    let second_miss = argmax_hits(&logits, &[1.0], 3)?;
    assert_eq!(second_miss, (0, 1), "下标 1 不是平局首命中");
    Ok(())
}

/// 一行对齐、一行错配时，均值落在两者 NLL 的中间区间。
#[test]
fn cross_entropy_averages_mixed_alignment() -> Result<()> {
    let logits = [0.0, 10.0, 0.0, 10.0, 0.0, 0.0];
    let labels = [1.0, 1.0];
    let loss = cross_entropy(&logits, &labels, 3)?;
    assert!(
        loss > 4.0 && loss < 6.0,
        "对齐行近 0、错配行近 10，均值应约 5，得到 {loss}"
    );
    Ok(())
}
}

