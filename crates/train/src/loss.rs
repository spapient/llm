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

pub mod objective;
