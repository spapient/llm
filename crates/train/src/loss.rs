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
mod tests {
    use crate::{argmax_hits, bce_with_logits, cross_entropy, hinge, huber, mse};
    use diagnostic::{Code, Result};

    #[test]
    fn mse_is_zero_for_identical_buffers() -> Result<()> {
        let a = [1.0f32, -2.0, 3.5];
        assert!(mse(&a, &a)? == 0.0);
        let b = [1.0f32, -2.0, 4.5];
        assert!((mse(&a, &b)? - 1.0 / 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn huber_matches_piecewise_formula() -> Result<()> {
        let pred = [0.0f32, 3.0, -1.5, 10.0];
        let target = [0.0f32, 1.0, -1.0, 0.0];
        // 误差 0 / 2 / 0.5 / 10，δ=1：0 + 1×(2−0.5) + 0.5×0.25 + 1×(10−0.5)，再除以 4。
        let want = (0.0f64 + 1.5 + 0.125 + 9.5) / 4.0;
        let got = huber(&pred, &target, 1.0)?;
        assert!(
            (f64::from(got) - want).abs() < 1e-6,
            "huber {got} vs {want}"
        );
        // δ 大到盖住全部误差时退化为 0.5×MSE。
        let got = huber(&[0.5f32, -0.25], &[0.0f32, 0.0], 100.0)?;
        assert!((f64::from(got) - (0.25f64 + 0.0625) / 4.0).abs() < 1e-6);
        assert!(huber(&[1.0f32], &[0.0f32], 0.0).is_err());
        Ok(())
    }

    #[test]
    fn hinge_is_zero_when_margin_is_satisfied() -> Result<()> {
        // y·p = 2 与 3 都过了 margin；y·p = −1 的样本贡献 1−(−1) = 2。
        let got = hinge(&[2.0f32, 3.0, -1.0], &[1.0f32, 1.0, 1.0])?;
        assert!((f64::from(got) - 2.0 / 3.0).abs() < 1e-6, "hinge {got}");
        // 负样本 y=−1、p=0.5 → 1 − (−0.5) = 1.5。
        assert!((f64::from(hinge(&[0.5f32], &[-1.0f32])?) - 1.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn bce_is_stable_at_saturation() -> Result<()> {
        // logit 100 且标签 1：损失应趋 0；标签 0 时应趋 100，且不出现 NaN/Inf。
        let logits = [100.0f32];
        assert!(bce_with_logits(&logits, &[1.0])? < 1e-6);
        assert!((bce_with_logits(&logits, &[0.0])? - 100.0).abs() < 1e-3);
        // 对称点：logit 0 与任意标签都应得到 ln2。
        assert!((bce_with_logits(&[0.0], &[1.0])? - std::f32::consts::LN_2).abs() < 1e-5);
        Ok(())
    }

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

pub mod objective {
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
                Self::Distill { teacher, config } => {
                    distill(logits, teacher, labels, vocab, *config)
                }
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
                (out.loss - distillation_loss(&student, &teacher, &labels, 3, config)?).abs()
                    < 1e-5,
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
}
