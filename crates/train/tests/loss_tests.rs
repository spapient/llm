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

/// 目标函数族的端到端选择：分类损失、蒸馏与 GRPO 走同一张 `Loss` 表。
mod objective {
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

/// 蒸馏目标：温度缩放、KL 与 α 混合的数值契约。
mod distill {
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
