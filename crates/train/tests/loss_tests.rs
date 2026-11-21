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
