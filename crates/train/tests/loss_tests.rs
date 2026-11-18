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
