//! 数值对拍的**逐元素与方向**判据：绝对误差、绝对+相对双项容差、有限值谓词与余弦方向一致性。
//!
//! 整批（一个全局分母）的相对误差口径在 [`crate::rel`]，它答的是「整批数合在一起差多少」，
//! 与这里的逐元素判据不可混用。三档判据的适用面：同尺度逐元素对拍用 [`approx_eq`]；量化
//! logits 的方向一致性用 [`cosine_similarity`] + [`COSINE_GUARD`]；解析梯度 vs 数值梯度这类
//! 随量级缩放的比对走 [`close`] 的「绝对 + 相对」双项预算，不套绝对容差。

/// 量化 logits 与 f32 参考的余弦回归下界（方向一致性，尺度不变）。
///
/// 合成基准与未来的真实样本集共用这一条阈值，避免各用例各抄一个数字、判据悄悄漂移。
pub const COSINE_GUARD: f32 = 0.999;

/// 逐个元素**绝对误差**比较：两套同尺度算法变体对拍走这一档。`NaN` 与 `NaN` 视为相等
/// （对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
///
/// 两个**相同的无限大**也算相等：`inf - inf` 是 `NaN`，只看差值的判定会把「两边同为无穷」
/// 判成不等，而它与 `NaN` 对 `NaN` 是同一个理由。
///
/// 长度不同直接返回 `false`：对拍的首要条件是形状一致。
pub fn approx_eq(left: &[f32], right: &[f32], tolerance: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right).all(|(a, b)| {
        if a == b || (a.is_nan() && b.is_nan()) {
            return true;
        }
        (a - b).abs() <= tolerance
    })
}

/// 双项容差标量判定：`|a-b| ≤ abs + rel·max(|a|,|b|)`。
///
/// 两侧同为 NaN、同为同号无穷算相等；异号无穷不算相等。
fn close_scalar(a: f32, b: f32, abs: f32, rel: f32) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() {
        return b.is_nan();
    }
    if a.is_infinite() {
        return false;
    }
    if b.is_infinite() {
        return false;
    }
    let tolerance = abs + rel * a.abs().max(b.abs());
    (a - b).abs() <= tolerance
}

/// 绝对 + 相对双项容差逐元素对拍：随量级缩放的比对（解析梯度 vs 数值梯度）走这一档。
///
/// 判据分三档：同尺度逐元素用 [`approx_eq`]；方向一致性用 [`cosine_similarity`]；
/// 量级相关的误差预算用本函数。长度不同直接返回 `false`。
pub fn close(left: &[f32], right: &[f32], abs: f32, rel: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .all(|(a, b)| close_scalar(*a, *b, abs, rel))
}

/// 绝对 + 相对双项容差断言：判据与 [`close`] 同源（`|a−b| ≤ abs + rel·|b|`），
/// 失败时把首个越界元素的下标与两侧取值写进 panic 消息——只留一句 `false` 的断言
/// 在逐元素核上无法定位是哪一格写错。
///
/// # Panics
///
/// 长度不同或任一元素越界时 panic。
#[track_caller]
pub fn assert_close(left: &[f32], right: &[f32], abs: f32, rel: f32) {
    assert_eq!(
        left.len(),
        right.len(),
        "对拍两侧长度不同（got {}，want {}）",
        left.len(),
        right.len()
    );
    for (index, (got, want)) in left.iter().zip(right).enumerate() {
        let tolerance = abs + rel * want.abs();
        assert!(
            (got - want).abs() <= tolerance,
            "第 {index} 个元素 {got} vs {want}（容差 {tolerance}）"
        );
    }
}

/// 切片内全部为有限值（非 NaN、非无穷）；数值对拍前的前置谓词。
#[must_use]
pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// 两条同长向量的余弦相似度；长度不同、或任一向量范数为零时返回 [`None`]。
///
/// 回归判据用：量化前向的 logits 与 f32 参考比对方向一致性（不要求逐元素相等）。累加在 `f64`
/// 完成，避免长向量上 `f32` 平方和与点积的累加误差把相似度压低；返回值落在 `[-1, 1]`。
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut left_sq = 0.0f64;
    let mut right_sq = 0.0f64;
    for (a, b) in left.iter().zip(right) {
        let a = f64::from(*a);
        let b = f64::from(*b);
        dot += a * b;
        left_sq += a * a;
        right_sq += b * b;
    }
    let denom = (left_sq * right_sq).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some((dot / denom) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_is_elementwise_and_length_sensitive() {
        assert!(approx_eq(&[1.0, 2.0], &[1.0, 2.0], 0.0));
        assert!(approx_eq(&[1.0, 2.0], &[1.0, 2.0004], 1e-3));
        assert!(!approx_eq(&[1.0, 2.0], &[1.0, 2.01], 1e-3));
        assert!(!approx_eq(&[1.0], &[1.0, 1.0], 1.0), "长度不同必须为假");
    }

    #[test]
    fn nan_matches_nan() {
        assert!(approx_eq(&[f32::NAN], &[f32::NAN], 0.0));
    }

    /// `inf - inf` 是 `NaN`：差值判定会把「两边同为无穷」判成不等，而它与 NaN 对 NaN 同理。
    #[test]
    fn equal_infinities_match() {
        for value in [f32::INFINITY, f32::NEG_INFINITY] {
            assert!(approx_eq(&[value], &[value], 0.0), "{value}");
        }
        assert!(!approx_eq(&[f32::INFINITY], &[f32::NEG_INFINITY], 1.0));
        assert!(!approx_eq(&[f32::INFINITY], &[1.0], 1.0));
    }

    #[test]
    fn close_allows_relative_slack_for_large_values() {
        assert!(close(&[1000.0], &[1000.5], 1e-3, 1e-3));
        assert!(!close(&[1000.0], &[1002.0], 1e-3, 1e-3));
        assert!(!close(&[1.0], &[1.0, 1.0], 0.0, 0.0), "长度不同必须为假");
    }

    #[test]
    fn close_nan_and_equal_infinites_match() {
        assert!(close(&[f32::NAN], &[f32::NAN], 0.0, 0.0));
        assert!(close(&[f32::INFINITY], &[f32::INFINITY], 0.0, 0.0));
        assert!(!close(&[f32::INFINITY], &[f32::NEG_INFINITY], 1.0, 1.0));
    }

    #[test]
    fn max_rel_err_uses_a_single_global_denominator() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        // 分母是 max|want| = 2.0，不是逐元素各自归一。
        let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
        assert!((err - 0.05).abs() < 1e-6, "{err}");
        assert!(within_rel(&[1.0, 2.0], &[1.0, 2.0], 1e-5));
        assert!(!within_rel(&[1.0, 2.5], &[1.0, 2.0], 1e-5));
    }

    #[test]
    fn max_rel_err_is_infinity_on_mismatched_lengths_or_nonfinite_mismatch() {
        assert_eq!(max_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY, "长度不同");
        assert_eq!(
            max_rel_err(&[f32::NAN], &[0.0]),
            f64::INFINITY,
            "NaN 对有限"
        );
        assert_eq!(
            max_rel_err(&[f32::INFINITY], &[1.0]),
            f64::INFINITY,
            "无穷对有限"
        );
        assert_eq!(
            max_rel_err(&[f32::INFINITY], &[f32::NEG_INFINITY]),
            f64::INFINITY,
            "异号无穷"
        );
    }

    #[test]
    fn max_rel_err_matches_nan_and_same_infinity() {
        assert_eq!(max_rel_err(&[f32::NAN], &[f32::NAN]), 0.0);
        assert_eq!(max_rel_err(&[f32::INFINITY], &[f32::INFINITY]), 0.0);
        assert_eq!(max_rel_err(&[f32::NEG_INFINITY], &[f32::NEG_INFINITY]), 0.0);
    }

    #[test]
    fn max_rel_err_falls_back_to_absolute_when_reference_is_zero() {
        // 参考全零：相对误差分母为零，退化为绝对误差，而不是误报「对不上」。
        assert_eq!(max_rel_err(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        assert_eq!(max_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 2.0);
    }

    #[test]
    fn rel_err_locates_the_worst_element() {
        let report = rel_err(&[1.0, 2.1, 3.0], &[1.0, 2.0, 3.0]);
        assert!(!report.mismatched);
        assert_eq!(report.scale, 3.0);
        let worst = report.worst.expect("0.1 的偏差落在第二个元素");
        assert_eq!((worst.index, worst.got, worst.want), (1, 2.1, 2.0));
        assert_eq!(
            report.ratio(),
            max_rel_err(&[1.0, 2.1, 3.0], &[1.0, 2.0, 3.0])
        );
        assert!(
            (report.ratio() - 0.1 / 3.0).abs() < 1e-6,
            "{}",
            report.ratio()
        );
    }

    #[test]
    fn rel_err_is_zero_and_unlocated_when_the_batches_match() {
        let report = rel_err(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!((report.err, report.mismatched), (0.0, false));
        assert_eq!(report.worst, None, "没有偏差就没有最差点");
        assert_eq!(rel_err(&[], &[]).ratio(), 0.0, "空批视为对上");
    }

    #[test]
    fn rel_err_flags_length_and_nonfinite_breaks() {
        let short = rel_err(&[1.0], &[1.0, 1.0]);
        assert!(short.mismatched);
        assert_eq!(short.ratio(), f64::INFINITY);
        assert_eq!(short.worst, None, "长度不同没有可比的下标");

        let nan = rel_err(&[1.0, f32::NAN], &[1.0, 2.0]);
        assert!(nan.mismatched);
        let at = nan.worst.expect("NaN 与有限值不一致落在第二个元素");
        assert_eq!((at.index, at.want), (1, 2.0));
        assert!(at.got.is_nan(), "{}", at.got);

        assert!(
            !rel_err(&[f32::INFINITY], &[f32::INFINITY]).mismatched,
            "同号无穷算对上"
        );
    }

    #[test]
    fn assert_within_rel_passes_inside_the_tolerance() {
        assert_within_rel(&[1.0, 2.0], &[1.0, 2.0], 0.0, "同一批数");
        assert_within_rel(&[1.0, 2.0], &[1.0, 2.000_004], 1e-5, "误差在阈值内");
    }

    #[test]
    #[should_panic(expected = "第 1 个元素 got=2.5 want=2")]
    fn assert_within_rel_names_the_offending_element() {
        assert_within_rel(&[1.0, 2.5], &[1.0, 2.0], 1e-3, "GEMM");
    }

    #[test]
    #[should_panic(expected = "长度不同")]
    fn assert_within_rel_reports_a_length_mismatch() {
        assert_within_rel(&[1.0], &[1.0, 2.0], 1e-3, "行批");
    }

    #[test]
    #[should_panic(expected = "第 0 个元素非有限值不一致")]
    fn assert_within_rel_reports_a_nonfinite_break() {
        assert_within_rel(&[f32::NAN], &[1.0], 1e-3, "注意力");
    }

    #[test]
    fn finite_predicate_rejects_nan_and_infinity() {
        assert!(all_finite(&[1.0, -2.0, 0.0]));
        assert!(!all_finite(&[f32::NAN]));
        assert!(!all_finite(&[f32::INFINITY]));
    }

    #[test]
    fn cosine_ranges_from_minus_one_to_one() {
        let same = [1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&same, &same), Some(1.0));
        assert_eq!(cosine_similarity(&same, &[-1.0, -2.0, -3.0]), Some(-1.0));
        let orthogonal = [1.0, 0.0];
        let other = [0.0, 1.0];
        assert_eq!(cosine_similarity(&orthogonal, &other), Some(0.0));
    }

    #[test]
    fn cosine_is_none_on_mismatched_or_zero_vectors() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 1.0]), None, "长度不同");
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), None, "零向量");
        assert_eq!(
            cosine_similarity(&[1.0, 2.0], &[0.0, 0.0]),
            None,
            "另一向量为零"
        );
    }

    #[test]
    fn cosine_is_scale_invariant() {
        let left = [1.0, 2.0, 3.0];
        let right = [2.0, 4.0, 6.0];
        assert_eq!(
            cosine_similarity(&left, &right),
            Some(1.0),
            "同向缩放不改方向"
        );
    }
}
}
