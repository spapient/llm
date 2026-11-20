//! 数值对拍口径：逐元素绝对误差、绝对+相对双项容差、全局相对误差、有限值谓词与余弦方向一致性。
//!
//! 判据分四档，不要混用：同尺度逐元素对拍用 [`approx_eq`]；量化 logits 的方向一致性用
//! [`cosine_similarity`] + [`COSINE_GUARD`]；解析梯度 vs 数值梯度这类随量级缩放的比对，
//! 走 [`close`] 的「绝对 + 相对」双项预算，不套绝对容差；内核 vs 标量参考的**整批**对拍走
//! [`max_rel_err`]（与 `cutile::reference` 同一口径）——它只有一个全局分母，答的是「整批数
//! 合在一起差多少」，不是逐元素各算各的。

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

/// 全局相对误差：`max|got − want| / max|want|`，与 `cutile::reference::max_rel_err` 同一口径。
///
/// 与 [`close`] 的逐元素预算不同，这里只有**一个**分母（`want` 的最大绝对值），所以它答的是
/// 「整批数合在一起差多少」这一个数，而不是逐元素各算各的——这正是内核对拍要的那条判据。
/// 放在夹具层是为了让跨层测试不必为了这一个数字去依赖 `cutile`。
///
/// 长度不等、或两侧非有限值不一致（一边 NaN 一边不是、一边有限一边无穷）返回
/// [`f64::INFINITY`]：那是「不是同一批数」，不是「误差很大」，两者在报告里必须能分开。
/// `want` 全零时退化为 `max|got|`（绝对误差）：分母为零的相对误差没有定义，报无穷会把
/// 「参考恰为零」误读成「对不上」。
#[must_use]
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    // 非有限值不能直接进下面的累加：NaN 会被 `max` 当缺席，漏报成「对上了」。逐对先定性，
    // 同是 NaN、同号 Inf 算对上，其余非有限组合一律判不一致。
    if !got.iter().zip(want).all(|(&g, &w)| nonfinite_match(g, w)) {
        return f64::INFINITY;
    }
    let max_want = want
        .iter()
        .fold(0.0f64, |acc, value| acc.max(f64::from(*value).abs()));
    let max_err = got.iter().zip(want).fold(0.0f64, |acc, (&g, &w)| {
        acc.max((f64::from(g) - f64::from(w)).abs())
    });
    if max_want == 0.0 {
        return max_err;
    }
    max_err / max_want
}

/// 非有限值配对是否一致；有限值交给数值误差比较。
fn nonfinite_match(got: f32, want: f32) -> bool {
    match (got.is_nan(), want.is_nan()) {
        (true, true) => true,
        (true, false) | (false, true) => false,
        // 非 NaN：同号 Inf 相等，或两边都是有限值；其余（Inf 对有限、异号 Inf）不一致。
        _ => got == want || (!got.is_infinite() && !want.is_infinite()),
    }
}

/// 全局相对误差是否在阈值内：`max_rel_err(got, want) <= tolerance`。
///
/// [`max_rel_err`] 的布尔投影，与 `cutile::reference::within` 对应，但不绑定具体数据类型——
/// 阈值由调用方按精度自定。
#[must_use]
pub fn within_rel(got: &[f32], want: &[f32], tolerance: f32) -> bool {
    max_rel_err(got, want) <= f64::from(tolerance)
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
