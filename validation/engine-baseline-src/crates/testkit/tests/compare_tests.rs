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

/// `close` 的相对项分母是 `max(|got|,|want|)`，`assert_close` 是 `|want|`：前者放行的
/// 元素后者必须拒绝。两者若被人"顺手统一"，这条会立刻红——那是在悄悄改容差语义。
#[test]
#[should_panic(expected = "容差")]
fn the_asymmetric_relative_budget_is_stricter_than_the_symmetric_one() {
    let (got, want, abs, rel) = (1.5f32, 1.0f32, 0.0f32, 0.4f32);
    assert!(
        close(&[got], &[want], abs, rel),
        "对称分母放行：0.5 ≤ 0.4·1.5"
    );
    assert_close(&[got], &[want], abs, rel);
}
