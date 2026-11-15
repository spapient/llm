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
fn gating_accepts_all_lists_and_ignores_blank_entries() {
    assert!(matches("all", "nvidia"));
    assert!(matches("cpu, nvidia", "nvidia"));
    assert!(matches(" cpu ", "cpu"));
    assert!(!matches("", "cpu"));
    assert!(!matches("cpu", "nvidia"));
}

#[test]
fn the_same_tag_reuses_one_directory() {
    let first = scratch_dir("a");
    let second = scratch_dir("a");
    assert!(first.is_dir());
    assert_eq!(first, second, "同标签同进程应复用同一目录");
    assert_ne!(first, scratch_dir("b"), "不同标签必须各自一个目录");
    let _ = std::fs::remove_dir_all(&first);
    let _ = std::fs::remove_dir_all(scratch_dir("b"));
}

#[test]
fn rng_is_deterministic_and_bounded() {
    let mut left = Rng::new(1234);
    let mut right = Rng::new(1234);
    for _ in 0..16 {
        let value = left.below(10);
        assert_eq!(value, right.below(10), "同种子序列一致");
        assert!(value < 10, "结果落在界内");
    }
    assert_eq!(Rng::new(7).below(1), 0, "bound=1 恒为 0");
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

#[test]
fn skip_registers_reason_and_count() {
    let before = skip_count();
    assert!(skip("测试用临时跳过原因"));
    assert!(skip_count() > before, "跳过计数必须单调递增");
    assert!(
        skipped_reasons()
            .iter()
            .any(|reason| reason == "测试用临时跳过原因")
    );
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
fn rng_value_stays_away_from_zero_and_is_signable() {
    let mut rng = Rng::new(42);
    for _ in 0..16 {
        let value = rng.value();
        assert!(value.abs() >= 0.5 && value.abs() <= 2.0);
    }
}

#[test]
fn rng_next_f32_is_in_unit_interval() {
    let mut rng = Rng::new(7);
    for _ in 0..16 {
        let value = rng.next_f32();
        assert!((0.0..1.0).contains(&value));
    }
}

#[test]
fn device_or_skip_registers_when_disabled() {
    let before = skip_count();
    assert!(device_or_skip("nvidia", "测试：未启用 nvidia"));
    assert!(skip_count() > before, "跳过计数必须单调递增");
}
