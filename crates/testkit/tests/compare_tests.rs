//! 对拍判据自测：逐元素容差、方向、整批/单位分母与生产参考实现交叉对拍。

use super::*;
use cutile::atom::DType;
use cutile::reference;

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
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
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
    assert_eq!(
        cosine_similarity(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]),
        Some(1.0)
    );
}

/// 对称分母与参考侧分母不可互换；这条钉住二者的语义差异。
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

#[test]
fn max_rel_err_uses_a_single_global_denominator() {
    assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((err - 0.05).abs() < 1e-6, "{err}");
    assert!(within_rel(&[1.0, 2.0], &[1.0, 2.0], 1e-5));
    assert!(!within_rel(&[1.0, 2.5], &[1.0, 2.0], 1e-5));
}

#[test]
fn max_rel_err_is_infinity_on_mismatched_lengths_or_nonfinite_mismatch() {
    assert_eq!(max_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY);
    assert_eq!(max_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
    assert_eq!(max_rel_err(&[f32::INFINITY], &[1.0]), f64::INFINITY);
    assert_eq!(
        max_rel_err(&[f32::INFINITY], &[f32::NEG_INFINITY]),
        f64::INFINITY
    );
}

#[test]
fn max_rel_err_matches_nan_and_same_infinity() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let pair = [value];
        assert_eq!(max_rel_err(&pair, &pair), 0.0);
        assert_eq!(mean_rel_err(&pair, &pair), 0.0, "成对非有限值不得污染均值");
        assert_eq!(
            rel_gate(&pair, &pair),
            RelGate {
                mean: 0.0,
                max: 0.0
            }
        );
    }
}

#[test]
fn max_rel_err_falls_back_to_absolute_when_reference_is_zero() {
    assert_eq!(max_rel_err(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    assert_eq!(max_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 2.0);
}

#[test]
fn rel_err_locates_the_worst_element() {
    let report = rel_err(&[1.0, 2.1, 3.0], &[1.0, 2.0, 3.0]);
    assert!(!report.mismatched);
    assert_eq!(report.scale, 3.0);
    let worst = report.worst.expect("0.1 的偏差落在第二个元素");
    assert_eq!((worst.idx, worst.got, worst.want), (1, 2.1, 2.0));
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
    assert_eq!(report.worst, None);
    assert_eq!(rel_err(&[], &[]).ratio(), 0.0, "空批视为对上");
}

#[test]
fn rel_err_flags_length_and_nonfinite_breaks() {
    let short = rel_err(&[1.0], &[1.0, 1.0]);
    assert!(short.mismatched);
    assert_eq!(short.ratio(), f64::INFINITY);
    assert_eq!(short.worst, None);
    let nan = rel_err(&[1.0, f32::NAN], &[1.0, 2.0]);
    assert!(nan.mismatched);
    let at = nan.worst.expect("NaN 与有限值不一致落在第二个元素");
    assert_eq!((at.idx, at.want), (1, 2.0));
    assert!(at.got.is_nan());
    assert!(!rel_err(&[f32::INFINITY], &[f32::INFINITY]).mismatched);
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

/// 覆盖空批、全零参考、跨 8 车道边界、次正规、溢出与三类非有限值。
fn samples() -> Vec<Vec<f32>> {
    vec![
        vec![],
        vec![0.0],
        vec![0.0, 0.0],
        vec![1.0, -2.0, 3.0],
        vec![1e-30, 3.0],
        vec![f32::MAX, 1.0],
        vec![f32::NAN, 1.0],
        vec![f32::INFINITY, 1.0],
        vec![f32::NEG_INFINITY, f32::INFINITY],
        (0..9).map(|i| i as f32 * 0.25 - 1.0).collect(),
    ]
}

/// 同一份样本矩阵驱动所有交叉对拍，避免每条判据各建一份语料。
fn for_each_sample(mut check: impl FnMut(&[f32], &[f32], usize, usize)) {
    let samples = samples();
    for (left, got) in samples.iter().enumerate() {
        for (right, want) in samples.iter().enumerate() {
            check(got, want, left, right);
        }
    }
}

#[test]
fn the_two_oracles_agree_bit_for_bit() {
    for_each_sample(|got, want, left, right| {
        assert_eq!(
            max_rel_err(got, want).to_bits(),
            reference::max_rel_err(got, want).to_bits(),
            "样本 {left}×{right} 的判据读数不同"
        );
        assert_eq!(
            rel_err(got, want).ratio().to_bits(),
            max_rel_err(got, want).to_bits()
        );
    });
}

#[test]
fn the_tolerance_ladder_is_shared_not_copied() {
    for_each_sample(|got, want, left, right| {
        for dtype in [DType::F32, DType::F64, DType::I8, DType::I4] {
            assert_eq!(
                within_ratio(got, want, reference::tolerance(dtype)),
                reference::within(got, want, dtype),
                "{dtype:?} 分档下样本 {left}×{right} 的判定不同"
            );
        }
    });
}

#[test]
fn the_worst_element_attains_the_reported_error() {
    let report = rel_err(&[1.0, 2.5, -3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
    let at = report.worst.expect("有偏差就有最差点");
    assert_eq!(at.idx, 2);
    assert_eq!((at.got, at.want), (-3.0, 3.0));
    assert_eq!(report.err, f64::from((at.got - at.want).abs()));
    assert_eq!(report.scale, 4.0);
}

#[test]
fn the_two_denominators_differ_only_in_the_floor() {
    let (got, want) = ([0.00011, 0.0002], [0.0001, 0.0002]);
    let batch = rel_err(&got, &want);
    let unit = rel_err_with(&got, &want, Scale::Unit);
    assert_eq!(batch.worst.map(|worst| worst.idx), Some(0));
    assert_eq!(unit.worst.map(|worst| worst.idx), Some(0));
    assert_eq!(batch.scale, f64::from(0.0002f32));
    assert_eq!(unit.scale, 1.0);
    assert_eq!(unit.ratio(), f64::from((0.00011f32 - 0.0001f32).abs()));
    assert!(batch.ratio() > unit.ratio() * 100.0);
    let (got, want) = ([1.5, 100.0], [1.0, 100.0]);
    assert!(rel_err_with(&got, &want, Scale::Unit).ratio() > rel_err(&got, &want).ratio() * 10.0);
}

#[test]
fn mean_rel_err_is_zero_on_matching_batches() {
    assert_eq!(mean_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert_eq!(mean_rel_err(&[], &[]), 0.0, "空批视为对上");
}

#[test]
fn mean_rel_err_averages_per_element_on_the_same_denominator() {
    let err = mean_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((err - 0.025).abs() < 1e-6, "{err}");
    let max = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((max - 0.05).abs() < 1e-6, "max 门仍是最差元素：{max}");
}

#[test]
fn mean_and_max_separate_whole_batch_drift_from_an_isolated_outlier() {
    let mean = mean_rel_err(&[1.0, 2.5], &[1.0, 2.0]);
    let max = max_rel_err(&[1.0, 2.5], &[1.0, 2.0]);
    assert!((mean - 0.125).abs() < 1e-6, "{mean}");
    assert!((max - 0.25).abs() < 1e-6, "{max}");
    assert!(max > mean);
}

#[test]
fn mean_rel_err_follows_the_unit_denominator_floor() {
    let mean = mean_rel_err_with(&[0.00011, 0.0002], &[0.0001, 0.0002], Scale::Unit);
    assert!((mean - 0.000005).abs() < 1e-9, "{mean}");
}

#[test]
fn mean_rel_err_is_infinity_on_mismatch_and_falls_back_to_absolute_on_zero_reference() {
    assert_eq!(mean_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY);
    assert_eq!(mean_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
    assert_eq!(mean_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 1.0);
}

#[test]
fn rel_gate_reports_mean_and_max_from_one_pass() {
    let gate = rel_gate(&[1.0, 2.5, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
    assert!((gate.mean - 0.03125).abs() < 1e-9, "{}", gate.mean);
    assert!((gate.max - 0.125).abs() < 1e-9, "{}", gate.max);
}

#[test]
fn rel_gate_separates_a_low_mean_from_a_loose_max_gate() {
    let gate = rel_gate(&[1.0, 2.5, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
    assert!(gate.passes(0.05, 0.2));
    assert!(!gate.passes(0.05, 0.1));
}

#[test]
fn rel_gate_fails_when_mean_drifts_across_the_strict_gate() {
    let gate = rel_gate(&[1.1, 2.1, 3.1, 4.1], &[1.0, 2.0, 3.0, 4.0]);
    assert!(gate.max <= 0.03);
    assert!(!gate.passes(0.02, 0.03));
}

#[test]
fn rel_gate_is_infinity_on_mismatched_lengths() {
    let gate = rel_gate(&[1.0], &[1.0, 1.0]);
    assert_eq!((gate.mean, gate.max), (f64::INFINITY, f64::INFINITY));
    assert!(!gate.passes(1.0, 1.0));
}

#[test]
#[should_panic(expected = "双门对拍失败")]
fn assert_rel_gate_panics_when_either_gate_fails() {
    assert_rel_gate(
        &[1.0, 2.5, 3.0, 4.0],
        &[1.0, 2.0, 3.0, 4.0],
        0.05,
        0.1,
        "GEMM 双门",
    );
}
