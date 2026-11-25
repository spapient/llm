//! 相对误差判据的自测：全局分母、最差元素定位、阈值投影，以及两条分母策略与
//! `cutile::reference::tolerance` 的交叉对拍。

use super::*;
use cutile::atom::DType;
use cutile::reference;

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
    assert_eq!((at.idx, at.want), (1, 2.0));
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

// 以下四条原在 `oracle.rs`：「对拍口径的唯一性」门禁——夹具层的读数与生产侧
// `cutile::reference` 必须是同一个判据。它们与上面几条测的是同一个概念（对拍判据），
// 所以合住一份，不为四条用例单独开一个 `src/` 模块。

/// 参与交叉对拍的样本：覆盖空批、全零参考、跨 8 车道边界、次正规、溢出与三类非有限值。
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

#[test]
fn the_two_oracles_agree_bit_for_bit() {
    let samples = samples();
    for (left, got) in samples.iter().enumerate() {
        for (right, want) in samples.iter().enumerate() {
            assert_eq!(
                max_rel_err(got, want).to_bits(),
                reference::max_rel_err(got, want).to_bits(),
                "样本 {left}×{right} 的判据读数不同"
            );
            assert_eq!(
                rel_err(got, want).ratio().to_bits(),
                max_rel_err(got, want).to_bits(),
                "样本 {left}×{right}：读数与投影不是同一个数"
            );
        }
    }
}

#[test]
fn the_tolerance_ladder_is_shared_not_copied() {
    let samples = samples();
    for (left, got) in samples.iter().enumerate() {
        for (right, want) in samples.iter().enumerate() {
            for dtype in [DType::F32, DType::F64, DType::I8, DType::I4] {
                assert_eq!(
                    within_ratio(got, want, reference::tolerance(dtype)),
                    reference::within(got, want, dtype),
                    "{dtype:?} 分档下样本 {left}×{right} 的判定不同"
                );
            }
        }
    }
}

#[test]
fn the_worst_element_attains_the_reported_error() {
    let got = [1.0, 2.5, -3.0, 4.0];
    let want = [1.0, 2.0, 3.0, 4.0];
    let report = rel_err(&got, &want);
    let at = report.worst.expect("有偏差就有最差点");
    assert_eq!(at.idx, 2, "−3 与 3 相差 6，是整批最大绝对误差");
    assert_eq!((at.got, at.want), (-3.0, 3.0));
    let attained = f64::from((at.got - at.want).abs());
    assert_eq!(report.err, attained, "读数必须是这个元素上的绝对误差");
    assert_eq!(report.scale, 4.0, "分母是整批最大的参考幅度");
}

/// 两条分母策略相差的**只有分母**：幅度远小于 1 的批次里单位口径给 1 的绝对地板，幅度大于 1 时
/// 单位口径反而更严。这条用例把「两种用法不是松紧之分」钉住：引擎的真机基准（`metal_tests`）
/// 用 [`Scale::Unit`]（GEMM 与注意力输出的量级差四个数量级），其余门禁与这里的交叉对拍用整批口径。
#[test]
fn the_two_denominators_differ_only_in_the_floor() {
    let got = [0.00011, 0.0002];
    let want = [0.0001, 0.0002];
    let batch = rel_err(&got, &want);
    let unit = rel_err_with(&got, &want, Scale::Unit);
    assert_eq!(batch.worst.map(|worst| worst.idx), Some(0));
    assert_eq!(unit.worst.map(|worst| worst.idx), Some(0));
    assert_eq!(
        batch.scale,
        f64::from(0.0002f32),
        "整批口径的分母是最大幅度"
    );
    assert_eq!(unit.scale, 1.0, "单位口径的分母在该元素上是地板 1");
    assert_eq!(unit.ratio(), f64::from((0.00011f32 - 0.0001f32).abs()));
    assert!(
        batch.ratio() > unit.ratio() * 100.0,
        "参考整体小于 1 时，整批口径把同一个绝对差放大成两个数量级以上"
    );

    let got = [1.5, 100.0];
    let want = [1.0, 100.0];
    assert!(
        rel_err_with(&got, &want, Scale::Unit).ratio() > rel_err(&got, &want).ratio() * 10.0,
        "参考幅度大于 1 时单位口径更严：两条是不同的问题，不是松紧两档"
    );
}

#[test]
fn mean_rel_err_is_zero_on_matching_batches() {
    assert_eq!(mean_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert_eq!(mean_rel_err(&[], &[]), 0.0, "空批视为对上");
}

#[test]
fn mean_rel_err_averages_per_element_on_the_same_denominator() {
    // Batch 分母 = max|want| = 2.0：elem0 无偏差，elem1 相对差 0.1/2.0 = 0.05。
    let err = mean_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((err - 0.025).abs() < 1e-6, "{err}");
    let max = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((max - 0.05).abs() < 1e-6, "max 门仍是最差元素：{max}");
}

#[test]
fn mean_and_max_separate_whole_batch_drift_from_an_isolated_outlier() {
    // 单点离群：max 被该点顶高，mean 被其余零偏差拉低，两门必须给出不同读数。
    let mean = mean_rel_err(&[1.0, 2.5], &[1.0, 2.0]);
    let max = max_rel_err(&[1.0, 2.5], &[1.0, 2.0]);
    assert!((mean - 0.125).abs() < 1e-6, "{mean}");
    assert!((max - 0.25).abs() < 1e-6, "{max}");
    assert!(max > mean, "离群门应高于平均门");
}

#[test]
fn mean_rel_err_follows_the_unit_denominator_floor() {
    let got = [0.00011, 0.0002];
    let want = [0.0001, 0.0002];
    let mean = mean_rel_err_with(&got, &want, Scale::Unit);
    assert!((mean - 0.000005).abs() < 1e-9, "{mean}");
}

#[test]
fn mean_rel_err_is_infinity_on_mismatch_and_falls_back_to_absolute_on_zero_reference() {
    assert_eq!(mean_rel_err(&[1.0], &[1.0, 1.0]), f64::INFINITY, "长度不同");
    assert_eq!(
        mean_rel_err(&[f32::NAN], &[0.0]),
        f64::INFINITY,
        "NaN 对有限"
    );
    // 参考全零：退化为绝对误差的平均，不套相对误差。
    assert_eq!(mean_rel_err(&[0.0, 2.0], &[0.0, 0.0]), 1.0);
}
