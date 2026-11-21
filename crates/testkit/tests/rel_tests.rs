use super::*;

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
