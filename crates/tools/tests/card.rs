//! 实测卡的集成测试：三态判定、基准记录、回归基线、统计区间、JSON 转义。
//! 只用公开面，不读实现即可换掉实现。

use diagnostic::{Code, Result};
use llm_tools::bench::{BenchRecord, Bottleneck, EntryKind, Geometry, Harness, PeakProfile};
use llm_tools::card::{Card, Direction, Protocol, Shortfall, Standing, Statistic, Tally};
use llm_tools::escape::quote;
use llm_tools::regress::{
    Baseline, Case, Direction as RegDir, Estimate, Judgement, OpenReason, SelfCheck, self_check,
};
use llm_tools::stats::{interval, mean, quantile, resample, sd};

/// 一台在闭集里的 A800：多数用例只关心门槛，不关心装置。
fn harness(host: &str) -> Harness {
    Harness::new(host, "cpu", "os", "aarch64", "rustc", "driver")
}

fn jitter() -> Protocol {
    Protocol::of(
        "E10",
        "decode_step_ns",
        Statistic::Cv,
        Direction::Lower,
        0.05,
    )
    .expect("协议")
}

testkit::test_cases! {
/// 同一协议下抖动为 0 的一列：区间退化成一个点，整段落在门槛内侧。
fn a_steady_series_passes_the_jitter_gate() {
    let card = Card::judge(jitter(), harness("A800"), &[10.0; 40]).expect("判");
    assert_eq!(card.standing(), Standing::Pass);
    let tally = card.tally().expect("有摘要");
    assert_eq!((tally.lower(), tally.upper()), (0.0, 0.0));
}

/// 抖动远超门槛的一列：区间整段越界，这一格记 FAIL 而不是 OPEN。
fn a_series_beyond_the_gate_fails() {
    let mut samples = Vec::new();
    for _ in 0..20 {
        samples.extend([1.0, 100.0]);
    }
    let card = Card::judge(jitter(), harness("A800"), &samples).expect("判");
    assert_eq!(card.standing(), Standing::Fail);
    assert!(
        card.tally().expect("有摘要").upper() > 0.05,
        "区间上界越过门槛"
    );
}

/// 区间跨着门槛：样本还不足以把「过」与「不过」分开，只能记 OPEN。
fn an_interval_across_the_gate_stays_open() {
    let protocol = Protocol::of(
        "E1",
        "tokens_per_s",
        Statistic::Mean,
        Direction::Higher,
        2.0,
    )
    .expect("协议")
    .with_least(3)
    .expect("下限");
    let card = Card::judge(protocol, harness("A800"), &[1.0, 2.0, 3.0]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Straddle));
    assert!(card.tally().expect("有摘要").lower() < 2.0);
    assert!(card.tally().expect("有摘要").upper() > 2.0);
}

/// 样本不够时不判：摘要照记，结论只有一个 OPEN 理由，不写「达标」也不写「未达标」。
fn too_few_samples_are_not_judged_at_all() {
    let card = Card::judge(jitter(), harness("A800"), &[10.0, 12.0, 11.0]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
    assert_eq!(card.tally().expect("摘要照记").count(), 3);
    assert_eq!(card.standing().shortfall(), Some(Shortfall::Samples));
    assert_ne!(card.standing(), Standing::Pass);
}

/// 闭集外的主机：样本再多也不是这份协议的证据。
fn a_host_outside_the_closure_cannot_pass() {
    let protocol = jitter().with_hosts(&["A800", "RTX_3090", "T600"]);
    let card = Card::judge(protocol, harness("macos-aarch64"), &[10.0; 40]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Host));
}

/// 闭集为空表示不设闭集：任何装置都按门槛判。
fn an_empty_closure_admits_every_host() {
    let card = Card::judge(jitter(), harness("macos-aarch64"), &[10.0; 40]).expect("判");
    assert_eq!(card.standing(), Standing::Pass);
}

/// 均值为 0 的列上变异系数没有定义：填 0 会让「完全抖动」看起来像「完全稳定」。
fn a_zero_mean_series_has_no_jitter_to_report() {
    let mut samples = Vec::new();
    for _ in 0..20 {
        samples.extend([-1.0, 1.0]);
    }
    let card = Card::judge(jitter(), harness("A800"), &samples).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Degenerate));
    assert!(card.tally().is_none(), "无定义就不写摘要");
}

/// 空白列：一个样本都没有，既不摘要也不判。
fn no_samples_report_no_summary() {
    let card = Card::judge(jitter(), harness("A800"), &[]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
    assert!(card.tally().is_none());
}

/// 非有限样本是采集坏了：报错，不许静默跳过。
fn a_broken_sample_is_reported_as_such() {
    let err = Card::judge(jitter(), harness("A800"), &[1.0, f64::NAN]).unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.message().contains("采集坏了"), "{}", err.message());
}

/// 重采样种子固定：同一份样本只会得到一份卡片，复核时能逐位对拍。
fn the_same_samples_produce_the_same_interval() {
    let samples = [9.0, 11.0, 10.5, 9.4, 12.0, 10.1, 9.9, 10.8];
    let first = Card::judge(jitter(), harness("A800"), &samples).expect("判");
    let second = Card::judge(jitter(), harness("A800"), &samples).expect("判");
    assert_eq!(
        first.tally().map(|tally| (tally.lower(), tally.upper())),
        second.tally().map(|tally| (tally.lower(), tally.upper()))
    );
}

/// 分位不许交叉：中位 ≤ 95 ≤ 99，区间下界 ≤ 上界。
fn the_summary_never_contradicts_itself() {
    let samples: Vec<f64> = (1..=100).map(f64::from).collect();
    let tally = Tally::of(&samples, Statistic::Mean)
        .expect("摘要")
        .expect("有定义");
    assert!(tally.lower() <= tally.upper(), "区间不许倒挂");
    assert!(tally.sd() >= 0.0, "标准差非负");
    assert_eq!(tally.count(), 100);
}

/// 归档里结论与理由必须相邻：读者不该靠推断知道这次是 OPEN 还是 FAIL。
fn the_archive_keeps_the_reason_next_to_the_verdict() {
    let open = Card::judge(jitter(), harness("A800"), &[10.0])
        .expect("判")
        .to_json();
    assert!(open.contains("\"standing\": \"open\""), "{open}");
    assert!(open.contains("\"shortfall\": \"samples\""), "{open}");
    assert!(open.contains("\"statistic\": \"cv\""), "{open}");
    assert!(open.contains("\"host\": \"A800\""), "{open}");

    let mut zero_mean = Vec::new();
    for _ in 0..15 {
        zero_mean.extend([-1.0, 1.0]);
    }
    let degenerate = Card::judge(jitter(), harness("A800"), &zero_mean)
        .expect("判")
        .to_json();
    assert!(degenerate.contains("\"tally\": null"), "{degenerate}");
    assert!(
        degenerate.contains("\"shortfall\": \"degenerate\""),
        "{degenerate}"
    );
}

/// 短名是归档口径：改措辞属于破坏性改动。
fn the_short_names_are_stable() {
    assert_eq!(Statistic::Mean.as_str(), "mean");
    assert_eq!(Statistic::Cv.as_str(), "cv");
    assert_eq!(Shortfall::Host.as_str(), "host");
    assert_eq!(Shortfall::Samples.as_str(), "samples");
    assert_eq!(Shortfall::Degenerate.as_str(), "degenerate");
    assert_eq!(Shortfall::Straddle.as_str(), "straddle");
    assert_eq!(Standing::Pass.as_str(), "pass");
    assert_eq!(Standing::Fail.as_str(), "fail");
    assert_eq!(Standing::Open(Shortfall::Host).as_str(), "open");
    assert_eq!(
        Standing::Open(Shortfall::Host).shortfall(),
        Some(Shortfall::Host)
    );
}

/// 构造期拒绝不可能成立的协议：空名、非有限门槛、下限 <2、置信水平越界。
fn a_protocol_rejects_impossible_settings() {
    let empty = Protocol::of("", "ns", Statistic::Mean, Direction::Lower, 1.0).unwrap_err();
    assert_eq!(empty.code(), Code::Invalid);
    let nan = Protocol::of("E1", "ns", Statistic::Mean, Direction::Lower, f64::NAN).unwrap_err();
    assert_eq!(nan.code(), Code::Invalid);
    assert!(nan.message().contains("有限数"), "{}", nan.message());
    assert_eq!(jitter().with_least(1).unwrap_err().code(), Code::Invalid);
}
}

// ---------------------------------------------------------------------------
// bench：基准记录的构造与不变式
// ---------------------------------------------------------------------------

fn bench_harness() -> Harness {
    Harness::new(
        "host-a",
        "cpu-x",
        "macos",
        "aarch64",
        "rustc 1.82",
        "cpu-reference",
    )
}

fn bench_geometry(shape: Vec<usize>, batch: u32, dtype: &str) -> Result<Geometry> {
    Geometry::new(shape, batch, dtype)
}

fn bench_record(name: &str, backend: &str, geom: Geometry) -> Result<BenchRecord> {
    BenchRecord::new(
        EntryKind::Op,
        name,
        backend,
        geom,
        bench_harness(),
        PeakProfile::default(),
    )
}

testkit::test_cases! {
fn a_minimal_record_keeps_observed_and_roofline_optional() {
    let geom = bench_geometry(vec![4096, 4096], 1, "f32").expect("几何合法");
    let record = bench_record("gemv_f32", "cpu-reference", geom).expect("记录合法");
    assert_eq!(record.kind(), EntryKind::Op);
    assert_eq!(record.name(), "gemv_f32");
    assert_eq!(record.backend(), "cpu-reference");
    assert_eq!(record.geometry().shape(), &[4096, 4096]);
    assert_eq!(record.geometry().batch(), 1);
    assert_eq!(record.geometry().dtype(), "f32");
    assert!(record.observed().is_none(), "未测量不预填实测");
    assert!(record.peak_profile().peak_bw_gb_s().is_none());
    assert!(record.peak_profile().peak_gflops().is_none());
}

fn observed_is_filled_only_when_measured() {
    let geom = bench_geometry(vec![4096], 1, "f32").expect("几何合法");
    let filled = bench_record("rope_f32", "cpu-reference", geom)
        .expect("记录合法")
        .with_observed(12.5)
        .expect("实测有限");
    assert_eq!(filled.observed(), Some(12.5));

    let geom = bench_geometry(vec![4096], 1, "f32").expect("几何合法");
    let nan = bench_record("rope_f32", "cpu-reference", geom)
        .expect("记录合法")
        .with_observed(f64::NAN);
    assert!(nan.is_err(), "非有限实测不能回填");
    assert_eq!(nan.unwrap_err().code(), Code::Invalid);
}

fn empty_identity_and_invalid_geometry_are_rejected() {
    let cases: [(&str, &str, Result<Geometry>); 5] = [
        ("", "cpu-reference", bench_geometry(vec![4], 1, "f32")),
        ("gemv", "", bench_geometry(vec![4], 1, "f32")),
        ("gemv", "cpu-reference", bench_geometry(vec![], 1, "f32")),
        ("gemv", "cpu-reference", bench_geometry(vec![4], 0, "f32")),
        ("gemv", "cpu-reference", bench_geometry(vec![4], 1, "")),
    ];
    for (name, backend, geom) in cases {
        let geom = match geom {
            Ok(geom) => geom,
            Err(_) => continue,
        };
        let err = bench_record(name, backend, geom).expect_err("必须拒绝");
        assert_eq!(err.code(), Code::Invalid, "输入：{name}/{backend}");
    }
}

fn invalid_geometry_itself_is_an_error() {
    assert!(bench_geometry(vec![], 1, "f32").is_err());
    assert!(bench_geometry(vec![4], 0, "f32").is_err());
    assert!(bench_geometry(vec![4], 1, "").is_err());
}

fn roofline_rejects_negative_or_nonfinite_peaks() {
    assert!(PeakProfile::new(Some(-1.0), None).is_err());
    assert!(PeakProfile::new(None, Some(f64::INFINITY)).is_err());
    let ok = PeakProfile::new(Some(1024.0), Some(512.0)).expect("上界合法");
    assert_eq!(ok.peak_bw_gb_s(), Some(1024.0));
    assert_eq!(ok.peak_gflops(), Some(512.0));
}

fn score_is_bandwidth_bound_below_the_ridge_point() {
    let profile = PeakProfile::new(Some(1000.0), Some(500.0)).expect("上界合法");
    let score = profile.score(0.25, 125.0).expect("打分合法");
    assert_eq!(score.side(), Bottleneck::Bandwidth);
    assert_eq!(score.ceiling_gflops(), 250.0);
    assert_eq!(score.percent(), 50.0);
}

fn score_is_compute_bound_at_or_above_the_ridge_point() {
    let profile = PeakProfile::new(Some(1000.0), Some(500.0)).expect("上界合法");
    let score = profile.score(2.0, 250.0).expect("打分合法");
    assert_eq!(score.side(), Bottleneck::Compute);
    assert_eq!(score.ceiling_gflops(), 500.0);
    assert_eq!(score.percent(), 50.0);
}

fn score_rejects_nonfinite_or_negative_inputs() {
    let profile = PeakProfile::new(Some(1000.0), Some(500.0)).expect("上界合法");
    assert!(profile.score(-0.25, 125.0).is_err());
    assert!(profile.score(f64::NAN, 125.0).is_err());
    assert!(profile.score(0.25, f64::INFINITY).is_err());
}

fn score_requires_both_peaks() {
    assert!(PeakProfile::default().score(0.25, 125.0).is_err());
    let no_flops = PeakProfile::new(Some(1000.0), None).expect("上界合法");
    assert!(no_flops.score(0.25, 125.0).is_err());
    let no_bw = PeakProfile::new(None, Some(500.0)).expect("上界合法");
    assert!(no_bw.score(0.25, 125.0).is_err());
}

fn bottleneck_short_names_are_stable() {
    assert_eq!(Bottleneck::Bandwidth.as_str(), "bandwidth");
    assert_eq!(Bottleneck::Compute.as_str(), "compute");
}

fn entry_kind_short_names_are_stable() {
    assert_eq!(EntryKind::Op.as_str(), "op");
    assert_eq!(EntryKind::Fused.as_str(), "fused");
    assert_eq!(EntryKind::Layer.as_str(), "layer");
    assert_eq!(EntryKind::Model.as_str(), "model");
    assert_eq!(EntryKind::Serving.as_str(), "serving");
}
}

// ---------------------------------------------------------------------------
// regress：回归基线的判定
// ---------------------------------------------------------------------------

fn baseline() -> Baseline {
    Baseline::new(vec![
        Case::new("tput", 100.0, 5.0, RegDir::Higher).expect("场景"),
        Case::new("latency", 50.0, 10.0, RegDir::Lower).expect("场景"),
    ])
    .expect("基线")
}

testkit::test_cases! {
fn a_higher_is_better_case_regresses_only_below_its_limit() {
    let baseline = baseline();
    assert_eq!(baseline.judge("tput", 98.0), Ok(Judgement::Pass));
    assert_eq!(baseline.judge("tput", 100.0), Ok(Judgement::Pass));
    assert!(matches!(
        baseline.judge("tput", 94.0),
        Ok(Judgement::Regressed { .. })
    ));
    assert!(matches!(
        baseline.judge("tput", 120.0),
        Ok(Judgement::Improved { .. })
    ));
}

fn a_lower_is_better_case_mirrors_the_comparison() {
    let baseline = baseline();
    assert_eq!(baseline.judge("latency", 52.0), Ok(Judgement::Pass));
    assert!(matches!(
        baseline.judge("latency", 56.0),
        Ok(Judgement::Regressed { .. })
    ));
    assert!(matches!(
        baseline.judge("latency", 40.0),
        Ok(Judgement::Improved { .. })
    ));
}

fn a_non_finite_measurement_never_passes() {
    assert!(matches!(
        baseline().judge("tput", f64::NAN),
        Ok(Judgement::Regressed { .. })
    ));
}

fn an_unknown_scenario_is_a_configuration_error() {
    let error = baseline().judge("missing", 1.0).expect_err("未知场景");
    assert_eq!(error.code(), Code::Missing);
    assert!(error.to_string().contains("tput"), "{error}");
}

fn impossible_cases_are_rejected() {
    assert_eq!(
        Case::new("x", 0.0, 5.0, RegDir::Higher).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Case::new("x", 1.0, 101.0, RegDir::Higher)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(Baseline::new(Vec::new()).unwrap_err().code(), Code::Invalid);
    let duplicated = Case::new("x", 1.0, 1.0, RegDir::Higher).expect("场景");
    assert_eq!(
        Baseline::new(vec![duplicated.clone(), duplicated])
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

fn judgements_and_directions_have_stable_names() {
    assert_eq!(RegDir::Higher.as_str(), "higher-is-better");
    assert_eq!(RegDir::Lower.as_str(), "lower-is-better");
    assert_eq!(Judgement::Pass.as_str(), "pass");
    assert!(
        Judgement::Improved {
            observed: 2.0,
            baseline: 1.0,
            effect: 1.0
        }
        .is_pass()
    );
}

fn a_higher_is_better_interval_splits_inside_outside_and_straddle() {
    let case = Case::new("tput", 100.0, 5.0, RegDir::Higher).expect("场景");
    assert_eq!(
        case.judge_interval(Estimate::interval(96.0, 97.0, 98.0).expect("区间")),
        Judgement::Pass
    );
    assert!(matches!(
        case.judge_interval(Estimate::interval(90.0, 91.0, 92.0).expect("区间")),
        Judgement::Regressed { .. }
    ));
    assert!(matches!(
        case.judge_interval(Estimate::interval(93.0, 96.0, 97.0).expect("区间")),
        Judgement::Open {
            reason: OpenReason::Straddle,
            ..
        }
    ));
    assert!(matches!(
        case.judge_interval(Estimate::interval(101.0, 102.0, 103.0).expect("区间")),
        Judgement::Improved { .. }
    ));
}

fn a_lower_is_better_interval_mirrors_the_three_states() {
    let case = Case::new("latency", 50.0, 10.0, RegDir::Lower).expect("场景");
    assert_eq!(
        case.judge_interval(Estimate::interval(48.0, 49.0, 52.0).expect("区间")),
        Judgement::Pass
    );
    assert!(matches!(
        case.judge_interval(Estimate::interval(56.0, 57.0, 58.0).expect("区间")),
        Judgement::Regressed { .. }
    ));
    assert!(matches!(
        case.judge_interval(Estimate::interval(52.0, 54.0, 57.0).expect("区间")),
        Judgement::Open {
            reason: OpenReason::Straddle,
            ..
        }
    ));
    assert!(matches!(
        case.judge_interval(Estimate::interval(44.0, 45.0, 46.0).expect("区间")),
        Judgement::Improved { .. }
    ));
}

fn an_open_judgement_carries_no_effect_size() {
    let case = Case::new("tput", 100.0, 5.0, RegDir::Higher).expect("场景");
    let open = case.judge_interval(Estimate::interval(93.0, 96.0, 97.0).expect("区间"));
    assert!(open.is_open());
    assert!(open.effect().is_none());
}

fn drift_within_the_invisible_band_is_not_improved() {
    let case = Case::with_epsilon("tput", 100.0, 5.0, 2.0, RegDir::Higher).expect("场景");
    assert_eq!(case.judge(101.0), Judgement::Pass);
    assert!(matches!(case.judge(103.0), Judgement::Improved { .. }));
}

fn effect_size_is_positive_for_improvement_and_negative_for_regression() {
    let higher = Case::new("tput", 100.0, 5.0, RegDir::Higher).expect("场景");
    let lower = Case::new("latency", 50.0, 10.0, RegDir::Lower).expect("场景");
    let regress_high = higher.judge(91.0);
    let regress_low = lower.judge(57.0);
    assert!(regress_high.effect().expect("效应") < 0.0);
    assert!(regress_low.effect().expect("效应") < 0.0);
    assert!((regress_high.effect().expect("效应") + 0.09).abs() < 1e-12);
    let improved = lower.judge(40.0);
    assert!(improved.effect().expect("效应") > 0.0);
}

fn a_reversed_or_non_finite_interval_is_rejected() {
    assert_eq!(
        Estimate::interval(5.0, 3.0, 4.0).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Estimate::interval(f64::NAN, 1.0, 2.0).unwrap_err().code(),
        Code::Invalid
    );
}

fn two_equivalent_measurements_agree() {
    let first = Estimate::interval(99.0, 100.0, 101.0).expect("区间");
    let second = Estimate::interval(98.0, 99.5, 101.0).expect("区间");
    assert!(self_check(&first, &second, 5.0).is_agree());
}

fn measurements_apart_beyond_the_band_disagree() {
    let first = Estimate::interval(95.0, 100.0, 105.0).expect("区间");
    let second = Estimate::interval(104.0, 110.0, 115.0).expect("区间");
    let check = self_check(&first, &second, 5.0);
    let ratio = check.ratio().expect("不一致携带偏差");
    assert!((ratio - 10.0 / 110.0).abs() < 1e-12);
}

fn disjoint_intervals_stay_open_straddle() {
    let first = Estimate::interval(90.0, 92.0, 94.0).expect("区间");
    let second = Estimate::interval(100.0, 102.0, 104.0).expect("区间");
    assert!(matches!(
        self_check(&first, &second, 5.0),
        SelfCheck::Open {
            reason: OpenReason::Straddle,
            ..
        }
    ));
}

fn a_non_positive_measurement_is_degenerate() {
    let zero = Estimate::point(0.0);
    let other = Estimate::point(100.0);
    assert!(matches!(
        self_check(&zero, &other, 5.0),
        SelfCheck::Open {
            reason: OpenReason::Degenerate,
            ..
        }
    ));
}
}

// ---------------------------------------------------------------------------
// stats：闭式统计量
// ---------------------------------------------------------------------------

testkit::test_cases! {
fn mean_and_sample_deviation_match_the_closed_form() {
    let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(mean(&samples), 3.0);
    assert!((sd(&samples) - 2.5_f64.sqrt()).abs() < 1e-12);
}

fn a_single_sample_has_no_spread() {
    assert_eq!(sd(&[7.0]), 0.0);
    assert_eq!(mean(&[]), 0.0);
}

fn quantiles_are_nearest_rank_not_interpolated() {
    let sorted = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(quantile(&sorted, 0.50), 2.0);
    assert_eq!(quantile(&sorted, 0.95), 4.0);
    assert_eq!(quantile(&sorted, 0.0), 1.0, "下界夹住在最小秩");
}

fn resampling_is_reproducible() {
    let samples = [9.0, 11.0, 10.5, 9.4, 12.0, 10.1, 9.9, 10.8];
    let first = interval(&samples, |drawn| Some(mean(drawn)));
    let second = interval(&samples, |drawn| Some(mean(drawn)));
    assert_eq!(first, second);
    let draws = first.expect("区间");
    assert!(draws.windows(2).all(|pair| pair[0] <= pair[1]), "必须升序");
    assert!(
        draws[0] <= 10.0 && 10.0 <= draws[draws.len() - 1],
        "区间应罩住真值"
    );
}

fn an_undefined_statistic_has_no_interval() {
    assert_eq!(interval(&[1.0, -1.0], |_| None), None);
    assert!(resample(&[], |_| Some(0.0)).is_empty());
}
}

// ---------------------------------------------------------------------------
// escape：JSON 字符串转义
// ---------------------------------------------------------------------------

fn quoted(text: &str) -> String {
    let mut out = String::new();
    quote(&mut out, text);
    out
}

testkit::test_cases! {
fn quotes_and_backslashes_are_escaped_and_unicode_passes_through() {
    assert_eq!(quoted("a\"b\\c"), "\"a\\\"b\\\\c\"");
    assert_eq!(quoted("延迟"), "\"延迟\"");
}

fn short_escapes_win_for_the_common_controls() {
    assert_eq!(quoted("a\nb\tc\rd"), "\"a\\nb\\tc\\rd\"");
}

fn other_controls_become_unicode_escapes() {
    assert_eq!(quoted("\u{7}"), "\"\\u0007\"");
    assert_eq!(quoted("\u{85}"), "\"\\u0085\"");
}
}
