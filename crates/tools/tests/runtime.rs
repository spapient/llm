//! P2–P4 测量层内核集成测试：趋势对照、形状网格、服务分相与 goodput。
//! 只用公开面，不读实现即可换掉实现。

use llm_tools::bench::{
    AllocProfile, Arrival, PhaseLatency, ServingScene, aligned_points, boundary_points,
    goodput_tokens_per_s, phase_latency, slo_hit_rate, tokens_per_s,
};
use llm_tools::regress::{HistorySample, Judgement, compare, median_baseline};

testkit::test_cases! {
fn serving_scene_and_arrival_names() {
    assert_eq!(ServingScene::SingleStream.as_str(), "single-stream");
    assert_eq!(ServingScene::MultiStream.as_str(), "multi-stream");
    assert_eq!(ServingScene::Server.as_str(), "server");
    assert_eq!(ServingScene::Offline.as_str(), "offline");
    assert_eq!(Arrival::OpenPoisson.as_str(), "open-poisson");
    assert_eq!(Arrival::ClosedConcurrency.as_str(), "closed-concurrency");
}

fn phase_latency_splits_gaps() {
    let phase: PhaseLatency = phase_latency(&[0.0, 10.0, 30.0, 60.0]).expect("时序合法");
    assert_eq!(phase.ttft_ns(), 0.0);
    assert_eq!(phase.e2e_ns(), 60.0);
    assert_eq!(phase.itl(), &[10.0, 20.0, 30.0]);
    assert!((phase.tpot_ns() - 20.0).abs() < 1e-9);
}

fn phase_latency_single_token_has_empty_itl() {
    let phase = phase_latency(&[5.0]).expect("单 token");
    assert_eq!(phase.ttft_ns(), 5.0);
    assert_eq!(phase.e2e_ns(), 5.0);
    assert!(phase.itl().is_empty());
    assert_eq!(phase.tpot_ns(), 0.0);
}

fn phase_latency_rejects_empty_nonfinite_or_regressed() {
    assert!(phase_latency(&[]).is_none());
    assert!(phase_latency(&[f64::INFINITY, 1.0]).is_none());
    assert!(phase_latency(&[1.0, 0.5]).is_none());
}

fn slo_hit_rate_and_throughput() {
    assert!((slo_hit_rate(3, 4) - 0.75).abs() < 1e-9);
    assert_eq!(slo_hit_rate(2, 0), 0.0);
    assert!((tokens_per_s(1, 1e9) - 1.0).abs() < 1e-9);
    assert_eq!(tokens_per_s(4, 0.0), 0.0);
}

fn goodput_applies_hit_rate() {
    let raw = tokens_per_s(100, 1e9);
    let good = goodput_tokens_per_s(100, 1e9, 9, 10);
    assert!((good - raw * 0.9).abs() < 1e-9);
}

fn median_baseline_is_middle_sample() {
    let history: Vec<HistorySample> = [10.0, 20.0, 30.0]
        .iter()
        .map(|v| HistorySample::new(*v).expect("合法"))
        .collect();
    assert!((median_baseline(&history).expect("有历史") - 20.0).abs() < 1e-9);
    assert!(median_baseline(&[]).is_none());
}

fn trend_regresses_when_current_exceeds_baseline() {
    let history: Vec<HistorySample> = [20.0, 21.0, 19.0]
        .iter()
        .map(|v| HistorySample::new(*v).expect("合法"))
        .collect();
    let regressed = compare(30.0, &history, 10.0).expect("可判定");
    assert!(regressed.is_regressed());
    let improved = compare(18.0, &history, 10.0).expect("可判定");
    assert!(matches!(improved, Judgement::Improved { .. }));
}

fn trend_rejects_invalid_sample_and_empty_history() {
    assert!(HistorySample::new(f64::NAN).is_none());
    assert!(HistorySample::new(-1.0).is_none());
    assert!(compare(20.0, &[], 10.0).is_none());
}

fn grid_aligned_and_boundary_points() {
    assert_eq!(aligned_points(16, 40), vec![16, 32]);
    assert_eq!(boundary_points(16, 40), vec![15, 31]);
    assert!(aligned_points(0, 40).is_empty());
    assert!(boundary_points(0, 40).is_empty());
}

fn alloc_profile_keeps_counts() {
    let profile = AllocProfile::new(12, 2048, 768);
    assert_eq!(profile.count(), 12);
    assert_eq!(profile.bytes(), 2048);
    assert_eq!(profile.peak_bytes(), 768);
}
}
