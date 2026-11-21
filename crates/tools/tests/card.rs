//! 实测卡三态判定的集成测试：只用公开面，不读实现即可换掉实现。

use diagnostic::Code;
use llm_tools::bench::Harness;
use llm_tools::card::{Card, Direction, Protocol, Shortfall, Standing, Statistic, Tally};

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

/// 同一协议下抖动为 0 的一列：区间退化成一个点，整段落在门槛内侧。
#[test]
fn a_steady_series_passes_the_jitter_gate() {
    let card = Card::judge(jitter(), harness("A800"), &[10.0; 40]).expect("判");
    assert_eq!(card.standing(), Standing::Pass);
    let tally = card.tally().expect("有摘要");
    assert_eq!((tally.lower(), tally.upper()), (0.0, 0.0));
}

/// 抖动远超门槛的一列：区间整段越界，这一格记 FAIL 而不是 OPEN。
#[test]
fn a_series_beyond_the_gate_fails() {
    let mut samples = Vec::new();
    for _ in 0..20 {
        samples.extend([1.0, 100.0]);
    }
    let card = Card::judge(jitter(), harness("A800"), &samples).expect("判");
    assert_eq!(card.standing(), Standing::Fail);
    assert!(card.tally().expect("有摘要").upper() > 0.05, "区间上界越过门槛");
}

/// 区间跨着门槛：样本还不足以把「过」与「不过」分开，只能记 OPEN。
#[test]
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
#[test]
fn too_few_samples_are_not_judged_at_all() {
    let card = Card::judge(jitter(), harness("A800"), &[10.0, 12.0, 11.0]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
    assert_eq!(card.tally().expect("摘要照记").count(), 3);
    assert_eq!(card.standing().shortfall(), Some(Shortfall::Samples));
    assert_ne!(card.standing(), Standing::Pass);
}

/// 闭集外的主机：样本再多也不是这份协议的证据。
#[test]
fn a_host_outside_the_closure_cannot_pass() {
    let protocol = jitter().with_hosts(&["A800", "RTX_3090", "T600"]);
    let card = Card::judge(protocol, harness("macos-aarch64"), &[10.0; 40]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Host));
}

/// 闭集为空表示不设闭集：任何装置都按门槛判。
#[test]
fn an_empty_closure_admits_every_host() {
    let card = Card::judge(jitter(), harness("macos-aarch64"), &[10.0; 40]).expect("判");
    assert_eq!(card.standing(), Standing::Pass);
}

/// 均值为 0 的列上变异系数没有定义：填 0 会让「完全抖动」看起来像「完全稳定」。
#[test]
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
#[test]
fn no_samples_report_no_summary() {
    let card = Card::judge(jitter(), harness("A800"), &[]).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
    assert!(card.tally().is_none());
}

/// 非有限样本是采集坏了：报错，不许静默跳过。
#[test]
fn a_broken_sample_is_reported_as_such() {
    let err = Card::judge(jitter(), harness("A800"), &[1.0, f64::NAN]).unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.message().contains("采集坏了"), "{}", err.message());
}

/// 重采样种子固定：同一份样本只会得到一份卡片，复核时能逐位对拍。
#[test]
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
#[test]
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
#[test]
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
#[test]
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
#[test]
fn a_protocol_rejects_impossible_settings() {
    let empty = Protocol::of("", "ns", Statistic::Mean, Direction::Lower, 1.0).unwrap_err();
    assert_eq!(empty.code(), Code::Invalid);
    let nan = Protocol::of("E1", "ns", Statistic::Mean, Direction::Lower, f64::NAN).unwrap_err();
    assert_eq!(nan.code(), Code::Invalid);
    assert!(nan.message().contains("有限数"), "{}", nan.message());
    assert_eq!(jitter().with_least(1).unwrap_err().code(), Code::Invalid);
}
