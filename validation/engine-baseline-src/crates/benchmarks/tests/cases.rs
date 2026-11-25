//! 用例：登记自检、归档对拍与三态判定；通用装配全部复用现有分析件与记录断言。

use super::*;

/// 读取本机档案；未登记主机明确跳过，不能拿他机基线判定。
fn local(what: &str) -> Option<(&'static HostProfile, String)> {
    let name = host();
    let profile = local_profile(&name);
    if profile.is_none() {
        testkit::skip(format!(
            "benchmarks: 主机 `{name}` 没有建档，{what}无从判定（跨机绝对值不可比）"
        ));
    }
    profile.map(|profile| (profile, name))
}

#[test]
pub(crate) fn registered_baselines_stay_within_their_budget() {
    let Some((profile, name)) = local("基线") else {
        return;
    };
    let ready: Vec<&Scenario> = SCENARIOS
        .iter()
        .filter(|sc| baseline_for(&name, sc.backend, sc.scenario).is_some())
        .collect();
    if ready.is_empty() {
        testkit::skip(format!("benchmarks: 主机 `{name}` 没有登记任何基线"));
        return;
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for scenario in ready {
        if judge_one(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
pub(crate) fn registered_ceilings_stay_within_their_budget() {
    let Some((profile, name)) = local("上界") else {
        return;
    };
    let ready: Vec<&Ceiling> = CEILINGS.iter().filter(|c| c.host == name).collect();
    if ready.is_empty() {
        testkit::skip(format!("benchmarks: 主机 `{name}` 没有登记搬运上界"));
        return;
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ceiling in ready {
        if judge_ceiling(manifest, profile, ceiling).is_none() {
            return;
        }
    }
}

#[test]
pub(crate) fn every_registered_host_is_fully_described() {
    for baseline in BASELINES {
        assert!(
            profile_for(baseline.host).is_some(),
            "主机 `{}` 有基线却没有档案",
            baseline.host
        );
    }
    for profile in HOSTS {
        let missing = missing_ceiling(profile);
        assert!(
            missing.is_none(),
            "主机 `{}` 缺少上界：{missing:?}",
            profile.host
        );
    }
}

/// 构造一条覆盖全字段的算子报告，数值只用于结构对拍。
fn op_report<'a>(host: &HostProfile, scenario: &'a Scenario) -> Report<'a> {
    Report {
        scenario,
        record: record_for(host, scenario, 1_500_000.0),
        roofline: Roofline::of(host, scenario, 2_000_000.0, 2_000_000.0),
        verdict: Judgement::Improved {
            observed: 1.0,
            baseline: 2.0,
        }
        .as_str(),
        baseline_ns: 2_000_000.0,
        budget_percent: 5.0,
    }
}

#[test]
pub(crate) fn the_report_carries_every_field_a_reader_needs_to_judge_it() {
    for scenario in SCENARIOS {
        let report = op_report(&HOSTS[0], scenario);
        assert_op_record(
            &HOSTS[0],
            &report.record,
            (
                scenario.scenario,
                scenario.backend,
                scenario.shape,
                scenario.batch,
                scenario.dtype,
            ),
            1_500_000.0,
        );
        let rendered = report.render();
        assert_op_report_fields(&rendered);
        let parsed = parsed_report(&rendered);
        assert_json_int(&parsed, "batch", i64::from(scenario.batch));
        assert_json_int(
            &parsed,
            "working_set_bytes",
            (report.roofline.read + report.roofline.write) as i64,
        );
    }
}

/// 两类无基线归档共用装配：缺测量时保留跳过语义，不静默通过后续条目。
macro_rules! archive_when_measured {
    ($name:ident, $items:expr, $run:ident, $what:literal) => {
        #[test]
        pub(crate) fn $name() {
            let Some((profile, _)) = local($what) else {
                return;
            };
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
            for scenario in $items {
                if $run(manifest, profile, scenario).is_none() {
                    return;
                }
            }
        }
    };
}

archive_when_measured!(
    registered_load_records_archive_when_measured,
    LOAD_SCENARIOS,
    archive_load,
    "装载记录"
);
archive_when_measured!(
    registered_model_records_archive_when_measured,
    DECODE_SCENARIOS,
    archive_model,
    "模型记录"
);

#[test]
pub(crate) fn load_records_carry_every_field_a_reader_needs_to_judge_it() {
    assert_eq!(LOAD_SCENARIOS.len(), if cfg!(unix) { 8 } else { 4 });
    for scenario in LOAD_SCENARIOS {
        let record = load_record(&HOSTS[0], scenario, 1_500.0);
        assert_op_record(
            &HOSTS[0],
            &record,
            (
                scenario.scenario,
                scenario.backend,
                scenario.shape,
                scenario.batch,
                scenario.dtype,
            ),
            1_500.0,
        );
        let rendered = LoadReport { scenario, record }.render();
        assert_load_report_fields(&rendered);
        let parsed = parsed_report(&rendered);
        assert_json_int(&parsed, "batch", i64::from(scenario.batch));
        assert_json_int(&parsed, "bytes_per_iter", scenario.bytes as i64);
    }
}

#[test]
pub(crate) fn model_records_carry_every_field_a_reader_needs_to_judge_it() {
    assert_eq!(DECODE_SCENARIOS.len(), 2, "单序列与同批四条各一条");
    for scenario in DECODE_SCENARIOS {
        let record = model_record(&HOSTS[0], scenario, 1_500_000.0);
        assert_eq!(record.kind(), EntryKind::Model);
        assert_record_fields(
            &record,
            &HOSTS[0],
            &RecordExpectation::new(
                scenario.scenario,
                "cpu-reference",
                &[MODEL_DIM, MODEL_OUT],
                scenario.batch,
                "f32",
                1_500_000.0,
            ),
        );
        let rendered = DecodeReport { scenario, record }.render();
        assert_decode_report_fields(&rendered);
        let parsed = parsed_report(&rendered);
        assert_json_int(&parsed, "batch", i64::from(scenario.batch));
        assert_json_int(&parsed, "tokens_per_iter", scenario.tokens as i64);
        assert_json_int(&parsed, "pages_in_use", scenario.pages as i64);
    }
}

// 协议卡只负责测量与搬运；门槛、区间和三态判定唯一由 `tool::card` 持有。
#[path = "decode_card.rs"]
mod decode_card;
#[path = "../src/fixtures.rs"]
#[allow(dead_code)]
mod fixtures;
use decode_card::{WINDOWS, archive, harness, jitter, protocol, samples};
use tool::card::{Card, Shortfall, Standing};

#[test]
fn a_real_decode_series_becomes_a_reviewable_card() {
    let samples = samples();
    assert_eq!(samples.len(), WINDOWS);
    assert!(samples.iter().all(|ns| *ns > 0.0), "时钟必须前进");
    let card = Card::judge(jitter(), harness(host()), &samples).expect("判");
    let tally = card.tally().expect("摘要");
    assert_eq!(tally.count(), samples.len());
    assert!(tally.sd() >= 0.0 && tally.lower() <= tally.upper());
    let text = std::fs::read_to_string(archive(&card, host())).expect("读回");
    let host_field = format!("\"host\": \"{}\"", host());
    for field in [
        "\"protocol\": \"E10\"",
        "\"statistic\": \"cv\"",
        &host_field,
    ] {
        assert!(text.contains(field), "卡片缺少 {field}");
    }
    let standing = card.standing();
    assert!(matches!(
        standing,
        Standing::Pass | Standing::Fail | Standing::Open(_)
    ));
    let reason = standing.shortfall().map_or("met", Shortfall::as_str);
    println!("E10 decode_step_ns → {} ({reason})", standing.as_str());
}

#[test]
fn the_same_series_passes_a_loose_gate_and_fails_an_impossible_one() {
    let samples = samples();
    let failed =
        Card::judge(protocol("gate", 0.0, WINDOWS), harness(host()), &samples).expect("判");
    let passed =
        Card::judge(protocol("gate", 1e9, WINDOWS), harness(host()), &samples).expect("判");
    assert_eq!(failed.standing(), Standing::Fail, "不可达门槛必须 FAIL");
    assert_eq!(passed.standing(), Standing::Pass, "极宽门槛必须 PASS");
    assert_eq!(
        passed.tally().expect("摘要").upper(),
        failed.tally().expect("摘要").upper(),
        "换门槛不能改变同一列样本的摘要"
    );
}

#[test]
fn an_unmeasured_cell_stays_open_never_a_silent_pass() {
    let samples = samples();
    let short = protocol("gate", 0.05, WINDOWS + 1);
    let card = Card::judge(short, harness(host()), &samples).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Samples));
    let outside = jitter().with_hosts(&["no-such-part"]);
    let card = Card::judge(outside, harness(host()), &samples).expect("判");
    assert_eq!(card.standing(), Standing::Open(Shortfall::Host));
}

/// 四张台账合成唯一注册集；重复身份直接视为登记错误。
fn registered_ids() -> std::collections::BTreeSet<String> {
    let extra = MEASUREMENT_GROUPS.iter().flat_map(|(group, benchmarks)| {
        benchmarks.iter().map(move |benchmark| (*group, *benchmark))
    });
    let ids: Vec<String> = SCENARIOS
        .iter()
        .map(|sc| (sc.group, sc.benchmark))
        .chain(LOAD_SCENARIOS.iter().map(|sc| (sc.group, sc.benchmark)))
        .chain(DECODE_SCENARIOS.iter().map(|sc| (sc.group, sc.benchmark)))
        .chain(extra)
        .map(|(group, benchmark)| measurement_id(group, benchmark))
        .collect();
    let unique: std::collections::BTreeSet<String> = ids.iter().cloned().collect();
    assert_eq!(ids.len(), unique.len(), "测量身份重复登记");
    unique
}

#[test]
pub(crate) fn no_measurement_is_filed_under_an_unregistered_group() {
    let root = criterion_root(Path::new(env!("CARGO_MANIFEST_DIR")));
    if !root.is_dir() {
        testkit::skip(format!(
            "benchmarks: 还没跑过基准（{} 不存在）",
            root.display()
        ));
        return;
    }
    let registered = registered_ids();
    let unregistered: Vec<String> = measured_ids(&root)
        .into_iter()
        .filter(|id| !registered.contains(id))
        .collect();
    assert!(unregistered.is_empty(), "未登记测量：{unregistered:?}");
}
