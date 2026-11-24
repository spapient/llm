//! 用例：登记自检、归档对拍与判定。

use super::*;

#[test]
pub(crate) fn registered_baselines_stay_within_their_budget() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有建档（跨机绝对值不可比），先在靶机上跑 cargo bench 后建档"
        ));
        return;
    };

    let ready: Vec<&Scenario> = SCENARIOS
        .iter()
        .filter(|sc| baseline_for(&host, sc.backend, sc.scenario).is_some())
        .collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有登记任何基线，先在靶机上跑 cargo bench 后登记"
        ));
        return;
    }

    for scenario in ready {
        if judge_one(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
pub(crate) fn registered_ceilings_stay_within_their_budget() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有建档，上界无从判定（跨机绝对值不可比）"
        ));
        return;
    };

    let ready: Vec<&Ceiling> = CEILINGS.iter().filter(|c| c.host == host).collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有登记搬运上界，roofline 的达成率无法计算"
        ));
        return;
    }

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
            "主机 `{}` 有基线却没有档案：机型、系统、工具链必须随数字一起归档",
            baseline.host
        );
    }
    for profile in HOSTS {
        for tier in ["l1", "l2", "dram"] {
            for traffic in ["read", "write"] {
                assert!(
                    ceiling_for(profile.host, tier, traffic).is_some(),
                    "主机 `{}` 缺少 {tier} 档的 {traffic} 上界",
                    profile.host
                );
            }
        }
    }
}

#[test]
pub(crate) fn the_report_carries_every_field_a_reader_needs_to_judge_it() {
    let host = &HOSTS[0];
    for scenario in SCENARIOS {
        let record = record_for(host, scenario, 1_500_000.0);
        assert_op_record(
            host,
            &record,
            (
                scenario.scenario,
                scenario.backend,
                scenario.shape,
                scenario.batch,
                scenario.dtype,
            ),
            1_500_000.0,
        );

        let report = Report {
            scenario,
            record,
            roofline: Roofline::of(host, scenario, 2_000_000.0, 2_000_000.0),
            verdict: Judgement::Improved {
                observed: 1.0,
                baseline: 2.0,
            }
            .as_str(),
            baseline_ns: 2_000_000.0,
            budget_percent: 5.0,
        };
        let rendered = report.render();
        assert_present(
            &rendered,
            &[
                "\"surface\"",
                "\"shape\"",
                "\"dtype\"",
                "\"backend\"",
                "\"metric\"",
                "\"direction\"",
                "\"budget_percent\"",
                "\"baseline_ns\"",
                "\"observed_ns\"",
                "\"verdict\"",
                "\"host\"",
                "\"cpu\"",
                "\"os\"",
                "\"toolchain\"",
                "\"read_bytes\"",
                "\"write_bytes\"",
                "\"working_set_bytes\"",
                "\"cache_tier\"",
                "\"flops\"",
                "\"read_ceiling_mbps\"",
                "\"write_ceiling_mbps\"",
                "\"roofline_ns\"",
                "\"bottleneck_side\"",
                "\"achieved_percent\"",
                "\"declared_ns\"",
                "\"ceiling_percent\"",
                "\"s_plus_percent\"",
                "\"s_plus_line\"",
                "\"mflops_per_s\"",
                "\"bottleneck\"",
            ],
            "",
        );
        let parsed = parsed_report(&rendered);
        assert_eq!(
            parsed_batch(&parsed),
            Some(i64::from(scenario.batch)),
            "批大小是口径的一部分：1 行 = decode 步，与 prefill 的多行批差三个数量级"
        );
        assert_eq!(
            parsed
                .get("working_set_bytes")
                .and_then(|value| value.integer().ok()),
            Some((report.roofline.read + report.roofline.write) as i64),
            "工作集是档位的判据：写错档位，达成率就没有意义"
        );
    }
}

#[test]
pub(crate) fn registered_load_records_archive_when_measured() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有建档，装载记录无从归档（跨机绝对值不可比）"
        ));
        return;
    };

    for scenario in LOAD_SCENARIOS {
        if archive_load(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
pub(crate) fn load_records_carry_every_field_a_reader_needs_to_judge_it() {
    let host = &HOSTS[0];
    let expected = if cfg!(unix) { 8 } else { 4 };
    assert_eq!(
        LOAD_SCENARIOS.len(),
        expected,
        "装载台账条目数与基准用例逐条对齐：File 四条 + mmap 四条（仅 Unix）"
    );
    for scenario in LOAD_SCENARIOS {
        let record = load_record(host, scenario, 1_500.0);
        assert_op_record(
            host,
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

        let report = LoadReport { scenario, record };
        let rendered = report.render();
        assert_present(
            &rendered,
            &[
                "\"surface\"",
                "\"shape\"",
                "\"dtype\"",
                "\"backend\"",
                "\"metric\"",
                "\"bytes_per_iter\"",
                "\"observed_ns\"",
                "\"gb_per_s\"",
                "\"host\"",
                "\"cpu\"",
                "\"os\"",
                "\"toolchain\"",
                "\"algorithm\"",
            ],
            "装载",
        );
        let parsed = parsed_report(&rendered);
        assert_eq!(parsed_batch(&parsed), Some(i64::from(scenario.batch)));
        assert_eq!(
            parsed
                .get("bytes_per_iter")
                .and_then(|value| value.integer().ok()),
            Some(scenario.bytes as i64),
            "字节账目与基准 Throughput::Bytes 同源"
        );
    }
}

#[test]
pub(crate) fn registered_model_records_archive_when_measured() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有建档，模型记录无从归档（跨机绝对值不可比）"
        ));
        return;
    };

    for scenario in DECODE_SCENARIOS {
        if archive_model(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
pub(crate) fn model_records_carry_every_field_a_reader_needs_to_judge_it() {
    let host = &HOSTS[0];
    assert_eq!(DECODE_SCENARIOS.len(), 2, "单序列与同批四条各一条");
    for scenario in DECODE_SCENARIOS {
        let record = model_record(host, scenario, 1_500_000.0);
        // 记录身份 / 几何 / 后端必须与场景描述符逐字段对上，避免归档元数据串台。
        assert_eq!(record.kind(), EntryKind::Model);
        assert_record_fields(
            &record,
            host,
            &RecordExpectation::new(
                scenario.scenario,
                "cpu-reference",
                &[MODEL_DIM, MODEL_OUT],
                scenario.batch,
                "f32",
                1_500_000.0,
            ),
        );

        let report = DecodeReport { scenario, record };
        let rendered = report.render();
        assert_present(
            &rendered,
            &[
                "\"surface\"",
                "\"shape\"",
                "\"dtype\"",
                "\"backend\"",
                "\"metric\"",
                "\"tokens_per_iter\"",
                "\"pages_in_use\"",
                "\"observed_ns\"",
                "\"tokens_per_s\"",
                "\"host\"",
                "\"cpu\"",
                "\"os\"",
                "\"toolchain\"",
                "\"algorithm\"",
            ],
            "模型",
        );
        let parsed = parsed_report(&rendered);
        assert_eq!(
            parsed_batch(&parsed),
            Some(i64::from(scenario.batch)),
            "批大小是口径的一部分：1 行 = decode 步，与 prefill 的多行批差三个数量级"
        );
        assert_eq!(
            parsed
                .get("tokens_per_iter")
                .and_then(|value| value.integer().ok()),
            Some(scenario.tokens as i64)
        );
        assert_eq!(
            parsed
                .get("pages_in_use")
                .and_then(|value| value.integer().ok()),
            Some(scenario.pages as i64),
            "页数是页池 in_use() 的可观测账目，不虚构峰值字节"
        );
    }
}

// ---- 协议卡：真实一步 decode 的样本 → `tool::card` 三态判定并归档 ----
// 把真实一步 decode 的实测样本接进 `tools::card` 的三态判定，并归档成一份可复核的卡片。
//
// 分工仍是刻意的：本文件只负责**测**（合成小栈走 [`Predict::logits`] 一步，逐窗口取墙钟）
// 与**搬运**（样本 → [`Card::judge`] → 归档）；判什么量、门槛多少、区间怎么算全在
// `tools::card`，这里不写第二份判定规则。
//
// 与 `tests/regress.rs` 的分工：那一边判「相对基线回退了没有」，需要 criterion 的区间估计
// 与登记基线；这一边判「这份协议卡这一格成立了没有」，只有绝对门槛与一列样本，所以不需要
// criterion，也不需要本机基线——`lld` 的 Must 本来就是绝对值。
//
// 断言只落在**与时间无关**的性质上：同一列样本换成不可达门槛必然 FAIL、换成极宽门槛必然
// PASS、样本数不到下限必然 OPEN、主机不在闭集里必然 OPEN。真实的 `E10`（变异系数 < 5%）
// 照 `lld` 的口径判出来归档，但不在测试里断言它通过——这台机器当时抖不抖是测量结果，
// 不是这个测试的前提。

#[path = "../src/fixtures.rs"]
#[allow(dead_code)]
mod fixtures;

use std::hint::black_box;
use std::time::Instant;

use engine::exec::Stream;
use engine::kv::{Cache, Pager};
use engine::{Batch, Sequence};
use fixtures::PROMPT;
use serve::Predict;
use serve::model::CpuModel;
use tool::card::{Card, Direction as CardDirection, Protocol, Shortfall, Standing, Statistic};

/// 热身步数：不计入样本，只用来把首次分页与寄存器预热甩掉。
const WARMUP: usize = 4;
/// 采样的步数：`lld` §10.10 的预注册窗口在机器上至少这么多，才谈得上区间。
const WINDOWS: usize = 32;
/// 页池规模：`WINDOWS + WARMUP` 步各追加一个 token，留足余量避免半途扩容。
const PAGES: usize = 128;
/// 每页 token 数。
const PAGE_SIZE: usize = 4;

/// 装配合成栈与一条序列；模型、批与序列在测量轮次间复用，前向内不重新分配权重。
fn fixture() -> (CpuModel, Batch, Sequence) {
    let model = fixtures::synthetic_model();

    let pager = Pager::new(Cache::new(PAGES, PAGE_SIZE).expect("页池"), 2).expect("行宽");
    let mut batch = Batch::new(pager);
    let sequence = Sequence::new(0, PROMPT, 2, PAGE_SIZE, Stream::new(0)).expect("序列");
    batch.admit(sequence.clone()).expect("准入");
    (model, batch, sequence)
}

/// 一步 decode 的每步墙钟（ns）：先热身 [`WARMUP`] 步（不采样），再取 [`WINDOWS`] 步。
///
/// 只测 `logits` 这一步：它是纯前向，样本里不掺调度、批管理或序列化的时间——那些是别的卡
/// 该判的量，混进来会让变异系数失去解释力。
fn decode_samples(model: &mut CpuModel, batch: &mut Batch, sequence: &Sequence) -> Vec<f64> {
    for _ in 0..WARMUP {
        black_box(model.logits(sequence, batch).expect("热身前向"));
    }
    (0..WINDOWS)
        .map(|_| {
            let started = Instant::now();
            black_box(model.logits(sequence, batch).expect("一步前向"));
            started.elapsed().as_nanos() as f64
        })
        .collect()
}

/// 本机装置身份：机型与系统版本这一层还没读（不猜），工具链取运行环境的架构与 `rustc`。
fn harness() -> Harness {
    Harness::new(
        host(),
        "unknown",
        std::env::consts::OS,
        std::env::consts::ARCH,
        "rustc",
        "cpu-reference",
    )
}

/// `lld` 的 E10 卡：一步 decode 的变异系数，95% 区间上界 < 5% 才算 PASS。
fn jitter() -> Protocol {
    Protocol::of(
        "E10",
        "decode_step_ns",
        Statistic::Cv,
        CardDirection::Lower,
        0.05,
    )
    .expect("E10 协议")
    .with_least(WINDOWS)
    .expect("样本下限")
}

/// 真实一列样本落成一张卡，并归档到工作区根的 `target/bench-report/<host>/card-<protocol>.json`。
/// 测试进程的 CWD 是 crate 根，所以归档路径必须从 `CARGO_MANIFEST_DIR` 起算，否则卡片落进
/// `crates/benchmarks/target/`，而 CI 从工作区根的 `target/bench-report/` 上传，报告静默丢失。
fn archive(card: &Card) -> std::path::PathBuf {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/bench-report")
        .join(host());
    std::fs::create_dir_all(&directory).expect("归档目录");
    let path = directory.join(format!("card-{}.json", card.protocol().name()));
    std::fs::write(&path, card.to_json()).expect("写卡");
    path
}

#[test]
fn a_real_decode_series_becomes_a_reviewable_card() {
    let (mut model, mut batch, sequence) = fixture();
    let samples = decode_samples(&mut model, &mut batch, &sequence);
    assert_eq!(samples.len(), WINDOWS, "窗口数就是协议下限");
    assert!(
        samples.iter().all(|ns| *ns > 0.0),
        "墙钟分辨率下每一步都必须是正数：0 说明时钟没走"
    );

    let card = Card::judge(jitter(), harness(), &samples).expect("判");
    let tally = card.tally().expect("窗口够、统计量有定义");
    assert_eq!(tally.count(), WINDOWS);
    assert_eq!(tally.count(), samples.len(), "摘要只覆盖这一列样本");
    assert!(
        tally.sd() >= 0.0 && tally.lower() <= tally.upper(),
        "区间不许倒挂"
    );

    let path = archive(&card);
    let text = std::fs::read_to_string(&path).expect("读回");
    assert!(
        text.contains("\"protocol\": \"E10\""),
        "卡片带协议名：{text}"
    );
    assert!(text.contains("\"statistic\": \"cv\""), "卡片带聚合方式");
    assert!(
        text.contains(&format!("\"host\": \"{}\"", host())),
        "卡片带装置身份"
    );

    // 判成什么由这台机器当时的抖动决定，不在测试里许诺；但三种结论必须都有名字。
    let standing = card.standing();
    assert!(
        matches!(
            standing,
            Standing::Pass | Standing::Fail | Standing::Open(_)
        ),
        "三态之外没有第四种结论"
    );
    let reason = standing.shortfall().map_or("met", Shortfall::as_str);
    println!("E10 decode_step_ns → {} ({reason})", standing.as_str());
}

#[test]
fn the_same_series_passes_a_loose_gate_and_fails_an_impossible_one() {
    let (mut model, mut batch, sequence) = fixture();
    let samples = decode_samples(&mut model, &mut batch, &sequence);

    // 变异系数按定义 ≥ 0：门槛取 0 时区间下界必然 ≥ 门槛，整段在外侧 → FAIL。
    let impossible = Protocol::of(
        "gate",
        "decode_step_ns",
        Statistic::Cv,
        CardDirection::Lower,
        0.0,
    )
    .expect("协议")
    .with_least(WINDOWS)
    .expect("样本下限");
    let failed = Card::judge(impossible, harness(), &samples).expect("判");
    assert_eq!(failed.standing(), Standing::Fail, "不可达门槛必须 FAIL");

    // 门槛取一个比任何抖动都大的数：区间上界必然远小于它 → PASS。
    let loose = Protocol::of(
        "gate",
        "decode_step_ns",
        Statistic::Cv,
        CardDirection::Lower,
        1e9,
    )
    .expect("协议")
    .with_least(WINDOWS)
    .expect("样本下限");
    let passed = Card::judge(loose, harness(), &samples).expect("判");
    assert_eq!(passed.standing(), Standing::Pass, "极宽门槛必须 PASS");
    assert!(
        passed.tally().expect("摘要").upper() == failed.tally().expect("摘要").upper(),
        "同一列样本的摘要只有一份，换门槛不改摘要"
    );
}

#[test]
fn an_unmeasured_cell_stays_open_never_a_silent_pass() {
    let (mut model, mut batch, sequence) = fixture();
    let samples = decode_samples(&mut model, &mut batch, &sequence);

    let short = Protocol::of(
        "gate",
        "decode_step_ns",
        Statistic::Cv,
        CardDirection::Lower,
        0.05,
    )
    .expect("协议")
    .with_least(WINDOWS + 1)
    .expect("样本下限");
    let card = Card::judge(short, harness(), &samples).expect("判");
    assert_eq!(
        card.standing(),
        Standing::Open(Shortfall::Samples),
        "差一个窗口也只能 OPEN，不许拿点估计冒充区间"
    );

    let outside = jitter().with_hosts(&["no-such-part"]);
    let card = Card::judge(outside, harness(), &samples).expect("判");
    assert_eq!(
        card.standing(),
        Standing::Open(Shortfall::Host),
        "闭集外的机器上跑出来的数不是这份协议的证据"
    );
}

#[test]
pub(crate) fn no_measurement_is_filed_under_an_unregistered_group() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..").join("target/criterion");
    let Ok(entries) = std::fs::read_dir(&root) else {
        testkit::skip(format!(
            "benchmarks: 还没跑过基准（{} 不存在）",
            root.display()
        ));
        return;
    };

    let groups: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    let registered = SCENARIOS
        .iter()
        .map(|sc| (sc.scenario, sc.group, sc.benchmark))
        .chain(
            LOAD_SCENARIOS
                .iter()
                .map(|sc| (sc.scenario, sc.group, sc.benchmark)),
        )
        .chain(
            DECODE_SCENARIOS
                .iter()
                .map(|sc| (sc.scenario, sc.group, sc.benchmark)),
        );

    for (scenario, group, benchmark) in registered {
        if estimates(manifest, group, benchmark).exists() {
            continue;
        }
        // 同一用例已经测出、却不是登记的那个分组：判定按登记路径永远读不到，会一直登记跳过。
        let filed: Vec<&Path> = groups
            .iter()
            .filter(|dir| {
                dir.join(on_disk(benchmark))
                    .join("new/estimates.json")
                    .exists()
            })
            .map(|dir| dir.strip_prefix(&root).unwrap_or(dir))
            .collect();
        assert!(
            filed.is_empty(),
            "{scenario} 的测量落在 {filed:?}，登记的却是 `{group}`（落盘名 `{}`）：判定会永远跳过它",
            on_disk(group)
        );
    }
}
