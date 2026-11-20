//! 把真实一步 decode 的实测样本接进 `tool::card` 的三态判定，并归档成一份可复核的卡片。
//!
//! 分工仍是刻意的：本文件只负责**测**（合成小栈走 [`Predict::logits`] 一步，逐窗口取墙钟）
//! 与**搬运**（样本 → [`Card::judge`] → 归档）；判什么量、门槛多少、区间怎么算全在
//! `tool::card`，这里不写第二份判定规则。
//!
//! 与 `tests/regress.rs` 的分工：那一边判「相对基线回退了没有」，需要 criterion 的区间估计
//! 与登记基线；这一边判「这份协议卡这一格成立了没有」，只有绝对门槛与一列样本，所以不需要
//! criterion，也不需要本机基线——`lld` 的 Must 本来就是绝对值。
//!
//! 断言只落在**与时间无关**的性质上：同一列样本换成不可达门槛必然 FAIL、换成极宽门槛必然
//! PASS、样本数不到下限必然 OPEN、主机不在闭集里必然 OPEN。真实的 `E10`（变异系数 < 5%）
//! 照 `lld` 的口径判出来归档，但不在测试里断言它通过——这台机器当时抖不抖是测量结果，
//! 不是这个测试的前提。

#[path = "../support/model.rs"]
mod support;

use std::hint::black_box;
use std::time::Instant;

use engine::exec::Stream;
use engine::kv::{Cache, Pager};
use engine::{Batch, Sequence};
use serve::Predict;
use serve::model::CpuModel;
use support::PROMPT;
use tool::bench::Harness;
use tool::card::{Card, Direction, Protocol, Shortfall, Standing, Statistic};

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
    let model = support::synthetic_model();

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

/// 本机标识：`os-arch`，与 `tests/regress.rs` 的归档目录同口径。
fn host() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
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
        Direction::Lower,
        0.05,
    )
    .expect("E10 协议")
    .with_least(WINDOWS)
    .expect("样本下限")
}

/// 真实一列样本落成一张卡，并归档到 `target/bench-report/<host>/card-<protocol>.json`。
fn archive(card: &Card) -> std::path::PathBuf {
    let directory = std::path::Path::new("target")
        .join("bench-report")
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
    let reason = standing.shortfall().map(Shortfall::as_str).unwrap_or("met");
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
        Direction::Lower,
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
        Direction::Lower,
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
        Direction::Lower,
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
