//! 把真实算子的实测值接进 `tool::regress` 的基线判定，并把元数据归档成一份可入 CI 的报告。
//!
//! 分工是刻意的：criterion 只负责测（预热、采样、区间估计），`tool::regress` 只负责判（方向、
//! 相对预算、"回退"与"改进"），本文件只负责**搬运**——从 criterion 的输出里取一个数，交给基线，
//! 再把结论与元数据写成报告。判定规则一条也不在这里重写，否则 CI 会各有一套。
//!
//! 三个"不可比"的处理：
//!
//! - **跨机不可比**：基线按 `os-arch` 登记（见 [`baseline_for`]）。没有本机基线时跳过，而不是
//!   拿别的机器的数判——那会把"A 机的单核速度"当成"这台机器的验收线"。
//! - **跨口径不可比**：判的是 `ns/iter` 而不是 FLOPs 或 GiB/s。`gemv` 的二进制位宽随权重格式
//!   变（f32 / i4 / 打包），同一形状的理论算力能差 4 倍；时间才是可比的量。
//! - **跨版本不可比**：基线与实测必须在同一工具链下取；换了编译器就重新登记（报告里记了架构）。
//!
//! 归档在 `target/bench-report/<host>/gemv_f32.json`，由 CI 的 bench job 收成 artifact。
//! 报告里所有数值都是**整数**（ns 取整、预算取整百分比）：仓库唯一的 JSON 解析器只认整数，
//! 于是这份归档可以用它自检，而不是"看起来像 JSON"。

use std::path::{Path, PathBuf};
use tool::regress::{Baseline, Case, Direction, Judgement};

/// 被测面：与 `criterion/gemv_f32.rs` 的分组名 / 用例名逐字一致。
const GROUP: &str = "gemv_f32";
const BENCHMARK: &str = "decode_1x4096x4096";

/// 场景名：`<算子>/<形状>`，同时是归档文件名的一部分。
const SCENARIO: &str = "gemv_f32/decode_1x4096x4096";

/// 一台机器的登记基线：`(ns/iter, 允许的相对偏差百分比)`。
///
/// 数值来自 `cargo bench -p llm-benches --bench gemv_f32` 的 mean 点估计，取整到 ns。
/// 新增机器就在这里加一行，并在注释里写清机型与日期——"基线"必须能追溯到一次真实测量。
fn baseline_for(host: &str) -> Option<(f64, f64)> {
    match host {
        // M4 Pro（2026-11-15，reference 实现 BLOCK=32）：2186061 ns / 100 次采样。
        "macos-aarch64" => Some((2_186_061.0, 5.0)),
        _ => None,
    }
}

/// 本机标识：`os-arch`。criterion 的绝对值只在同一台机器上可比，标识也就到这一层。
fn host() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// criterion 的 mean 点估计（`ns/iter`）。
///
/// 只依赖 `estimates.json` 里两个键名（`mean` 与 `point_estimate`）：criterion 的统计口径演进过，
/// 但这两个名字是它对外报告的核心。读不出来就明说"读不出"，不猜一个数。
fn mean_ns(text: &str) -> Option<f64> {
    let mean = text.find("\"mean\"")?;
    let key = text[mean..].find("\"point_estimate\"")?;
    let tail = &text[mean + key + "\"point_estimate\"".len()..];
    let start = tail.find(|c: char| c == '-' || c.is_ascii_digit())?;
    let number: String = tail[start..]
        .chars()
        .take_while(|c| {
            *c == '-' || *c == '.' || c.is_ascii_digit() || *c == 'e' || *c == '+' || *c == 'E'
        })
        .collect();
    number.parse().ok()
}

/// criterion 的输出目录：`target/criterion/<group>/<bench>/new/estimates.json`。
///
/// 目标目录在**工作区根**（`crates/benches/../..`），不在本成员的目录下：criterion 由
/// `cargo bench` 调用，写的是调用方的 `target/`。
fn estimates(manifest: &Path) -> PathBuf {
    manifest
        .join("../..")
        .join("target/criterion")
        .join(GROUP)
        .join(BENCHMARK)
        .join("new/estimates.json")
}

/// 本机基线；没有登记就登记一次跳过并返回 `None`。
fn local_baseline(host: &str) -> Option<(f64, f64)> {
    if let Some(pair) = baseline_for(host) {
        return Some(pair);
    }
    testkit::skip(format!(
        "benches: 主机 `{host}` 没有登记 gemv_f32 基线（跨机绝对值不可比），先在靶机上跑 cargo bench 后登记"
    ));
    None
}

/// 实测值；还没跑过基准就登记跳过并返回 `None`。
fn observed_ns(path: &Path) -> Option<f64> {
    let Ok(text) = std::fs::read_to_string(path) else {
        testkit::skip(format!(
            "benches: 还没有 gemv_f32 的测量（{}），先跑 cargo bench -p llm-benches --bench gemv_f32",
            path.display()
        ));
        return None;
    };
    Some(mean_ns(&text).unwrap_or_else(|| {
        panic!(
            "读不出 {GROUP}/{BENCHMARK}/new/estimates.json 的 mean.point_estimate：\
             criterion 的输出格式变了，判定口径要跟着改"
        )
    }))
}

/// 一份归档报告：元数据 + 实测 + 结论。
struct Report {
    host: String,
    verdict: &'static str,
    observed_ns: f64,
    baseline_ns: f64,
    budget_percent: f64,
}

impl Report {
    /// 渲染成整数数值的 JSON（见模块文档最后一段）。
    fn render(&self) -> String {
        format!(
            "{{\n  \"surface\": \"{GROUP}\",\n  \"benchmark\": \"{BENCHMARK}\",\n  \
             \"scenario\": \"{SCENARIO}\",\n  \"op\": \"op::gemv_f32\",\n  \
             \"shape\": [4096, 4096],\n  \"batch\": 1,\n  \"dtype\": \"f32\",\n  \
             \"backend\": \"cpu-reference\",\n  \"metric\": \"ns_per_iter\",\n  \
             \"direction\": \"{}\",\n  \"budget_percent\": {},\n  \
             \"baseline_ns\": {},\n  \"observed_ns\": {},\n  \
             \"verdict\": \"{}\",\n  \"host\": \"{}\",\n  \
             \"profile\": \"bench\",\n  \"arch\": \"{}\"\n}}\n",
            Direction::Lower.as_str(),
            self.budget_percent.round() as i64,
            self.baseline_ns.round() as i64,
            self.observed_ns.round() as i64,
            self.verdict,
            self.host,
            std::env::consts::ARCH,
        )
    }
}

#[test]
fn the_decode_gemv_stays_within_its_registered_baseline() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let (baseline_ns, budget, observed) =
        match (local_baseline(&host), observed_ns(&estimates(manifest))) {
            (Some((baseline_ns, budget)), Some(observed)) => (baseline_ns, budget, observed),
            _ => return,
        };

    let case = Case::new(SCENARIO, baseline_ns, budget, Direction::Lower).expect("基线合法");
    let baseline = Baseline::new(vec![case.clone()]).expect("单场景基线");
    let judgement = baseline.judge(SCENARIO, observed).expect("场景已登记");

    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(&host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let report = Report {
        host: host.clone(),
        verdict: judgement.as_str(),
        observed_ns: observed,
        baseline_ns,
        budget_percent: budget,
    };
    let rendered = report.render();
    // 自检：归档必须是能被仓库唯一 JSON 解析器读懂的文档（整数约定见模块文档）。
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    std::fs::write(dir.join(format!("{GROUP}.json")), &rendered).expect("归档可写");

    if judgement.is_pass() {
        return;
    }
    panic!(
        "gemv_f32 回退：实测 {observed:.0} ns，基线 {baseline_ns:.0} ns，预算 {budget:.0}%（上限 {:.0} ns）。\
         先确认不是这台机器在忙；同一口径下确实回退了，要么修，要么按新测量重新登记基线",
        case.limit()
    );
}

#[test]
fn the_report_carries_every_field_a_reader_needs_to_judge_it() {
    let report = Report {
        host: "linux-x86_64".to_string(),
        verdict: Judgement::Improved {
            observed: 1.0,
            baseline: 2.0,
        }
        .as_str(),
        observed_ns: 1_500_000.0,
        baseline_ns: 2_000_000.0,
        budget_percent: 5.0,
    };
    let rendered = report.render();
    for field in [
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
    ] {
        assert!(
            rendered.contains(field),
            "报告缺少 {field}：读者无法自己判断结论"
        );
    }
    let parsed = weights::json::parse(&rendered).expect("合法 JSON");
    assert_eq!(
        parsed.get("batch").and_then(|value| value.integer().ok()),
        Some(1),
        "批大小是口径的一部分：1 行 = decode 步，与 prefill 的 4096 行差三个数量级"
    );
}
