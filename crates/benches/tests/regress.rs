//! 把真实算子的实测值接进 `tool::regress` 的基线判定，并把元数据归档成一份可入 CI 的报告。
//!
//! 分工是刻意的：criterion 只负责测（预热、采样、区间估计），`tool::regress` 只负责判（方向、
//! 相对预算、"回退"与"改进"），本文件只负责**搬运**——从 criterion 的输出里取一个数，交给基线，
//! 再把结论与元数据写成报告。判定规则一条也不在这里重写，否则 CI 会各有一套。
//!
//! 三条"不可比"的处理：
//!
//! - **跨机不可比**：基线按 `os-arch` 登记（见 [`baseline_for`]）。没有本机基线时跳过，而不是
//!   拿别的机器的数判——那会把"A 机的单核速度"当成"这台机器的验收线"。
//! - **跨口径不可比**：判的是 `ns/iter` 而不是 FLOPs 或 GiB/s。`gemv` 的二进制位宽随权重格式
//!   变（f32 / i4 / 打包），同一形状的理论算力能差 4 倍；时间才是可比的量。
//! - **跨版本不可比**：基线与实测必须在同一工具链下取；换了编译器就重新登记（报告里记了架构）。
//!
//! 被测面在 [`SCENARIOS`] 表里逐场景声明：criterion 分组 / 用例名与 `criterion/*.rs` 逐字一致，
//! 归档元数据（算子、形状、批大小、dtype）也从这张表取，新增一条基准只加一行，不复制判定流程。
//!
//! 归档在 `target/bench-report/<host>/<group>.json`，由 CI 的 bench job 收成 artifact。
//! 报告里所有数值都是**整数**（ns 取整、预算取整百分比）：仓库唯一的 JSON 解析器只认整数，
//! 于是这份归档可以用它自检，而不是"看起来像 JSON"。

use std::path::{Path, PathBuf};
use tool::regress::{Case, Direction, Judgement};

/// 一条回归基线场景：criterion 被测面 + 归档元数据。
///
/// 基线数值不写在结构里（基线随主机变化），由 [`baseline_for`] 按 `(host, scenario)` 查
/// [`BASELINES`]；结构只描述"测哪条基准、报告怎么写"。
struct Scenario {
    /// criterion 分组名（`target/criterion/<group>/...`）。
    group: &'static str,
    /// criterion 用例名（`.../<benchmark>/new/estimates.json`）。
    benchmark: &'static str,
    /// 场景名：`<算子>/<形状>`，同时是 regress `Case` 名与归档口径的一部分。
    scenario: &'static str,
    /// 算子标识（归档报告的 `op` 字段）。
    op: &'static str,
    /// 形状（归档报告的 `shape` 字段）。
    shape: &'static [usize],
    /// 元素类型（归档报告的 `dtype` 字段）。
    dtype: &'static str,
    /// 批大小：1 行 = 一个 decode 步，与 prefill 的多行批差三个数量级。
    batch: u32,
}

/// 全部接入基线判定的真实算子基准；分组 / 用例名与 `criterion/*.rs` 逐字一致。
const SCENARIOS: &[Scenario] = &[
    Scenario {
        group: "gemv_f32",
        benchmark: "decode_1x4096x4096",
        scenario: "gemv_f32/decode_1x4096x4096",
        op: "op::gemv_f32",
        shape: &[4096, 4096],
        dtype: "f32",
        batch: 1,
    },
    Scenario {
        group: "softmax_f32",
        benchmark: "decode_1x4096",
        scenario: "softmax_f32/decode_1x4096",
        op: "op::softmax_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
    },
    Scenario {
        group: "rope_f32",
        benchmark: "decode_1x4096",
        scenario: "rope_f32/decode_1x4096",
        op: "op::rope_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
    },
    Scenario {
        group: "rmsnorm_f32",
        benchmark: "decode_1x4096",
        scenario: "rmsnorm_f32/decode_1x4096",
        op: "op::rmsnorm_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
    },
    Scenario {
        group: "attention_f32",
        benchmark: "flash_decode_128x64",
        scenario: "attention_f32/flash_decode_128x64",
        op: "op::attention",
        shape: &[128, 64],
        dtype: "f32",
        batch: 1,
    },
    Scenario {
        group: "gemm_f32",
        benchmark: "prefill_256x256_k8",
        scenario: "gemm_f32/prefill_256x256_k8",
        op: "op::gemm_f32",
        shape: &[256, 256],
        dtype: "f32",
        batch: 1,
    },
    Scenario {
        group: "silu_f32",
        benchmark: "decode_1x4096",
        scenario: "silu_f32/decode_1x4096",
        op: "op::silu_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
    },
];

/// 各主机登记基线：`(host, scenario, ns/iter, 允许的相对偏差%)`。
///
/// 数值来自 `cargo bench -p llm-benches --bench <group>` 的 mean 点估计，取整到 ns。
/// 新增机器或新算子就在这里加一行，并在注释里写清机型与日期——"基线"必须能追溯到一次真实测量。
const BASELINES: &[(&str, &str, f64, f64)] = &[
    // M4 Pro（2026-11-15，reference 实现 BLOCK=32）：2186061 ns / 100 次采样。
    (
        "macos-aarch64",
        "gemv_f32/decode_1x4096x4096",
        2_186_061.0,
        5.0,
    ),
    // M4 Pro（2026-11-16，三趟标量参考）：15492 ns / 100 次采样；噪声相对均值约 4.5%，预算取 10%。
    ("macos-aarch64", "softmax_f32/decode_1x4096", 15_492.0, 10.0),
    // M4 Pro（2026-11-16，成对旋转参考）：9487 ns / 100 次采样；噪声相对均值约 6.4%，预算取 10%。
    ("macos-aarch64", "rope_f32/decode_1x4096", 9_487.0, 10.0),
    // M4 Pro（2026-11-16，两趟标量参考）：2374 ns / 100 次采样；噪声相对均值约 1.5%，预算取 10%。
    ("macos-aarch64", "rmsnorm_f32/decode_1x4096", 2_374.0, 10.0),
    // M4 Pro（2026-11-16，online flash 参考 128×64）：517442 ns / 100 次采样；噪声相对均值约 1.5%，预算取 10%。
    (
        "macos-aarch64",
        "attention_f32/flash_decode_128x64",
        517_442.0,
        10.0,
    ),
    // M4 Pro（2026-11-16，三重循环参考 256×256，K=8）：16599 ns / 100 次采样；噪声相对均值约 1.5%，预算取 10%。
    (
        "macos-aarch64",
        "gemm_f32/prefill_256x256_k8",
        16_599.0,
        10.0,
    ),
    // M4 Pro（2026-11-16，逐元素 SiLU 参考）：3287 ns / 100 次采样；噪声相对均值约 1.5%，预算取 10%。
    ("macos-aarch64", "silu_f32/decode_1x4096", 3_287.0, 10.0),
];

/// 本机标识：`os-arch`。criterion 的绝对值只在同一台机器上可比，标识也就到这一层。
fn host() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// 查某主机某场景的登记基线 `(ns/iter, 预算%)`；未登记返回 `None`。
fn baseline_for(host: &str, scenario: &str) -> Option<(f64, f64)> {
    BASELINES
        .iter()
        .find(|(h, s, _, _)| *h == host && *s == scenario)
        .map(|(_, _, ns, budget)| (*ns, *budget))
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

/// criterion 的输出目录：`target/criterion/<group>/<benchmark>/new/estimates.json`。
///
/// 目标目录在**工作区根**（`crates/benches/../..`），不在本成员的目录下：criterion 由
/// `cargo bench` 调用，写的是调用方的 `target/`。
fn estimates(manifest: &Path, scenario: &Scenario) -> PathBuf {
    manifest
        .join("../..")
        .join("target/criterion")
        .join(scenario.group)
        .join(scenario.benchmark)
        .join("new/estimates.json")
}

/// 实测值；还没跑过基准就登记跳过并返回 `None`。
fn observed_ns(path: &Path) -> Option<f64> {
    let Ok(text) = std::fs::read_to_string(path) else {
        testkit::skip(format!(
            "benches: 还没有基准测量（{}），先跑 cargo bench -p llm-benches",
            path.display()
        ));
        return None;
    };
    Some(mean_ns(&text).unwrap_or_else(|| {
        panic!(
            "读不出 {} 的 mean.point_estimate：criterion 的输出格式变了，判定口径要跟着改",
            path.display()
        )
    }))
}

/// 一份归档报告：元数据 + 实测 + 结论。
struct Report<'a> {
    scenario: &'a Scenario,
    host: String,
    verdict: &'static str,
    observed_ns: f64,
    baseline_ns: f64,
    budget_percent: f64,
}

/// 把形状渲染成 JSON 整数数组：`[4096, 4096]` / `[4096]`。
fn shape_json(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|dim| dim.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

impl<'a> Report<'a> {
    /// 渲染成整数数值的 JSON（见模块文档最后一段）。
    fn render(&self) -> String {
        let scenario = self.scenario;
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"{}\",\n  \"op\": \"{}\",\n  \
             \"shape\": {},\n  \"batch\": {},\n  \"dtype\": \"{}\",\n  \
             \"backend\": \"cpu-reference\",\n  \"metric\": \"ns_per_iter\",\n  \
             \"direction\": \"{}\",\n  \"budget_percent\": {},\n  \"baseline_ns\": {},\n  \"observed_ns\": {},\n  \"verdict\": \"{}\",\n  \"host\": \"{}\",\n  \"profile\": \"bench\",\n  \"arch\": \"{}\"\n}}\n",
            scenario.group,
            scenario.benchmark,
            scenario.scenario,
            scenario.op,
            shape_json(scenario.shape),
            scenario.batch,
            scenario.dtype,
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

/// 判定单个场景：建 `Case`、比对实测、归档报告；回退则 panic。
fn judge_one(manifest: &Path, host: &str, scenario: &Scenario) -> Option<()> {
    let (baseline_ns, budget) = baseline_for(host, scenario.scenario)?;
    let observed = observed_ns(&estimates(manifest, scenario))?;

    let case =
        Case::new(scenario.scenario, baseline_ns, budget, Direction::Lower).expect("基线合法");
    let judgement = case.judge(observed);

    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let report = Report {
        scenario,
        host: host.to_string(),
        verdict: judgement.as_str(),
        observed_ns: observed,
        baseline_ns,
        budget_percent: budget,
    };
    let rendered = report.render();
    // 自检：归档必须是能被仓库唯一 JSON 解析器读懂的文档（整数约定见模块文档）。
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    std::fs::write(dir.join(format!("{}.json", scenario.group)), &rendered).expect("归档可写");

    if judgement.is_pass() {
        return Some(());
    }
    panic!(
        "{} 回退：实测 {observed:.0} ns，基线 {baseline_ns:.0} ns，预算 {budget:.0}%（上限 {:.0} ns）。\
         先确认不是这台机器在忙；同一口径下确实回退了，要么修，要么按新测量重新登记基线",
        scenario.scenario,
        case.limit()
    );
}

#[test]
fn registered_baselines_stay_within_their_budget() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();

    let ready: Vec<&Scenario> = SCENARIOS
        .iter()
        .filter(|sc| baseline_for(&host, sc.scenario).is_some())
        .collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benches: 主机 `{host}` 没有登记任何基线（跨机绝对值不可比），先在靶机上跑 cargo bench 后登记"
        ));
        return;
    }

    for scenario in ready {
        if judge_one(manifest, &host, scenario).is_none() {
            return;
        }
    }
}

#[test]
fn the_report_carries_every_field_a_reader_needs_to_judge_it() {
    for scenario in SCENARIOS {
        let report = Report {
            scenario,
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
            Some(scenario.batch as i64),
            "批大小是口径的一部分：1 行 = decode 步，与 prefill 的多行批差三个数量级"
        );
    }
}
