//! Gate 结构本身：阈值解析、闭集键校验、棘轮结清。

use std::collections::{BTreeMap, BTreeSet};

use diagnostic::data::{Table, Value};
use llm_tools::gate::{Gate, Metric, Over};

use super::scan_integration_test;

/// 本仓库五个阈值的键与值；改动它们就改 `lints/vocabulary.toml`。
const GATE_KEYS: [(&str, i64); 5] = [
    ("ccn_max", 5),
    ("nloc_max", 60),
    ("file_max_lines", 899),
    ("file_min_lines", 300),
    ("test_max_lines", 800),
];

/// 一份与仓库同阈值、没有存量债的 `[gate]` 表。
fn sound_gate() -> Table {
    let gate = GATE_KEYS
        .iter()
        .map(|(key, value)| (*key, Value::int(*value)))
        .collect();
    Table::from_pairs(vec![("gate", Value::table(gate))])
}

/// 造一张只含一条 ccn 债务的 gate 与一份实测超限：登记值 `ccn`、实测 `measured`。
fn debt_gate(ccn: i64, measured: usize) -> (Gate, Over) {
    let mut config = sound_gate();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(ccn)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let over = Over {
        key: "crates/x/src/y.rs:run#0".into(),
        metric: Metric::Ccn,
        value: measured,
    };
    (gate, over)
}

/// 项目配置：三个阈值来自构建期生成的 `lints/vocabulary.toml`；存量债务只允许挂在入口编排层
///（`crates/cli/src/` 下的编排与二进制入口），运行时生产代码 `crates/*/src` 一条都不能有。
#[test]
fn the_project_config_lists_only_root_entry_legacy_debt() {
    let gate = Gate::load().expect("项目配置合法");
    let limits =
        [Metric::Ccn, Metric::Nloc, Metric::Lines, Metric::Call].map(|metric| gate.limit(metric));
    assert_eq!(
        limits,
        [5, 60, 899, 0],
        "阈值来自构建期生成的 `lints/vocabulary.toml`；`Call` 没有预算"
    );
    for msg in gate.stale(&BTreeMap::new()) {
        assert!(
            msg.starts_with("crates/cli/src/"),
            "运行时生产代码没有债务条目，存量只能是入口编排既有逻辑：{msg}"
        );
    }
}

#[test]
fn a_test_boulder_over_the_limit_must_fail() {
    let gate = Gate::load().expect("阈值配置加载");
    let limit = gate.test_limit();
    let dir = std::env::temp_dir().join(format!("llm-gate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("造临时目录");
    let path = dir.join("boulder.rs");
    // 每行都是一个合法空行：本用例量的是行数棘轮，不是词法。
    let body = "\n".repeat(limit + 1);
    std::fs::write(&path, &body).expect("写临时文件");

    let mut bad = Vec::new();
    scan_integration_test(&gate, &dir, &path, &mut bad);
    assert_eq!(bad.len(), 1, "超过 {limit} 行必须恰好报一条：{bad:?}");
    assert!(bad[0].contains("超过上限"), "{bad:?}");

    std::fs::write(&path, "\n".repeat(limit)).expect("重写临时文件");
    let mut ok = Vec::new();
    scan_integration_test(&gate, &dir, &path, &mut ok);
    assert!(ok.is_empty(), "等于上限不该报：{ok:?}");
    std::fs::remove_dir_all(&dir).expect("清理临时目录");
}

#[test]
fn every_threshold_must_be_positive() {
    for (key, _) in GATE_KEYS {
        let mut config = sound_gate();
        config.set(&format!("gate.{key}"), Value::int(0));
        assert_eq!(
            Gate::from_table(&config).unwrap_err().message(),
            "[gate] 阈值必须为正",
            "`{key} = 0` 必须被拒"
        );
    }
}

#[test]
fn an_unknown_gate_key_is_an_error_not_noise() {
    let mut config = sound_gate();
    config.set("gate.simd_nloc", Value::int(96));
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate] 未知键 simd_nloc",
        "从未被读到的档位必须报错，而不是静默失效"
    );
}

#[test]
fn a_threshold_must_be_a_positive_integer() {
    let mut config = sound_gate();
    config.set("gate.ccn_max", Value::text("5"));
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate] 缺正整数 ccn_max"
    );
}

#[test]
fn a_vocabulary_without_a_gate_section_is_rejected() {
    assert_eq!(
        Gate::from_table(&Table::new()).unwrap_err().message(),
        "vocabulary.toml 缺 [gate]"
    );
}

#[test]
fn a_debt_row_without_a_metric_is_rejected() {
    let mut config = sound_gate();
    config.push(
        "gate.debt",
        Value::table(vec![("key", Value::text("crates/x/src/y.rs:run#0"))]),
    );
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate.debt] crates/x/src/y.rs:run#0 无度量"
    );
}

#[test]
fn a_debt_row_registers_only_the_metrics_it_names() {
    let (gate, over) = debt_gate(9, 9);
    assert!(gate.reject_reason(&over).is_none(), "登记范围内应通过");
    assert!(
        gate.reject_reason(&Over { value: 10, ..over }).is_some(),
        "超过登记值应失败"
    );
    assert_eq!(
        gate.stale(&BTreeMap::new()).len(),
        1,
        "未观测到的条目应提示删除"
    );
}

/// 登记值高于实测即失败：基线是快照，不是预算，改善必须在同一提交里记账。
#[test]
fn a_register_above_the_measurement_must_be_lowered() {
    let (gate, over) = debt_gate(47, 6);
    assert_eq!(
        gate.reject_reason(&over).unwrap(),
        "crates/x/src/y.rs:run#0: Ccn 实测 6 < 登记 47：把基线条目改成 6",
        "顶着 47 的登记值活着的 ccn=6 必须被要求记账"
    );
}

/// 同一身份登记 ccn 与 nloc：只观测到 ccn 仍超限时，nloc 那条必须被要求删除。
#[test]
fn a_satisfied_metric_must_drop_even_when_the_identity_is_still_live() {
    let mut config = sound_gate();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(9)),
            ("nloc", Value::int(90)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let live = BTreeMap::from([(
        "crates/x/src/y.rs:run#0".to_string(),
        BTreeSet::from([Metric::Ccn]),
    )]);
    assert_eq!(
        gate.stale(&live),
        vec!["crates/x/src/y.rs:run#0: Nloc 已达标，删除基线条目".to_string()],
        "身份仍在册不能让已达标的那条度量继续豁免"
    );
    assert_eq!(
        gate.stale(&BTreeMap::new()).len(),
        2,
        "一条都没观测到就该全删"
    );
}

/// 三种判定：恶化失败、登记范围内通过、全新身份失败。
#[test]
fn ratchet_blocks_new_and_worse_allows_within_budget() {
    let (gate, over) = debt_gate(9, 10);
    assert!(gate.reject_reason(&over).is_some(), "恶化必须失败");
    assert!(
        gate.reject_reason(&Over { value: 9, ..over }).is_none(),
        "登记范围内应通过（此时仍超阈值）"
    );
    assert!(
        gate.reject_reason(&Over {
            key: "crates/x/src/y.rs:new#0".into(),
            ..over
        })
        .is_some(),
        "全新身份直接失败"
    );
}
