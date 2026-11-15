use diagnostic::data::Value;

use super::*;

/// 本仓库的三个阈值：改动它们就改 `lints/vocabulary.toml`。
const SOUND: [i64; 3] = [5, 60, 899];

/// 一份与仓库同阈值、没有存量债的 `[gate]` 表。
fn sound() -> Table {
    let gate = THRESHOLDS
        .iter()
        .zip(SOUND)
        .map(|(key, value)| (*key, Value::int(value)))
        .collect();
    Table::from_pairs(vec![("gate", Value::table(gate))])
}

/// 项目配置：三个阈值来自构建期生成的 `lints/vocabulary.toml`；存量债务只允许挂在根入口
/// `src/`（待下沉 `tool` 的既有编排/排版逻辑），运行时生产代码 `crates/*/src` 一条都不能有。
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
    for key in gate.debt.keys() {
        assert!(
            key.starts_with("src/"),
            "运行时生产代码没有债务条目，存量只能是根入口既有逻辑：{key}"
        );
    }
}

#[test]
fn every_threshold_must_be_positive() {
    for key in THRESHOLDS {
        let mut config = sound();
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
    let mut config = sound();
    config.set("gate.simd_nloc", Value::int(96));
    assert_eq!(
        Gate::from_table(&config).unwrap_err().message(),
        "[gate] 未知键 simd_nloc",
        "从未被读到的档位必须报错，而不是静默失效"
    );
}

#[test]
fn a_threshold_must_be_a_positive_integer() {
    let mut config = sound();
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
    let mut config = sound();
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
    let mut config = sound();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(9)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let over = Over {
        key: "crates/x/src/y.rs:run#0".into(),
        metric: Metric::Ccn,
        value: 9,
    };
    assert!(gate.check(&over).is_none(), "登记范围内应通过");
    assert!(
        gate.check(&Over { value: 10, ..over }).is_some(),
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
    let mut config = sound();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(47)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let over = Over {
        key: "crates/x/src/y.rs:run#0".into(),
        metric: Metric::Ccn,
        value: 6,
    };
    assert_eq!(
        gate.check(&over).unwrap(),
        "crates/x/src/y.rs:run#0: Ccn 实测 6 < 登记 47：把基线条目改成 6",
        "顶着 47 的登记值活着的 ccn=6 必须被要求记账"
    );
}

/// 同一身份登记 ccn 与 nloc：只观测到 ccn 仍超限时，nloc 那条必须被要求删除。
///
/// 这是旧实现的实际漏洞：按身份判断时，`live` 只要含这个 key，两条都算「仍有超标」，
/// 于是改好的那半永远留在表里。
#[test]
fn a_satisfied_metric_must_drop_even_when_the_identity_is_still_live() {
    let mut config = sound();
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
///
/// 登记值由本地表给出：项目基线清零后，仓库里没有可以借用的登记值。
#[test]
fn ratchet_blocks_new_and_worse_allows_within_budget() {
    let mut config = sound();
    config.push(
        "gate.debt",
        Value::table(vec![
            ("key", Value::text("crates/x/src/y.rs:run#0")),
            ("ccn", Value::int(9)),
        ]),
    );
    let gate = Gate::from_table(&config).expect("表合法");
    let over = Over {
        key: "crates/x/src/y.rs:run#0".into(),
        metric: Metric::Ccn,
        value: 10,
    };
    assert!(gate.check(&over).is_some(), "恶化必须失败");
    assert!(
        gate.check(&Over { value: 9, ..over }).is_none(),
        "登记范围内应通过（此时仍超阈值）"
    );
    assert!(
        gate.check(&Over {
            key: "crates/x/src/y.rs:new#0".into(),
            ..over
        })
        .is_some(),
        "全新身份直接失败"
    );
}
