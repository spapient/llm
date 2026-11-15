use diagnostic::data::{Table, Value};

use super::*;

/// 一份只有阈值的 `[gate]`：行数上限故意收紧到 10，便于用短文本验证边界。
fn gate() -> Gate {
    let thresholds = vec![
        ("ccn_max", Value::int(5)),
        ("nloc_max", Value::int(60)),
        ("file_max_lines", Value::int(10)),
    ];
    Gate::from_table(&Table::from_pairs(vec![("gate", Value::table(thresholds))]))
        .expect("测试阈值合法")
}

#[test]
fn an_oversized_file_is_flagged() {
    let text = (0..12).map(|_| "x").collect::<Vec<_>>().join("\n");
    assert_eq!(file_lines(&text, &gate()), Some(12));
}

#[test]
fn a_file_within_budget_is_silent() {
    let text = (0..8).map(|_| "x").collect::<Vec<_>>().join("\n");
    assert_eq!(file_lines(&text, &gate()), None);
}
