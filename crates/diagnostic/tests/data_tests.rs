use super::*;

fn sample() -> Table {
    Table::from_pairs(vec![
        ("op", Value::table(vec![("name", Value::text("add_f32"))])),
        (
            "axes",
            Value::table(vec![("N", Value::array(vec![Value::int(64)]))]),
        ),
        (
            "variants",
            Value::array(vec![
                Value::table(vec![("name", Value::text("first"))]),
                Value::table(vec![("name", Value::text("second"))]),
            ]),
        ),
    ])
}

#[test]
fn from_pairs_keeps_the_given_order() {
    let table = sample();
    let names: Vec<&str> = table
        .entries()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["op", "axes", "variants"]);
    assert_eq!(table.at("op.name").and_then(as_str), Some("add_f32"));
}

#[test]
fn a_dotted_path_reaches_into_arrays() {
    let table = sample();
    assert_eq!(
        table.at("variants[1].name").and_then(as_str),
        Some("second"),
        "下标必须能选到数组里的那一项"
    );
    assert!(table.at("variants[2].name").is_none(), "越界就是 None");
}

#[test]
fn set_replaces_in_place_and_push_appends() {
    let mut table = sample();
    table.set("op.name", Value::text("changed"));
    assert_eq!(table.at("op.name").and_then(as_str), Some("changed"));
    let keys: Vec<&str> = table
        .entries()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(keys, ["op", "axes", "variants"], "替换不该改变位置");
    table.push(
        "variants",
        Value::table(vec![("name", Value::text("third"))]),
    );
    assert_eq!(table.at("variants[2].name").and_then(as_str), Some("third"));
}

#[test]
fn at_mut_edits_in_place() {
    let mut table = sample();
    if let Some(slot) = table.at_mut("axes.N[0]") {
        *slot = Value::int(1);
    }
    assert_eq!(table.at("axes.N[0]"), Some(&Value::Int(1)));
    assert!(table.at_mut("op.missing").is_none());
}

#[test]
fn remove_drops_the_key_without_leaving_a_hole() {
    let mut table = sample();
    assert!(table.remove("axes").is_some());
    assert!(!table.contains("axes"));
    assert_eq!(table.entries().len(), 2);
}

#[test]
fn a_path_index_on_a_non_array_or_a_malformed_index_is_none() {
    let table = sample();
    // 下标用在表上：表不是数组，越界形态直接 None，不panic。
    assert!(table.at("op[0].name").is_none());
    // 非数字下标不构成合法索引；数组上没有同名键，整条路径取空。
    assert!(table.at("variants[abc].name").is_none());
    // 空路径不是任何段。
    assert!(table.at("").is_none());
}

fn as_str(value: &Value) -> Option<&str> {
    match value {
        Value::Str(text) => Some(text),
        _ => None,
    }
}
