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

fn as_str(value: &Value) -> Option<&str> {
    match value {
        Value::Str(text) => Some(text),
        _ => None,
    }
}
