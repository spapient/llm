use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn emit_covers_every_value_shape() {
    let table: toml::Table = toml::from_str(
        "[head]\nname = \"add\"\nrank = 7\nratio = 0.5\non = true\n\
         tags = [\"a\", \"b\"]\nnested = { k = 1 }",
    )
    .unwrap();
    let out = emit(&table, "root").unwrap();
    assert!(out.starts_with("Table::from_pairs(vec!["));
    assert!(out.contains("Value::text(\"add\")"));
    assert!(out.contains("Value::int(7)"));
    assert!(out.contains("Value::real(0.5)"));
    assert!(out.contains("Value::flag(true)"));
    assert!(out.contains("Value::array(vec!["));
    assert!(out.contains("Value::table(vec!["));
    assert!(out.contains("Value::int(1)"));
}

#[test]
fn a_non_finite_float_or_datetime_is_rejected_with_path() {
    let nan: toml::Table = toml::from_str("x = nan").unwrap();
    let err = emit(&nan, "bad").unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.message().contains("bad.x") && err.message().contains("有限浮点"));

    let date: toml::Table = toml::from_str("x = 2026-01-01").unwrap();
    let err = emit(&date, "bad").unwrap_err();
    assert!(err.message().contains("bad.x") && err.message().contains("日期"));
}

#[test]
fn collect_sorts_and_filters_toml_files() {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("b.toml"), "x = 1").unwrap();
    std::fs::write(dir.join("a.toml"), "x = 2").unwrap();
    std::fs::write(dir.join("note.txt"), "ignore").unwrap();
    std::fs::create_dir(dir.join("sub.toml")).unwrap();

    let names: Vec<String> = collect(&dir)
        .unwrap()
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["a.toml", "b.toml"]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_accepts_a_table_and_rejects_a_parse_error() {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let ok = dir.join("ok.toml");
    std::fs::write(&ok, "x = 1").unwrap();
    assert_eq!(read(&ok).unwrap()["x"].as_integer(), Some(1));

    let broken = dir.join("broken.toml");
    std::fs::write(&broken, "x = ").unwrap();
    let err = read(&broken).unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.message().contains("解析"));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn relative_strips_the_base_prefix() {
    let base = Path::new("/repo/crates/op");
    let path = base.join("ops/add.toml");
    assert_eq!(relative(base, &path), "ops/add.toml");
}

/// 以纳秒时间戳命名的唯一临时目录，避免并行测试撞名。
fn temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("buildgen-{nanos}"))
}
