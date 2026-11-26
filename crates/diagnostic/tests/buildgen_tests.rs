use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

testkit::test_cases! {
fn read_preserves_every_toml_value_shape() {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("all.toml");
    std::fs::write(
        &path,
        "[head]\nname = \"add\"\nrank = 7\nratio = 0.5\non = true\n\
         tags = [\"a\", \"b\"]\nnested = { k = 1 }\nday = 2026-01-01\n",
    )
    .unwrap();
    let root = read(&path).unwrap();
    let head = root["head"].as_table().unwrap();
    assert_eq!(head["name"].as_str(), Some("add"));
    assert_eq!(head["rank"].as_integer(), Some(7));
    assert_eq!(head["ratio"].as_float(), Some(0.5));
    assert_eq!(head["on"].as_bool(), Some(true));
    assert_eq!(head["tags"].as_array().unwrap().len(), 2);
    assert_eq!(head["nested"]["k"].as_integer(), Some(1));
    assert!(head["day"].as_datetime().is_some());
    std::fs::remove_dir_all(&dir).unwrap();
}

fn read_preserves_non_finite_numbers_for_domain_validation() {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("float.toml");
    std::fs::write(&path, "nan = nan\ninf = inf\n").unwrap();
    let table = read(&path).unwrap();
    assert!(table["nan"].as_float().unwrap().is_nan());
    assert!(table["inf"].as_float().unwrap().is_infinite());
    std::fs::remove_dir_all(&dir).unwrap();
}

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

fn relative_strips_the_base_prefix() {
    let base = Path::new("/repo/crates/prof");
    let path = base.join("ops/add.toml");
    assert_eq!(relative(base, &path), "ops/add.toml");
}

fn file_stem_and_extension_read_utf8_name() {
    let path = Path::new("ops/add.toml");
    assert_eq!(file_stem(path), Some("add"));
    assert_eq!(extension(path).as_deref(), Some("toml"));

    let bare = Path::new("readme");
    assert_eq!(extension(bare), None);
}

fn merge_tables_nests_recursively_and_replaces_arrays() {
    let mut base = toml::from_str::<toml::Table>("[a]\nx = 1\ny = 2\ntags = [\"old\"]\n").unwrap();
    let overlay = toml::from_str::<toml::Table>("[a]\ny = 9\ntags = [\"new\"]\nz = 3\n").unwrap();

    merge_tables(&mut base, overlay);

    let nested = base["a"].as_table().unwrap();
    assert_eq!(nested["x"].as_integer(), Some(1));
    assert_eq!(nested["y"].as_integer(), Some(9));
    assert_eq!(nested["z"].as_integer(), Some(3));
    assert_eq!(
        nested["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(toml::Value::as_str)
            .collect::<Vec<_>>(),
        vec![Some("new")]
    );
}

fn env_required_rejects_an_unset_variable() {
    // 成功分支只是 `env::var` 透传；缺失或非 UTF-8 统一归 [`Code::Missing`]。
    let error = env_required("BUILDGEN_TEST_ENV_VAR_UNSET").unwrap_err();
    assert_eq!(error.code(), Code::Missing);
}

fn slices_render_quoted_and_expression_elements() {
    let values = ["a".to_string(), "b".to_string()];
    assert_eq!(string_slice(&values), "&[\"a\", \"b\"]");
    assert_eq!(expr_slice(&values), "&[a,b]");
    assert_eq!(expr_slice(&[]), "&[]");
}

fn read_text_returns_utf8_body_or_missing_error() {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("body.txt");
    std::fs::write(&path, "hello").unwrap();
    assert_eq!(read_text(&path).unwrap(), "hello");
    assert!(read_text(&dir.join("missing")).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

fn files_lists_only_regular_files_sorted() {
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("b.txt"), "1").unwrap();
    std::fs::write(dir.join("a.txt"), "2").unwrap();
    std::fs::create_dir(dir.join("sub")).unwrap();
    std::fs::write(dir.join("sub/c.txt"), "3").unwrap();

    let names: Vec<String> = files(&dir)
        .unwrap()
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["a.txt", "b.txt"]);

    std::fs::remove_dir_all(&dir).unwrap();
}

fn subtable_and_string_read_required_keys_or_fail() {
    let table = toml::from_str::<toml::Table>("[meta]\nname = \"x\"\n").unwrap();

    let meta = subtable(&table, "meta").unwrap();
    assert_eq!(string(meta, "name").unwrap(), "x");
    assert_eq!(subtable(&table, "meta").unwrap(), meta);

    assert!(subtable(&table, "missing").is_err());
    assert!(string(meta, "missing").is_err());
}
}

/// 临时目录：进程号 + 原子序号 + 纳秒，避免同纳秒内并行用例撞名。
fn temp_dir() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("buildgen-{}-{seq}-{nanos}", std::process::id()))
}
