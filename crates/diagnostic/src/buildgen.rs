//! 构建期共享内核：把“扫描语料 → 读取 TOML → 展开成 [`data`](crate::data) 构造表达式”
//! 三段各 crate `build.rs` 重复的逻辑收敛到一处。
//!
//! 本模块只在 `build` feature 下编译，**不进入运行时**：默认 feature 仍是纯 std、零依赖，
//! 运行期依旧没有任何 TOML 解析器（ADR 0022）。这里也不决定语料放在哪、外层如何包装
//! （`OnceLock` / `names()` 等由各 build.rs 保留），只提供四个正交的纯函数。

use core::fmt::Write as _;
use std::path::{Path, PathBuf};

/// 列出 `dir` 下的 `.toml` 文件，按路径升序，供“目录型”语料（`ops/`、`devices/`）使用。
///
/// 子目录与非 `.toml` 文件被过滤；读目录失败直接带路径 panic——这是构建期错误，
/// 没有可恢复的运行时语义。
pub fn collect(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("读取 {} 失败：{error}", dir.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("toml")
        })
        .collect();
    files.sort();
    files
}

/// 读取并解析一个 TOML 文件，要求顶层是表；任何失败都带路径 panic。
pub fn read(path: &Path) -> toml::Table {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("读取 {} 失败：{error}", path.display()));
    let value: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("解析 {} 失败：{error}", path.display()));
    let toml::Value::Table(table) = value else {
        panic!("`{}` 的顶层必须是 TOML 表", path.display());
    };
    table
}

/// 把一张 TOML 表展开成 `Table::from_pairs(vec![…])` 表达式。
///
/// `path` 是当前表的点号路径，仅用于在非法值（非有限浮点、日期时间）panic 时定位。
#[must_use]
pub fn emit(table: &toml::Table, path: &str) -> String {
    format!("Table::from_pairs({})", pairs(table, path))
}

/// 生成表内的 `vec![("key", value), …]` 片段，顶层表与 `Value::table` 共用。
fn pairs(table: &toml::Table, path: &str) -> String {
    let mut out = String::from("vec![");
    for (key, value) in table {
        let child = format!("{path}.{key}");
        let _ = write!(out, "(\"{key}\", {}), ", value_expr(value, &child));
    }
    out.push(']');
    out
}

/// 把一个 TOML 值展开成对应的 `Value::…` 构造表达式。
fn value_expr(value: &toml::Value, path: &str) -> String {
    match value {
        toml::Value::String(text) => format!("Value::text({text:?})"),
        toml::Value::Integer(number) => format!("Value::int({number})"),
        toml::Value::Float(number) => {
            if !number.is_finite() {
                panic!("`{path}` 不是有限浮点数：声明语料不使用 NaN/inf");
            }
            format!("Value::real({number:?})")
        }
        toml::Value::Boolean(flag) => format!("Value::flag({flag})"),
        toml::Value::Datetime(_) => {
            panic!("`{path}` 是 TOML 日期时间：声明语料不使用日期，请改成字符串")
        }
        toml::Value::Array(items) => {
            let mut out = String::from("Value::array(vec![");
            for (index, item) in items.iter().enumerate() {
                let child = format!("{path}[{index}]");
                let _ = write!(out, "{}, ", value_expr(item, &child));
            }
            out.push_str("])");
            out
        }
        toml::Value::Table(table) => format!("Value::table({})", pairs(table, path)),
    }
}

/// 相对 `base` 的展示路径，给 `cargo:rerun-if-changed` 用，构建日志不带绝对路径。
#[must_use]
pub fn relative(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn emit_covers_every_value_shape() {
        let table: toml::Table = toml::from_str(
            "[head]\nname = \"add\"\nrank = 7\nratio = 0.5\non = true\n\
             tags = [\"a\", \"b\"]\nnested = { k = 1 }",
        )
        .unwrap();
        let out = emit(&table, "root");
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
        let err = panic_message(|| emit(&nan, "bad"));
        assert!(err.contains("bad.x") && err.contains("有限浮点"));

        let date: toml::Table = toml::from_str("x = 2026-01-01").unwrap();
        let err = panic_message(|| emit(&date, "bad"));
        assert!(err.contains("bad.x") && err.contains("日期"));
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
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a.toml", "b.toml"]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_accepts_a_table_and_rejects_the_rest() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("ok.toml");
        std::fs::write(&ok, "x = 1").unwrap();
        assert_eq!(read(&ok)["x"].as_integer(), Some(1));

        let broken = dir.join("broken.toml");
        std::fs::write(&broken, "x = ").unwrap();
        assert!(panic_message(|| read(&broken)).contains("解析"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn relative_strips_the_base_prefix() {
        let base = Path::new("/repo/crates/op");
        let path = base.join("ops/add.toml");
        assert_eq!(relative(base, &path), "ops/add.toml");
    }

    /// 运行闭包并捕获 panic 的消息文本，忽略闭包的正常返回值。
    fn panic_message<Ret>(f: impl FnOnce() -> Ret) -> String {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(_) => panic!("期望 panic 但没有发生"),
            Err(payload) => payload
                .downcast_ref::<String>()
                .map_or_else(|| String::from("?"), String::clone),
        }
    }

    /// 以纳秒时间戳命名的唯一临时目录，避免并行测试撞名。
    fn temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("buildgen-{nanos}"))
    }
}
