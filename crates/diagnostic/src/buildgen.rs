//! 构建期共享内核：把“扫描语料 → 读取 TOML → 展开成 [`data`](crate::data) 构造表达式”
//! 三段各 crate `build.rs` 重复的逻辑收敛到一处。
//!
//! 本模块只在 `build` feature 下编译，**不进入运行时**：默认 feature 仍是纯 std、零依赖，
//! 运行期依旧没有任何 TOML 解析器（ADR 0022）。所有失败都以 [`Result`] 返回，由构建脚本
//! 决定如何 panic——本模块不出现 `panic!` 或裸 `unwrap`，与仓库“错误即值”的纪律一致。
//! 这里也不决定语料放在哪、外层如何包装，只提供四个正交的纯函数。

use crate::{Code, Error, Result};
use core::fmt::Write as _;
use std::path::{Path, PathBuf};

/// 列出 `dir` 下的 `.toml` 文件，按路径升序，供“目录型”语料（`ops/`、`devices/`）使用。
/// 子目录与非 `.toml` 文件被过滤。
pub fn collect(dir: &Path) -> Result<Vec<PathBuf>> {
    let read = std::fs::read_dir(dir).map_err(|error| io_err(dir, error))?;
    let mut files = Vec::new();
    for entry in read {
        files.push(entry.map_err(|error| io_err(dir, error))?.path());
    }
    files.retain(|path| is_toml(path));
    files.sort();
    Ok(files)
}

/// 构造一个目录读取失败的 `io` 错误，消息带目录路径。
fn io_err(dir: &Path, error: std::io::Error) -> Error {
    Error::new(Code::Io, format!("读取 {} 失败：{error}", dir.display()))
}

/// 路径是否为 `.toml` 普通文件；目录与其他扩展名被排除。
fn is_toml(path: &Path) -> bool {
    path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("toml")
}

/// 读取并解析一个 TOML 文件，要求顶层是表。
pub fn read(path: &Path) -> Result<toml::Table> {
    let text = std::fs::read_to_string(path).map_err(|error| Error::from_io(path, &error))?;
    let value: toml::Value = toml::from_str(&text).map_err(|error| {
        Error::new(
            Code::Invalid,
            format!("解析 {} 失败：{error}", path.display()),
        )
    })?;
    match value {
        toml::Value::Table(table) => Ok(table),
        _ => Err(Error::new(
            Code::Invalid,
            format!("`{}` 的顶层必须是 TOML 表", path.display()),
        )),
    }
}

/// 把一张 TOML 表展开成 `Table::from_pairs(vec![…])` 表达式。
///
/// `path` 是当前表的点号路径，仅用于在非法值（非有限浮点、日期时间）的错误里定位。
pub fn emit(table: &toml::Table, path: &str) -> Result<String> {
    Ok(format!("Table::from_pairs({})", pairs(table, path)?))
}

/// 生成表内的 `vec![("key", value), …]` 片段，顶层表与 `Value::table` 共用。
fn pairs(table: &toml::Table, path: &str) -> Result<String> {
    let mut out = String::from("vec![");
    for (key, value) in table {
        let child = format!("{path}.{key}");
        let _ = write!(out, "(\"{key}\", {}), ", value_expr(value, &child)?);
    }
    out.push(']');
    Ok(out)
}

/// 按值的形态分发：标量交给 [`scalar_expr`]，数组与表各自内联展开，把单函数复杂度压到阈值内。
fn value_expr(value: &toml::Value, path: &str) -> Result<String> {
    match value {
        toml::Value::Array(items) => array_expr(items, path),
        toml::Value::Table(table) => Ok(format!("Value::table({})", pairs(table, path)?)),
        scalar => scalar_expr(scalar, path),
    }
}

/// 四类合法标量的构造表达式；其余形态（日期时间等）交给 [`invalid_scalar`]。
fn scalar_expr(value: &toml::Value, path: &str) -> Result<String> {
    match value {
        toml::Value::String(text) => Ok(format!("Value::text({text:?})")),
        toml::Value::Integer(number) => Ok(format!("Value::int({number})")),
        toml::Value::Float(number) => float_expr(*number, path),
        toml::Value::Boolean(flag) => Ok(format!("Value::flag({flag})")),
        other => invalid_scalar(other, path),
    }
}

/// 浮点值表达式：非有限值返回带路径的 `invalid` 错误。
fn float_expr(number: f64, path: &str) -> Result<String> {
    if !number.is_finite() {
        return Err(Error::new(
            Code::Invalid,
            format!("`{path}` 不是有限浮点数：声明语料不使用 NaN/inf"),
        ));
    }
    Ok(format!("Value::real({number:?})"))
}

/// 非法标量：日期时间提示改字符串，其余视为内部不变量破坏。
fn invalid_scalar(value: &toml::Value, path: &str) -> Result<String> {
    match value {
        toml::Value::Datetime(_) => Err(Error::new(
            Code::Invalid,
            format!("`{path}` 是 TOML 日期时间：声明语料不使用日期，请改成字符串"),
        )),
        _ => Err(Error::new(Code::Internal, "标量分支收到非标量值")),
    }
}

/// 数组值的 `Value::array(vec![…])` 表达式。
fn array_expr(items: &[toml::Value], path: &str) -> Result<String> {
    let mut out = String::from("Value::array(vec![");
    for (index, item) in items.iter().enumerate() {
        let child = format!("{path}[{index}]");
        let _ = write!(out, "{}, ", value_expr(item, &child)?);
    }
    out.push_str("])");
    Ok(out)
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
}
