//! 构建期共享内核：通用的路径处理与 TOML 加工；各 crate 的 `build.rs` 生成自己的
//! 强类型静态视图。
//!
//! - `build-std`（纯 std）：[`file_stem`]、[`extension`]、[`files`]、[`collect`]、[`relative`]、
//!   [`env_required`]、[`read_text`]、[`string_slice`]、[`expr_slice`]。
//! - `build`（再带 `toml`）：[`read`]、[`merge_tables`]、[`subtable`]、[`string`]。

use crate::{Code, Error, Result};
use std::env;
use std::path::{Path, PathBuf};

/// 文件名主干（去掉最后一个扩展名）；非 UTF-8 返回 `None`。
#[must_use]
pub fn file_stem(path: &Path) -> Option<&str> {
    path.file_stem().and_then(|stem| stem.to_str())
}

/// 文件扩展名（不含点）；非 UTF-8 返回 `None`。
#[must_use]
pub fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_string)
}

/// 列出 `dir` 下的普通文件，按路径升序；子目录与非文件条目被排除。
///
/// # Errors
///
/// 目录无法读取时返回 [`Code::Io`]，错误消息带目录路径。
pub fn files(dir: &Path) -> Result<Vec<PathBuf>> {
    let read = std::fs::read_dir(dir).map_err(|error| io_err(dir, &error))?;
    let mut files = Vec::new();
    for entry in read {
        let path = entry.map_err(|error| io_err(dir, &error))?.path();
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// 列出 `dir` 下的 `.toml` 文件，按路径升序，供“目录型”语料（`ops/`、`devices/`）使用。
/// 子目录与非 `.toml` 文件被过滤。
///
/// # Errors
///
/// 目录无法读取时返回 [`Code::Io`]，错误消息带目录路径。
pub fn collect(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = files(dir)?;
    files.retain(|path| is_toml(path));
    Ok(files)
}

/// 构造一个目录读取失败的 `io` 错误，消息带目录路径。
fn io_err(dir: &Path, error: &std::io::Error) -> Error {
    Error::new(Code::Io, format!("读取 {} 失败：{error}", dir.display()))
}

/// 路径是否为 `.toml` 普通文件；目录与其他扩展名被排除。
fn is_toml(path: &Path) -> bool {
    path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("toml")
}

/// 相对 `base` 的展示路径，给 `cargo:rerun-if-changed` 用，构建日志不带绝对路径。
#[must_use]
pub fn relative(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// 读取构建脚本必带的环境变量（`CARGO_MANIFEST_DIR` / `OUT_DIR` 等）；不猜缺省目录。
///
/// # Errors
///
/// 变量未设置或不是合法 Unicode 时返回 [`Code::Missing`]，消息带变量名。
pub fn env_required(name: &str) -> Result<String> {
    env::var(name).map_err(|_| {
        Error::new(
            Code::Missing,
            format!("`{name}` 未设置：只有 build.rs 会读它"),
        )
    })
}

/// 字符串切片字面量：元素保持 Rust 字符串引号，写进生成代码的 `&[..]`。
#[must_use]
pub fn string_slice(values: &[String]) -> String {
    format!("&{values:?}")
}

/// 表达式切片字面量：元素已是表达式，不再加引号，逗号内联拼接。
#[must_use]
pub fn expr_slice(values: &[String]) -> String {
    format!("&[{}]", values.join(","))
}

/// 读取一份文件正文（不解析 TOML）；不解释内容，失败出口由调用方决定。
///
/// # Errors
///
/// 文件读不到时返回 [`Code::Missing`]（路径不存在）或 [`Code::Io`]（其余），消息带文件路径。
pub fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|error| Error::from_io(path, &error))
}

/// 读取并解析一个 TOML 文件，要求顶层是表。
///
/// # Errors
///
/// 文件读不到时返回 [`Code::Io`]；TOML 解析失败或顶层不是表时返回
/// [`Code::Invalid`]，错误消息带文件路径。
#[cfg(feature = "build")]
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

/// 深度合并：`overlay` 覆盖 `base`，同名子表递归合并；数组与标量整体替换，不逐项拼接。
#[cfg(feature = "build")]
pub fn merge_tables(base: &mut toml::Table, overlay: toml::Table) {
    for (key, value) in overlay {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(inner)), toml::Value::Table(update)) => {
                merge_tables(inner, update);
            }
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

/// 取必填子表；键缺失或该键不是表时拒绝。
///
/// # Errors
///
/// 键缺失或该键不是表时返回 [`Code::Invalid`]，消息带键名。
#[cfg(feature = "build")]
pub fn subtable<'a>(table: &'a toml::Table, key: &str) -> Result<&'a toml::Table> {
    table
        .get(key)
        .and_then(toml::Value::as_table)
        .ok_or_else(|| Error::new(Code::Invalid, format!("缺子表 `{key}`")))
}

/// 取必填字符串值；键缺失或不是字符串时拒绝。
///
/// # Errors
///
/// 键缺失或该键不是字符串时返回 [`Code::Invalid`]，消息带键名。
#[cfg(feature = "build")]
pub fn string<'a>(table: &'a toml::Table, key: &str) -> Result<&'a str> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .ok_or_else(|| Error::new(Code::Invalid, format!("缺字符串键 `{key}`")))
}

#[cfg(all(test, feature = "build"))]
#[path = "../tests/buildgen_tests.rs"]
mod tests;
