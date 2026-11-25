//! 构建期共享内核：扫描语料并读取 TOML；各 crate 直接生成自己的强类型静态视图。

use crate::{Code, Error, Result};
use std::path::{Path, PathBuf};

/// 列出 `dir` 下的 `.toml` 文件，按路径升序，供“目录型”语料（`ops/`、`devices/`）使用。
/// 子目录与非 `.toml` 文件被过滤。
///
/// # Errors
///
/// 目录无法读取时返回 [`Code::Io`]，错误消息带目录路径。
pub fn collect(dir: &Path) -> Result<Vec<PathBuf>> {
    let read = std::fs::read_dir(dir).map_err(|error| io_err(dir, &error))?;
    let mut files = Vec::new();
    for entry in read {
        files.push(entry.map_err(|error| io_err(dir, &error))?.path());
    }
    files.retain(|path| is_toml(path));
    files.sort();
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

/// 读取并解析一个 TOML 文件，要求顶层是表。
///
/// # Errors
///
/// 文件读不到时返回 [`Code::Io`]；TOML 解析失败或顶层不是表时返回
/// [`Code::Invalid`]，错误消息带文件路径。
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

/// 相对 `base` 的展示路径，给 `cargo:rerun-if-changed` 用，构建日志不带绝对路径。
#[must_use]
pub fn relative(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
#[path = "../tests/buildgen_tests.rs"]
mod tests;
