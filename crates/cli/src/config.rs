//! 配置文件层（ADR 0048）：发现顺序、TOML 展平、逐格入槽。
//!
//! 优先级最低的一层，因此这里只"写进槽位"，判定谁赢在 [`crate::settings`]。嵌套表展平到
//! 点号键（`[cache] dir = "x"` 与 `cache.dir = "x"` 等价），深度上限写死在这里，因为它是
//! **文件形态**的约束，不是解析策略。
//!
//! **不拥有**：优先级与来源打印（[`crate::settings`]）、键是否登记（[`crate::key`]）。

use crate::key::{Key, Origin, dotted_key, home, invalid, rebase_value, value_of_text};
use crate::settings::Settings;
use diagnostic::{Code, Error, Result};
use std::path::{Path, PathBuf};

/// 配置文件的候选位置（不含显式 `--config` / `CUTILE_CONFIG`）；顺序即优先级。
pub(crate) fn discover_files() -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from("cutile.toml")];
    #[cfg(windows)]
    if let Some(appdata) = std::env::var_os("APPDATA").filter(|value| !value.is_empty()) {
        candidates.push(PathBuf::from(appdata).join("cutile").join("config.toml"));
    }
    #[cfg(not(windows))]
    if let Some(home) = home() {
        candidates.push(home.join(".config").join("cutile").join("config.toml"));
    }
    candidates
}

impl Settings {
    /// 配置文件层发现（ADR 0048）：`CUTILE_CONFIG` → `./cutile.toml` → 平台用户目录；
    /// 只读第一份命中，文件不存在不是错误。
    pub(crate) fn read_config(&mut self, lookup: impl Fn(&str) -> Option<String>) -> Result<()> {
        if let Some(path) = lookup("CUTILE_CONFIG") {
            return self.read_config_at(Path::new(&path));
        }
        for path in discover_files() {
            if path.is_file() {
                return self.read_config_at(&path);
            }
        }
        Ok(())
    }

    /// 读一份配置文件：键未登记 / 类型不符 / 坏 TOML 都是 `invalid`，文件不存在是 `missing`。
    pub(crate) fn read_config_at(&mut self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            let code = if path.is_file() {
                Code::Io
            } else {
                Code::Missing
            };
            Error::new(code, format!("读配置文件 {} 失败：{error}", path.display()))
                .with("cli::settings")
        })?;
        let table: toml::Table = toml::from_str(&text).map_err(|error| {
            invalid(format!(
                "配置文件 {} 不是合法 TOML：{error}",
                path.display()
            ))
        })?;
        self.apply_table(path, table, "", 0)
    }

    /// 递归展平嵌套表：`[cache] dir = "x"` 与 `cache.dir = "x"` 都归到点号键 `cache.dir`。
    fn apply_table(
        &mut self,
        path: &Path,
        table: toml::Table,
        prefix: &str,
        depth: u32,
    ) -> Result<()> {
        if depth > 8 {
            return Err(invalid(format!(
                "配置文件 {} 嵌套超过 8 层",
                path.display()
            )));
        }
        for (name, raw) in table {
            let dotted = dotted_name(prefix, &name);
            self.apply_entry(path, dotted, raw, depth)?;
        }
        Ok(())
    }

    /// 一条配置项：表继续递归，标量入槽。
    fn apply_entry(
        &mut self,
        path: &Path,
        dotted: String,
        raw: toml::Value,
        depth: u32,
    ) -> Result<()> {
        if let toml::Value::Table(inner) = raw {
            self.apply_table(path, inner, &dotted, depth + 1)
        } else {
            self.apply_value(path, &dotted, raw)
        }
    }

    /// 一条标量配置：键解析、标量提取、类型解析与入槽。
    fn apply_value(&mut self, path: &Path, dotted: &str, raw: toml::Value) -> Result<()> {
        let key = dotted_key(dotted)
            .map_err(|error| error.with(format!("配置文件 {}", path.display())))?;
        let origin = Origin::File(path.to_path_buf());
        let text = scalar_text(&origin, dotted, key, &raw)?;
        let value = value_of_text(key, &origin, &text)?;
        let value = rebase_value(value, self.roots().base());
        self.store(key, value, origin);
        Ok(())
    }
}

/// 嵌套表键名拼接：顶层键没有前缀。
fn dotted_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// 标量值转文本：字符串 / 整数 / 浮点；数组与嵌套表不在这里出现（表已在上层递归）。
fn scalar_text(origin: &Origin, dotted: &str, key: Key, raw: &toml::Value) -> Result<String> {
    raw.as_str()
        .map(str::to_string)
        .or_else(|| raw.as_integer().map(|value| value.to_string()))
        .or_else(|| raw.as_float().map(|value| value.to_string()))
        .ok_or_else(|| {
            invalid(format!(
                "{origin}提供的 `{dotted}`（{}）非法：不支持数组值",
                key.doc()
            ))
        })
}
