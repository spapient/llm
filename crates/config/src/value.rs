//! 配置值的类型系统。
//!
//! 一个键接受什么形状（路径 / 文本 / 整数区间 / 布尔 / 枚举）由 [`Kind`] 声明一次，
//! 配置文件、环境变量与命令行三条入口共用同一个校验函数，因此"文本怎么变成值"只有一份实现。

use crate::toml;
use std::fmt;
use std::path::{Path, PathBuf};

/// 一个键接受的取值形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 路径；相对路径由 [`Value::rebase`] 按来源的基准目录解析。
    Path,
    /// 自由文本（空字符串非法）。
    Text,
    /// 整数闭区间。
    Int {
        /// 下界（含）。
        lo: i64,
        /// 上界（含）。
        hi: i64,
    },
    /// 布尔：`1/0`、`true/false`、`on/off`、`yes/no` 都接受。
    Bool,
    /// 枚举：只能取给定集合之一。
    Choice(&'static [&'static str]),
}

impl Kind {
    /// 从文本解析（环境变量、`--set`、配置文件的字符串）。
    pub fn parse(self, text: &str) -> Result<Value, String> {
        let text = text.trim();
        match self {
            Self::Path => {
                if text.is_empty() {
                    return Err("路径为空".to_string());
                }
                Ok(Value::Path(PathBuf::from(text)))
            }
            Self::Text => {
                if text.is_empty() {
                    return Err("文本为空".to_string());
                }
                Ok(Value::Text(text.to_string()))
            }
            Self::Int { lo, hi } => {
                let number = text
                    .parse::<i64>()
                    .map_err(|_| format!("`{text}` 不是整数"))?;
                if number < lo || number > hi {
                    return Err(format!("{number} 不在 {lo}..{hi} 内"));
                }
                Ok(Value::Int(number))
            }
            Self::Bool => match text.to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => Ok(Value::Bool(true)),
                "0" | "false" | "off" | "no" => Ok(Value::Bool(false)),
                _ => Err(format!("`{text}` 不是布尔值")),
            },
            Self::Choice(names) => {
                if names.contains(&text) {
                    Ok(Value::Text(text.to_string()))
                } else {
                    Err(format!("`{text}` 不是 {} 之一", names.join("/")))
                }
            }
        }
    }

    /// 从 TOML 值解析；形状不符时返回可直接拼进报错的说明。
    pub fn from_toml(self, raw: &toml::Value) -> Result<Value, String> {
        match (self, raw) {
            (Self::Path, toml::Value::Str(text)) => self.parse(text),
            (Self::Text, toml::Value::Str(text)) => self.parse(text),
            (Self::Choice(_), toml::Value::Str(text)) => self.parse(text),
            (Self::Int { .. }, toml::Value::Int(number)) => self.parse(&number.to_string()),
            (Self::Bool, toml::Value::Bool(flag)) => Ok(Value::Bool(*flag)),
            (kind, other) => Err(format!(
                "需要 {}，实际是 {}",
                kind.describe(),
                type_name(other)
            )),
        }
    }

    /// 形状的人话描述，用于报错与文档。
    pub fn describe(self) -> &'static str {
        match self {
            Self::Path => "路径",
            Self::Text => "非空文本",
            Self::Int { .. } => "整数",
            Self::Bool => "布尔值",
            Self::Choice(_) => "枚举值",
        }
    }
}

fn type_name(value: &toml::Value) -> &'static str {
    match value {
        toml::Value::Str(_) => "字符串",
        toml::Value::Int(_) => "整数",
        toml::Value::Float(_) => "浮点数",
        toml::Value::Bool(_) => "布尔值",
        toml::Value::Array(_) => "数组",
        toml::Value::Table(_) => "表",
    }
}

/// 解析后的配置值。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Value {
    /// 已按基准目录解析过的路径。
    Path(PathBuf),
    /// 文本（枚举也走这里）。
    Text(String),
    /// 整数。
    Int(i64),
    /// 布尔。
    Bool(bool),
}

impl Value {
    /// 相对路径按 `base` 解析；其它值原样返回。
    pub fn rebase(self, base: &Path) -> Self {
        match self {
            Self::Path(path) if path.is_relative() => Self::Path(base.join(path)),
            other => other,
        }
    }

    /// 取路径。
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => Some(path),
            _ => None,
        }
    }

    /// 取文本。
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// 取整数。
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(number) => Some(*number),
            _ => None,
        }
    }

    /// 取布尔。
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(flag) => Some(*flag),
            _ => None,
        }
    }

    /// 打印与落盘用的文本形式。
    pub fn render(&self) -> String {
        match self {
            Self::Path(path) => path.display().to_string(),
            Self::Text(text) => text.clone(),
            Self::Int(number) => number.to_string(),
            Self::Bool(flag) => flag.to_string(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_and_texts_reject_empty() {
        assert!(Kind::Path.parse("   ").is_err());
        assert!(Kind::Text.parse("").is_err());
        assert_eq!(
            Kind::Path.parse(" target/cache ").unwrap(),
            Value::Path(PathBuf::from("target/cache"))
        );
    }

    #[test]
    fn integers_are_range_checked() {
        let kind = Kind::Int { lo: 1, hi: 64 };
        assert_eq!(kind.parse("8").unwrap(), Value::Int(8));
        assert!(kind.parse("0").is_err());
        assert!(kind.parse("65").is_err());
        assert!(kind.parse("八").is_err());
    }

    #[test]
    fn booleans_accept_the_usual_spellings() {
        assert_eq!(Kind::Bool.parse("on").unwrap(), Value::Bool(true));
        assert_eq!(Kind::Bool.parse("FALSE").unwrap(), Value::Bool(false));
        assert!(Kind::Bool.parse("maybe").is_err());
    }

    #[test]
    fn choices_list_the_allowed_values() {
        const LEVELS: &[&str] = &["debug", "info", "trace"];
        let kind = Kind::Choice(LEVELS);
        assert_eq!(kind.parse("debug").unwrap(), Value::Text("debug".into()));
        let error = kind.parse("verbose").unwrap_err();
        assert!(error.contains("debug/info/trace"), "{error}");
    }

    #[test]
    fn toml_values_are_checked_against_the_kind() {
        let kind = Kind::Path;
        assert!(kind.from_toml(&toml::Value::Int(1)).is_err());
        assert_eq!(
            kind.from_toml(&toml::Value::Str("a".into())).unwrap(),
            Value::Path(PathBuf::from("a"))
        );
        assert_eq!(
            Kind::Bool.from_toml(&toml::Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn relative_paths_are_rebased_and_absolute_ones_kept() {
        let base = Path::new("/work/project");
        assert_eq!(
            Value::Path(PathBuf::from("target/x")).rebase(base),
            Value::Path(PathBuf::from("/work/project/target/x"))
        );
        assert_eq!(
            Value::Path(PathBuf::from("/abs/x")).rebase(base),
            Value::Path(PathBuf::from("/abs/x"))
        );
        assert_eq!(Value::Int(3).rebase(base), Value::Int(3));
    }
}
