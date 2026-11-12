//! 配置错误：每条都带"哪个键、值从哪来"，调用方不需要再补上下文。

use crate::key::Key;
use crate::source::Origin;
use std::fmt;
use std::path::PathBuf;

/// 配置层错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// 读配置文件失败。
    Io {
        /// 文件路径。
        path: PathBuf,
        /// 底层错误信息。
        message: String,
    },
    /// TOML 语法错误。
    Syntax {
        /// 文件路径。
        path: PathBuf,
        /// 1 起始的行号。
        line: usize,
        /// 说明。
        message: String,
    },
    /// 配置文件里出现闭集之外的键：当作错误，而不是静默忽略。
    UnknownKey {
        /// 文件路径。
        path: PathBuf,
        /// 未登记的键（点号形式）。
        key: String,
        /// 当前应用登记的键（点号形式）：错误信息里直接列出，省掉一次跳转。
        allowed: Vec<&'static str>,
    },
    /// `schema` 版本与本 crate 不匹配。
    Schema {
        /// 文件路径。
        path: PathBuf,
        /// 文件里写的值。
        found: String,
        /// 本 crate 接受的版本。
        expected: u16,
    },
    /// 某个来源给出的值不符合该键的形状。
    BadValue {
        /// 键。
        key: Key,
        /// 值从哪来。
        origin: Origin,
        /// 说明。
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "读取 `{}` 失败：{message}", path.display()),
            Self::Syntax {
                path,
                line,
                message,
            } => write!(f, "`{}` 第 {line} 行：{message}", path.display()),
            Self::UnknownKey { path, key, allowed } => write!(
                f,
                "`{}` 出现未登记的键 `{key}`；本应用登记的是：`{}`",
                path.display(),
                allowed.join("`, `")
            ),
            Self::Schema {
                path,
                found,
                expected,
            } => write!(
                f,
                "`{}` 的 schema = {found}，本版本只接受 {expected}",
                path.display()
            ),
            Self::BadValue {
                key,
                origin,
                message,
            } => write!(
                f,
                "{origin}提供的 `{}`（{}）非法：{message}",
                key.file_key(),
                key.doc()
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Layer;

    #[test]
    fn unknown_keys_list_the_closed_set() {
        let error = Error::UnknownKey {
            path: PathBuf::from("/p/cutile.toml"),
            key: "cache.direkt".to_string(),
            allowed: vec!["cache.dir", "work.dir"],
        };
        let text = error.to_string();
        assert!(text.contains("cache.direkt"), "{text}");
        assert!(text.contains("cache.dir"), "{text}");
    }

    #[test]
    fn bad_values_name_the_key_and_the_origin() {
        let error = Error::BadValue {
            key: Key::CacheDir,
            origin: Origin::new(Layer::Env, "CUTILE_CACHE_DIR"),
            message: "路径为空".to_string(),
        };
        let text = error.to_string();
        assert!(text.contains("cache.dir"), "{text}");
        assert!(text.contains("CUTILE_CACHE_DIR"), "{text}");
    }
}
