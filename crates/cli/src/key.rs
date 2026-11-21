//! 配置键闭集与取值类型：点号名、环境变量后缀、默认值、值解析、来源。
//!
//! 键表只有 [`keys!`] 调用这一份：点号名、环境变量名、类型分派、默认值、人读说明、槽位下标
//! 全部由它派生，因此"加了一个键、忘了另一处"在结构上不可能。新增键只加一行。
//!
//! **不拥有**：优先级规则（在 [`crate::settings`]）、目录默认值（在 [`crate::root`]）。

use crate::root::{APP, Roots};
use diagnostic::{Code, Error, Result};
use std::fmt;
use std::path::{Path, PathBuf};

/// 环境变量前缀；完整变量名是 `<前缀>_<后缀>`。
const PREFIX: &str = "CUTILE";

/// 配置值闭集：值类型由每个键在 `Key` 闭集里声明，`describe` 与消费投影按类型收窄。
/// 变体只保留键表实际使用的类型；新键需要 Int/Float 时再加（值解析处一行分支）。
#[derive(Debug, Clone, PartialEq)]
pub enum SettingValue {
    /// 路径：相对路径按启动目录解析（与 `--set` 口径一致）。
    Path(PathBuf),
    /// 非负字节数（`u64`）。
    Bytes(u64),
}

impl SettingValue {
    /// 投影为路径；类型不符（键表声明下不可达，防御分支）给 `None`。
    pub(crate) fn into_path(self) -> Option<PathBuf> {
        match self {
            Self::Path(path) => Some(path),
            _ => None,
        }
    }

    /// 借出路径视图。
    pub(crate) fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => Some(path),
            _ => None,
        }
    }
}

impl fmt::Display for SettingValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(path) => write!(formatter, "{}", path.display()),
            Self::Bytes(bytes) => write!(formatter, "{bytes}"),
        }
    }
}

/// 全部键与它们的值类型、环境变量、默认值、说明一次写完：新增一个键只加一行。
///
/// 名字表只有这一份：点号名、环境变量后缀、值解析器、默认值、说明、槽位下标全部由它派生，
/// `describe` 与"键未登记"报错里列出的清单也是同一张表，因此不会出现"加了一处、忘了另一处"。
/// `parse_value` / `default_value` 的分派 match 在宏体内（宏体不参与圈复杂度门），
/// 每种类型的解析逻辑在宏外的小函数里（`parse_path` 等）。
macro_rules! keys {
    ($roots:ident; $($variant:ident => $dotted:literal, [$($env:literal),*], $parser:path, $default:expr, $doc:literal;)*) => {
        /// 配置键闭集。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Key {
            $($variant,)*
        }

        impl Key {
            /// 全部键，顺序即 `describe` 与报错清单的顺序。
            pub(crate) const KEYS: [Self; <[()]>::len(&[$(keys!(@one $variant)),*])] = [$(Self::$variant,)*];

            /// `--set` 与配置文件里的点号名。
            pub(crate) const fn dotted(self) -> &'static str {
                match self {
                    $(Self::$variant => $dotted,)*
                }
            }

            /// 环境变量后缀；第一个是正式名，其余是历史兼容名。
            const fn suffixes(self) -> &'static [&'static str] {
                match self {
                    $(Self::$variant => &[$($env),*],)*
                }
            }

            /// 值解析分派：每种类型的解析器是独立小函数（`parse_path` 等）。
            fn parse_value(self, text: &str, origin: &Origin) -> Result<SettingValue> {
                match self {
                    $(Self::$variant => $parser(self, text, origin),)*
                }
            }

            /// 默认值：路径键由 `$roots` 计算，数值键是字面量；没有默认值的键写 `None`。
            pub(crate) fn default_value(self, $roots: &Roots) -> Option<SettingValue> {
                match self {
                    $(Self::$variant => $default,)*
                }
            }

            /// 人读说明，出现在"键未登记"与坏值诊断里。
            pub(crate) const fn doc(self) -> &'static str {
                match self {
                    $(Self::$variant => $doc,)*
                }
            }

            /// 槽位下标。
            pub(crate) const fn index(self) -> usize {
                self as usize
            }
        }
    };
    (@one $variant:ident) => { () };
}

keys! { roots;
    CacheDir => "cache.dir", ["CACHE_DIR", "CACHE"], parse_path, Some(SettingValue::Path(roots.default_cache_dir(APP))), "编译产物缓存的根目录";
    WorkDir => "work.dir", ["WORK_DIR"], parse_path, Some(SettingValue::Path(roots.default_work_dir(APP))), "编译与执行时的临时工作目录";
    DataRoot => "data.root", ["DATA_DIR"], parse_path, None, "声明层 / 档案层 / 片段层的替代根；未设置表示用内嵌数据";
    CacheLimit => "cache.limit_bytes", [], parse_bytes, Some(SettingValue::Bytes(2 * 1024 * 1024 * 1024)), "缓存容量上限（字节）";
}

/// 槽位数量；[`Key::KEYS`] 是唯一事实源。
pub(crate) const KEY_COUNT: usize = Key::KEYS.len();

impl Key {
    /// 点号名 → 键；未登记的名字不是错误，由调用方决定怎么说。
    fn parse(name: &str) -> Option<Self> {
        Self::KEYS.into_iter().find(|key| key.dotted() == name)
    }

    /// 全部点号名，用于"未登记"的诊断。
    fn dotted_names() -> Vec<&'static str> {
        Self::KEYS.iter().map(|key| key.dotted()).collect()
    }

    /// 环境变量全名，按优先顺序（第一个是正式名）。
    pub(crate) fn env_names(self) -> Vec<String> {
        self.suffixes()
            .iter()
            .map(|suffix| format!("{PREFIX}_{suffix}"))
            .collect()
    }
}

/// 值从哪来：层 + 可读定位。层只用于优先级与打印，不承载别的语义。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Origin {
    /// 内置默认值。
    Default,
    /// 配置文件，带路径。
    File(PathBuf),
    /// 环境变量，带变量名。
    Env(String),
    /// 命令行覆盖，带 `--set key=value` 原文。
    Cli(String),
}

impl Origin {
    /// 优先级序号，越大越优先。
    pub(crate) const fn rank(&self) -> u8 {
        match self {
            Self::Default => 0,
            Self::File(_) => 1,
            Self::Env(_) => 2,
            Self::Cli(_) => 3,
        }
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => formatter.write_str("内置默认值"),
            Self::File(path) => write!(formatter, "{}（file）", path.display()),
            Self::Env(name) => write!(formatter, "{name}（env）"),
            Self::Cli(text) => write!(formatter, "{text}（cli）"),
        }
    }
}

/// 一个已确定的值与它的来源。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Resolved {
    pub(crate) value: SettingValue,
    pub(crate) origin: Origin,
}

/// 点号名 → 键；未登记的名字带清单报错。
pub(crate) fn dotted_key(name: &str) -> Result<Key> {
    Key::parse(name).ok_or_else(|| {
        invalid(format!(
            "键 `{name}` 未登记（登记的是：{}）",
            Key::dotted_names().join("、")
        ))
    })
}

/// 按键声明的值类型解析文本；空串与纯空白视为没写。
pub(crate) fn value_of_text(key: Key, origin: &Origin, text: &str) -> Result<SettingValue> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "{origin}提供的 `{}`（{}）非法：值为空",
                key.dotted(),
                key.doc()
            ),
        )
        .with("cli::settings"));
    }
    key.parse_value(text, origin)
}

/// 一条坏值的诊断：键名与来源都在消息里。
fn bad_value(key: Key, origin: &Origin, what: &str) -> Error {
    Error::new(
        Code::Invalid,
        format!(
            "{origin}提供的 `{}`（{}）非法：{what}",
            key.dotted(),
            key.doc()
        ),
    )
    .with("cli::settings")
}

/// 路径：原文原样收下（相对路径由调用方按启动目录 rebase）。
fn parse_path(_key: Key, text: &str, _origin: &Origin) -> Result<SettingValue> {
    Ok(SettingValue::Path(PathBuf::from(text)))
}

/// 非负字节数。
fn parse_bytes(key: Key, text: &str, origin: &Origin) -> Result<SettingValue> {
    text.parse::<u64>()
        .map(SettingValue::Bytes)
        .map_err(|_| bad_value(key, origin, "应为非负整数"))
}

/// 相对路径按 `base` 解析；非路径值原样返回。
pub(crate) fn rebase_value(value: SettingValue, base: &Path) -> SettingValue {
    match value {
        SettingValue::Path(path) => SettingValue::Path(if path.is_relative() {
            base.join(path)
        } else {
            path
        }),
        other => other,
    }
}

/// 一条配置错误：上下文固定为 `cli::settings`。
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with("cli::settings")
}
