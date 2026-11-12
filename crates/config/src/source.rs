//! 值的来源：分层、优先级、可追溯。
//!
//! 一次 `Settings::load()` 会按固定顺序把三层来源并入 [`Sources`]：
//! 用户级文件 → 环境变量 → 命令行覆盖。高优先级后写，低优先级不会覆盖高优先级。

use crate::app::App;
use crate::error::Error;
use crate::key::Key;
use crate::paths::Roots;
use crate::toml::{self, Table};
use crate::value::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// 来源层；数值越大优先级越高。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    /// 内置默认值。
    Default,
    /// 配置文件。
    File,
    /// 环境变量。
    Env,
    /// 命令行覆盖。
    Cli,
}

impl Layer {
    /// 优先级序号。
    pub const fn rank(self) -> u8 {
        match self {
            Self::Default => 0,
            Self::File => 1,
            Self::Env => 2,
            Self::Cli => 3,
        }
    }

    /// 打印用名字。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::File => "file",
            Self::Env => "env",
            Self::Cli => "cli",
        }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 一个值具体从哪来：层 + 可读定位（文件路径、环境变量名、命令行）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    layer: Layer,
    detail: String,
}

impl Origin {
    /// 构造。
    pub fn new(layer: Layer, detail: impl Into<String>) -> Self {
        Self {
            layer,
            detail: detail.into(),
        }
    }

    /// 内置默认值。
    pub fn defaulted() -> Self {
        Self::new(Layer::Default, "内置默认值")
    }

    /// 层。
    pub fn layer(&self) -> Layer {
        self.layer
    }

    /// 定位信息。
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.layer == Layer::Default {
            f.write_str(&self.detail)
        } else {
            write!(f, "{}（{}）", self.detail, self.layer.as_str())
        }
    }
}

/// 尚未补默认值的键值集合。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sources {
    entries: BTreeMap<Key, (Value, Origin)>,
}

impl Sources {
    /// 空集合。
    pub fn new() -> Self {
        Self::default()
    }

    /// 写入一个值；只有优先级不低于已有来源时才生效。
    ///
    /// 返回 `true` 表示该值成为最终值，`false` 表示被更高优先级的来源挡住。
    pub fn set(&mut self, key: Key, value: Value, origin: Origin) -> bool {
        let accepted = match self.entries.get(&key) {
            Some((_, existing)) => origin.layer() >= existing.layer(),
            None => true,
        };
        if accepted {
            self.entries.insert(key, (value, origin));
        }
        accepted
    }

    /// 合并另一批来源（按同样的优先级规则逐键裁决）。
    pub fn merge(&mut self, other: Self) {
        for (key, (value, origin)) in other.entries {
            self.set(key, value, origin);
        }
    }

    /// 取值。
    pub fn get(&self, key: Key) -> Option<&Value> {
        self.entries.get(&key).map(|(value, _)| value)
    }

    /// 值的来源。
    pub fn origin(&self, key: Key) -> Option<&Origin> {
        self.entries.get(&key).map(|(_, origin)| origin)
    }

    /// 是否设置过（含默认层之外的任何来源）。
    pub fn contains(&self, key: Key) -> bool {
        self.entries.contains_key(&key)
    }

    /// 已设置键的数量。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否一个键都没设置。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 逐键遍历（键序稳定，便于打印）。
    pub fn iter(&self) -> impl Iterator<Item = (Key, &Value, &Origin)> {
        self.entries
            .iter()
            .map(|(key, (value, origin))| (*key, value, origin))
    }
}

/// 读取一个配置文件。
///
/// `schema` 缺省视为当前版本，写了就必须等于 [`crate::SCHEMA_VER`]；键必须是 [`App::KEYS`]
/// 里登记过的（即“本应用的键”，别的应用的键不算“历史兼容名”）；
/// 相对路径按**该文件所在目录**解析，而不是当前工作目录。
pub fn file<A: App>(path: &Path) -> Result<Sources, Error> {
    let text = std::fs::read_to_string(path).map_err(|error| Error::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let table = toml::parse(&text).map_err(|error| Error::Syntax {
        path: path.to_path_buf(),
        line: error.line,
        message: error.message,
    })?;

    check_schema(&table, path)?;
    check_unknown_keys::<A>(&table, path)?;

    let base = path.parent().unwrap_or(Path::new("."));
    let mut sources = Sources::new();
    for key in A::KEYS.iter().copied() {
        let Some(raw) = lookup(&table, key.file_key()) else {
            continue;
        };
        let value = key
            .kind()
            .from_toml(raw)
            .map_err(|message| Error::BadValue {
                key,
                origin: Origin::new(Layer::File, path.display().to_string()),
                message,
            })?;
        sources.set(
            key,
            value.rebase(base),
            Origin::new(Layer::File, path.display().to_string()),
        );
    }
    Ok(sources)
}

/// 从环境变量层收集。
///
/// `lookup` 让调用方决定从哪里读（生产传 `std::env::var` 的包装，测试传固定映射）；
/// 相对路径按进程启动目录解析，符合 shell 里设 `CUTILE_CACHE_DIR=build/cache` 的直觉。
///
/// 变量名是 `<前缀>_<后缀>`：前缀来自 [`App::ENV_PREFIX`]，因此同一个键闭集可以被多个应用
/// 复用（`CUTILE_CACHE_DIR` / `LLM_CACHE_DIR`），而键表里不出现任何应用名。
pub fn env<A: App>(
    lookup: impl Fn(&str) -> Option<String>,
    roots: &Roots,
) -> Result<Sources, Error> {
    let mut sources = Sources::new();
    for key in A::KEYS.iter().copied() {
        let Some((name, text)) = key
            .env_vars::<A>()
            .into_iter()
            .find_map(|name| lookup(&name).map(|text| (name, text)))
        else {
            continue;
        };
        let value = key.kind().parse(&text).map_err(|message| Error::BadValue {
            key,
            origin: Origin::new(Layer::Env, name.clone()),
            message,
        })?;
        sources.set(
            key,
            value.rebase(roots.base()),
            Origin::new(Layer::Env, name),
        );
    }
    Ok(sources)
}

/// 解析一条命令行覆盖（`--set cache.dir=/tmp/x` 中的 `cache.dir=/tmp/x`）。
///
/// 命令行层优先级最高，相对路径按进程启动目录解析。
pub fn cli_override<A: App>(text: &str, roots: &Roots) -> Result<(Key, Value, Origin), Error> {
    let (name, raw) = text.split_once('=').ok_or_else(|| Error::BadValue {
        key: Key::CacheDir,
        origin: Origin::new(Layer::Cli, format!("--set {text}")),
        message: "缺少 `=`：应写成 `key=value`".to_string(),
    })?;
    let key = Key::from_file_key_in::<A>(name).ok_or_else(|| Error::BadValue {
        key: Key::CacheDir,
        origin: Origin::new(Layer::Cli, format!("--set {text}")),
        message: format!(
            "键 `{}` 不属于应用 `{}`（它登记的是：{}）",
            name.trim(),
            A::DIR,
            Key::file_keys_of::<A>().join("、")
        ),
    })?;
    let value = key.kind().parse(raw).map_err(|message| Error::BadValue {
        key,
        origin: Origin::new(Layer::Cli, format!("--set {text}")),
        message,
    })?;
    Ok((
        key,
        value.rebase(roots.base()),
        Origin::new(Layer::Cli, format!("--set {}", key.file_key())),
    ))
}

fn check_schema(table: &Table, path: &Path) -> Result<(), Error> {
    match table.get("schema") {
        None => Ok(()),
        Some(toml::Value::Int(found)) if *found == i64::from(crate::SCHEMA_VER) => Ok(()),
        Some(other) => Err(Error::Schema {
            path: path.to_path_buf(),
            found: scalar(other),
            expected: crate::SCHEMA_VER,
        }),
    }
}

fn check_unknown_keys<A: App>(table: &Table, path: &Path) -> Result<(), Error> {
    let mut leaves = Vec::new();
    collect_leaves(table, "", &mut leaves);
    for leaf in leaves {
        if Key::from_file_key_in::<A>(&leaf).is_none() {
            return Err(Error::UnknownKey {
                path: path.to_path_buf(),
                key: leaf,
                allowed: Key::file_keys_of::<A>(),
            });
        }
    }
    Ok(())
}

fn collect_leaves(table: &Table, prefix: &str, out: &mut Vec<String>) {
    for (name, value) in table.entries() {
        if prefix.is_empty() && name == "schema" {
            continue;
        }
        let key = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}.{name}")
        };
        match value {
            toml::Value::Table(inner) => collect_leaves(inner, &key, out),
            _ => out.push(key),
        }
    }
}

fn lookup<'a>(table: &'a Table, dotted: &str) -> Option<&'a toml::Value> {
    match dotted.split_once('.') {
        None => table.get(dotted),
        Some((head, tail)) => table.table(head).and_then(|inner| lookup(inner, tail)),
    }
}

fn scalar(value: &toml::Value) -> String {
    match value {
        toml::Value::Str(text) => text.clone(),
        toml::Value::Int(number) => number.to_string(),
        toml::Value::Float(number) => number.to_string(),
        toml::Value::Bool(flag) => flag.to_string(),
        toml::Value::Array(_) => "数组".to_string(),
        toml::Value::Table(_) => "表".to_string(),
    }
}

/// 临时目录里的配置文件；调用方负责清理。
#[cfg(test)]
pub(crate) fn temp_file(name: &str, body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("llm-config-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("建临时目录");
    let path = dir.join(name);
    std::fs::write(&path, body).expect("写临时配置");
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn higher_layers_win_and_lower_ones_are_ignored() {
        let mut sources = Sources::new();
        assert!(sources.set(
            Key::CacheDir,
            Value::Path(PathBuf::from("/default")),
            Origin::defaulted()
        ));
        assert!(sources.set(
            Key::CacheDir,
            Value::Path(PathBuf::from("/env")),
            Origin::new(Layer::Env, "CUTILE_CACHE_DIR")
        ));
        assert!(!sources.set(
            Key::CacheDir,
            Value::Path(PathBuf::from("/file")),
            Origin::new(Layer::File, "cutile.toml")
        ));
        assert_eq!(
            sources.get(Key::CacheDir),
            Some(&Value::Path(PathBuf::from("/env")))
        );
        assert_eq!(sources.origin(Key::CacheDir).unwrap().layer(), Layer::Env);
    }

    #[test]
    fn file_values_are_rebased_to_the_file_directory() {
        let path = temp_file(
            "rebase.toml",
            "schema = 1\n[cache]\ndir = \"build/cache\"\n",
        );
        let sources = file::<crate::app::Cutile>(&path).expect("可解析");
        let dir = path.parent().unwrap().join("build/cache");
        assert_eq!(sources.get(Key::CacheDir), Some(&Value::Path(dir)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_keys_are_errors_not_noise() {
        let path = temp_file("unknown.toml", "schema = 1\n[cache]\ndirekt = \"x\"\n");
        let error = file::<crate::app::Cutile>(&path).unwrap_err();
        assert!(matches!(error, Error::UnknownKey { .. }), "{error}");
        assert!(error.to_string().contains("cache.direkt"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn schema_is_optional_but_checked_when_present() {
        let path = temp_file("noschema.toml", "[cache]\ndir = \"x\"\n");
        assert!(
            file::<crate::app::Cutile>(&path).is_ok(),
            "缺省 schema 按当前版本处理"
        );
        let path = temp_file("badschema.toml", "schema = 2\n");
        assert!(matches!(
            file::<crate::app::Cutile>(&path).unwrap_err(),
            Error::Schema { .. }
        ));
    }

    #[test]
    fn wrong_shapes_name_the_key_and_the_file() {
        let path = temp_file("shape.toml", "schema = 1\n[cache]\ndir = 3\n");
        let error = file::<crate::app::Cutile>(&path).unwrap_err();
        assert!(matches!(
            error,
            Error::BadValue {
                key: Key::CacheDir,
                ..
            }
        ));
        assert!(error.to_string().contains("cache.dir"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn env_layer_reads_the_first_present_name() {
        let roots = Roots::new("/work");
        let sources = env::<crate::app::Cutile>(
            |name| match name {
                "CUTILE_CACHE" => Some("legacy/cache".to_string()),
                _ => None,
            },
            &roots,
        )
        .expect("环境变量合法");
        assert_eq!(
            sources.get(Key::CacheDir),
            Some(&Value::Path(PathBuf::from("/work/legacy/cache")))
        );
        assert_eq!(
            sources.origin(Key::CacheDir).unwrap().detail(),
            "CUTILE_CACHE"
        );
    }

    #[test]
    fn env_layer_rejects_bad_values_loudly() {
        let roots = Roots::new("/work");
        let error = env::<crate::app::Cutile>(
            |name| (name == "CUTILE_WORK_DIR").then(|| "   ".to_string()),
            &roots,
        )
        .unwrap_err();
        assert!(error.to_string().contains("CUTILE_WORK_DIR"), "{error}");
    }

    #[test]
    fn cli_overrides_are_typed_and_rebased() {
        let roots = Roots::new("/work");
        let (key, value, origin) =
            cli_override::<crate::app::Cutile>("work.dir=tmp/w", &roots).unwrap();
        assert_eq!(key, Key::WorkDir);
        assert_eq!(value, Value::Path(PathBuf::from("/work/tmp/w")));
        assert_eq!(origin.layer(), Layer::Cli);
        assert!(cli_override::<crate::app::Cutile>("work.dir", &roots).is_err());
        assert!(cli_override::<crate::app::Cutile>("work.nope=x", &roots).is_err());
    }

    /// 只登记 `data.root` 的应用：cutile 的键对它来说不是"历史兼容名"，而是写错了应用。
    struct Demo;

    impl App for Demo {
        const DIR: &'static str = "demo";
        const FILE: &'static str = "demo.toml";
        const ENV_PREFIX: &'static str = "DEMO";
        const KEYS: &'static [Key] = &[Key::DataRoot];
    }

    #[test]
    fn another_application_rejects_keys_it_never_registered() {
        let path = temp_file("demo-foreign.toml", "schema = 1\n[cache]\ndir = \"x\"\n");
        let error = file::<Demo>(&path).unwrap_err();
        assert!(matches!(error, Error::UnknownKey { .. }), "{error}");
        let text = error.to_string();
        assert!(text.contains("cache.dir"), "{text}");
        assert!(
            !text.contains("work.dir"),
            "错误信息只该列本应用登记的键：{text}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn another_application_reads_only_its_own_environment_variables() {
        let roots = Roots::new("/work");
        let sources = env::<Demo>(
            |name| match name {
                "DEMO_DATA_DIR" => Some("data/layer".to_string()),
                // 本应用没登记 `cache.dir`，这个名字连读都不该读。
                "DEMO_CACHE_DIR" => Some("/cache".to_string()),
                _ => None,
            },
            &roots,
        )
        .expect("合法");
        assert_eq!(sources.get(Key::CacheDir), None, "`cache.dir` 不属于 demo");
        assert_eq!(
            sources.get(Key::DataRoot),
            Some(&Value::Path(PathBuf::from("/work/data/layer")))
        );
    }

    #[test]
    fn a_cli_override_outside_the_application_key_set_names_the_available_keys() {
        let roots = Roots::new("/work");
        let error = cli_override::<Demo>("cache.dir=/tmp/x", &roots).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("cache.dir"), "{text}");
        assert!(text.contains("data.root"), "要给出本应用可用的键：{text}");
        let (key, ..) = cli_override::<Demo>("data.root=/tmp/d", &roots).expect("合法");
        assert_eq!(key, Key::DataRoot);
    }
}
