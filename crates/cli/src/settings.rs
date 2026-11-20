//! 调用参数：配置键的**解析策略**——默认值、配置文件、`CUTILE_*` 环境变量、`--set key=value`。
//!
//! 这是全仓**唯一**读配置文件、`CUTILE_*` 与 `--set` 的地方。优先级从低到高（ADR 0048）：
//!
//! ```text
//! 内置默认  <  配置文件 cutile.toml  <  环境变量 CUTILE_*  <  命令行 --set key=value
//! ```
//!
//! 每个值都带来源，[`Settings::describe`] 逐键打印"值 ← 来源"，因此"为什么这次用的是这个
//! 缓存目录"永远可以当场回答。
//!
//! 为什么策略住在这里而不是内核里：`cutile` 只知道"我要一个缓存目录、一个工作目录、一个可选的
//! 数据根"（`cutile::dirs::Dirs`），**不该决定它们在哪个盘上**。默认值面向发布后的二进制
//! （`~/Library/Caches/cutile/…`），是**部署策略**；`--set` 是**命令行文法**；配置文件是
//! **部署形态**；三者都属于本层。内核因此只依赖 `std::path`，可以被任何嵌入方接管
//! （测试给临时目录，服务给容器卷）。
//!
//! 它曾经是独立成员 `llm-config` 的一半（ADR 0006 → 0020），先并进 `cutile`，随后第四层
//! （用户级配置文件）在 ADR 0022 被删除；ADR 0023 再把剩下的策略整块上移到本层——`cutile`
//! 从此只收纯数据。ADR 0048 恢复了配置文件层：语料仍在构建期解析，配置文件的 schema 只覆盖
//! `Key` 闭集登记的运行时策略键；`toml` 是 L7 组合根的唯一外部依赖例外（D5 修订）。
//!
//! # 部署
//!
//! 默认目录全部是**平台用户级目录**，不依赖仓库布局，也不依赖进程启动目录：
//!
//! | 平台 | 缓存 |
//! |---|---|
//! | macOS | `~/Library/Caches/cutile/cache` |
//! | Linux/BSD | `${XDG_CACHE_HOME:-~/.cache}/cutile/cache` |
//! | Windows | `%LOCALAPPDATA%\cutile\cache` |
//!
//! 同一个二进制在任何目录运行都命中同一份缓存；需要偏离默认时用配置文件、`CUTILE_*` 或 `--set`。
//!
//! # 例子
//!
//! ```
//! use llm_cli::settings::{Roots, Settings};
//!
//! let roots = Roots::new("/tmp/demo").with_base("/work");
//! let mut settings = Settings::load_with(&roots, |name| {
//!     (name == "CUTILE_WORK_DIR").then(|| "build/work".to_string())
//! })
//! .expect("没有坏值");
//!
//! // 环境变量里的相对路径按启动目录解析，不按缓存根。
//! assert_eq!(settings.work_dir(), std::path::PathBuf::from("/work/build/work"));
//!
//! // 命令行覆盖优先级最高。
//! let rest = settings
//!     .apply_arguments(&["--set".to_string(), "cache.dir=/cli/cache".to_string()])
//!     .expect("合法覆盖");
//! assert!(rest.is_empty());
//! assert_eq!(settings.cache_dir(), std::path::PathBuf::from("/cli/cache"));
//! ```

use diagnostic::{Code, Error, Result};
use std::fmt;
use std::path::{Path, PathBuf};

/// 应用子目录名：默认目录是 `<cache home>/cutile/{cache,work}`。
const APP: &str = "cutile";

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
    fn into_path(self) -> Option<PathBuf> {
        match self {
            Self::Path(path) => Some(path),
            _ => None,
        }
    }

    /// 借出路径视图。
    fn as_path(&self) -> Option<&Path> {
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
        enum Key {
            $($variant,)*
        }

        impl Key {
            /// 全部键，顺序即 `describe` 与报错清单的顺序。
            const KEYS: [Self; <[()]>::len(&[$(keys!(@one $variant)),*])] = [$(Self::$variant,)*];

            /// `--set` 与配置文件里的点号名。
            const fn dotted(self) -> &'static str {
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
            fn default_value(self, $roots: &Roots) -> Option<SettingValue> {
                match self {
                    $(Self::$variant => $default,)*
                }
            }

            /// 人读说明，出现在"键未登记"与坏值诊断里。
            const fn doc(self) -> &'static str {
                match self {
                    $(Self::$variant => $doc,)*
                }
            }

            /// 槽位下标。
            const fn index(self) -> usize {
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
const KEY_COUNT: usize = Key::KEYS.len();

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
    fn env_names(self) -> Vec<String> {
        self.suffixes()
            .iter()
            .map(|suffix| format!("{PREFIX}_{suffix}"))
            .collect()
    }
}

/// 值从哪来：层 + 可读定位。层只用于优先级与打印，不承载别的语义。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Origin {
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
    const fn rank(&self) -> u8 {
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
struct Resolved {
    value: SettingValue,
    origin: Origin,
}

/// 路径解析的根。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roots {
    cache_home: PathBuf,
    base: PathBuf,
}

impl Roots {
    /// 用同一个目录充当缓存根与相对路径基准；不读任何环境变量（测试与嵌入方用这个）。
    pub fn new(home: impl Into<PathBuf>) -> Self {
        let home = home.into();
        Self {
            cache_home: home.clone(),
            base: home,
        }
    }

    /// 生产入口：缓存取平台标准目录，相对路径以进程启动目录为基准。
    pub fn discover() -> Self {
        Self {
            cache_home: cache_home(),
            base: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    /// 指定相对路径的解析基准。
    #[must_use]
    pub fn with_base(mut self, base: impl Into<PathBuf>) -> Self {
        self.base = base.into();
        self
    }

    /// 缓存根（其下是 `cutile/cache` 与 `cutile/work`）。
    pub fn cache_home(&self) -> &Path {
        &self.cache_home
    }

    /// 环境变量与 `--set` 里相对路径的解析基准：进程启动目录，符合 shell 习惯。
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// 默认缓存目录：`<cache home>/<app>/cache`。
    pub fn default_cache_dir(&self, app: &str) -> PathBuf {
        self.cache_home.join(app).join("cache")
    }

    /// 默认工作目录：`<cache home>/<app>/work`（可被系统清理，不承载事实数据）。
    pub fn default_work_dir(&self, app: &str) -> PathBuf {
        self.cache_home.join(app).join("work")
    }
}

fn home() -> Option<PathBuf> {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .filter_map(std::env::var_os)
        .find(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(not(target_os = "macos"))]
fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn fallback_home() -> PathBuf {
    std::env::temp_dir()
}

#[cfg(target_os = "macos")]
fn cache_home() -> PathBuf {
    home()
        .map(|home| home.join("Library").join("Caches"))
        .unwrap_or_else(fallback_home)
}

#[cfg(windows)]
fn cache_home() -> PathBuf {
    env_dir("LOCALAPPDATA")
        .or_else(|| home().map(|home| home.join("AppData").join("Local")))
        .unwrap_or_else(fallback_home)
}

#[cfg(not(any(target_os = "macos", windows)))]
fn cache_home() -> PathBuf {
    env_dir("XDG_CACHE_HOME")
        .or_else(|| home().map(|home| home.join(".cache")))
        .unwrap_or_else(fallback_home)
}

/// 解析后的设置：值 + 来源 + 路径根。每个进程加载一次，之后按引用传递。
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    roots: Roots,
    slots: [Option<Resolved>; KEY_COUNT],
}

impl Settings {
    /// 生产入口：配置文件（`CUTILE_CONFIG` 或默认发现）→ 环境变量 → 默认值；
    /// 命令行覆盖由 [`Settings::apply_arguments`] 注入。
    ///
    /// 非 UTF-8 的环境变量视为未设置（`std::env::var` 的语义）。
    pub fn load() -> Result<Self> {
        let mut settings = Self::defaults(&Roots::discover());
        settings.read_config(|name| std::env::var(name).ok())?;
        settings.read_env(|name| std::env::var(name).ok())?;
        Ok(settings)
    }

    /// 可注入入口：`lookup` 决定环境变量从哪读，`roots` 决定默认目录在哪；**不读配置文件**
    /// （文件层只在生产入口 [`Settings::load`]、`--config` 或 [`Settings::load_config`] 触发）。
    pub fn load_with(roots: &Roots, lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let mut settings = Self::defaults(roots);
        settings.read_env(lookup)?;
        Ok(settings)
    }

    /// 读一份配置文件写入文件层（优先级低于环境变量与 `--set`）；`--config` 与测试用这个入口。
    ///
    /// # Errors
    ///
    /// 文件不存在 `missing`；键未登记 / 类型不符 / 坏 TOML 都是 `invalid`。
    pub fn load_config(&mut self, path: &Path) -> Result<()> {
        self.read_config_at(path)
    }

    /// 只用默认值（不读配置文件与环境变量）。
    pub fn defaults(roots: &Roots) -> Self {
        Self {
            roots: roots.clone(),
            slots: [const { None }; KEY_COUNT],
        }
    }

    /// 路径根。
    pub fn roots(&self) -> &Roots {
        &self.roots
    }

    /// 摘出全部 `--set key=value` 与 `--config <path>` 并写入设置，返回其余参数。
    ///
    /// # Errors
    ///
    /// `--set` 后面没有值、键未登记、缺 `=`、或值为空时返回 `invalid`；
    /// `--config` 后面的路径不存在时返回 `missing`。
    pub fn apply_arguments(&mut self, args: &[String]) -> Result<Vec<String>> {
        let mut rest = Vec::with_capacity(args.len());
        let mut index = 0;
        while index < args.len() {
            index = self.next_index(args, index, &mut rest)?;
        }
        Ok(rest)
    }

    /// 一条参数的下标推进：`--set` / `--config` 消费两条，其余原样留下。
    fn next_index(
        &mut self,
        args: &[String],
        index: usize,
        rest: &mut Vec<String>,
    ) -> Result<usize> {
        if args[index] == "--set" {
            self.apply_at(args, index)
        } else if args[index] == "--config" {
            self.apply_config_at(args, index)
        } else {
            rest.push(args[index].clone());
            Ok(index + 1)
        }
    }

    /// 处理 `args[index] == "--set"`，返回下一条要看的参数下标。
    fn apply_at(&mut self, args: &[String], index: usize) -> Result<usize> {
        let pair = args
            .get(index + 1)
            .ok_or_else(|| invalid("`--set` 缺少 `key=value`"))?;
        self.apply_override(pair)
            .map_err(|error| error.with("--set"))?;
        Ok(index + 2)
    }

    /// 处理 `args[index] == "--config"`，返回下一条要看的参数下标。
    fn apply_config_at(&mut self, args: &[String], index: usize) -> Result<usize> {
        let path = args
            .get(index + 1)
            .ok_or_else(|| invalid("`--config` 缺少文件路径"))?;
        self.read_config_at(Path::new(path))
            .map_err(|error| error.with("--config"))?;
        Ok(index + 2)
    }

    /// 写入一条命令行覆盖（`key=value`）。命令行层优先级最高，相对路径按启动目录解析。
    ///
    /// # Errors
    ///
    /// 键未登记、缺 `=`、或取值为空时返回 `invalid`。
    pub fn apply_override(&mut self, text: &str) -> Result<()> {
        let (name, raw) = text
            .split_once('=')
            .ok_or_else(|| invalid(format!("`--set {text}` 缺少 `=`：应写成 `key=value`")))?;
        let key = dotted_key(name.trim())?;
        let origin = Origin::Cli(format!("--set {text}"));
        let value = value_of_text(key, &origin, raw)?;
        let value = rebase_value(value, self.roots.base());
        self.store(key, value, origin);
        Ok(())
    }

    /// 缓存根目录。
    pub fn cache_dir(&self) -> PathBuf {
        self.resolved(Key::CacheDir)
            .and_then(|(value, _)| value.into_path())
            .unwrap_or_else(|| self.roots.default_cache_dir(APP))
    }

    /// 临时工作目录。
    pub fn work_dir(&self) -> PathBuf {
        self.resolved(Key::WorkDir)
            .and_then(|(value, _)| value.into_path())
            .unwrap_or_else(|| self.roots.default_work_dir(APP))
    }

    /// 声明层 / 档案层 / 片段层的替代根目录；`None` 表示用被调用方的内置数据。
    pub fn data_root(&self) -> Option<&Path> {
        self.slots[Key::DataRoot.index()]
            .as_ref()
            .and_then(|slot| slot.value.as_path())
    }

    /// 缓存容量上限；默认值与 [`cutile::cache::CACHE_LIMIT`] 相同（2 GiB）。`cache gc` 没给
    /// `--limit` 时消费这个投影，内核常量只在投影缺失时兜底。
    pub fn cache_limit_bytes(&self) -> Option<u64> {
        self.resolved(Key::CacheLimit)
            .and_then(|(value, _)| match value {
                SettingValue::Bytes(bytes) => Some(bytes),
                _ => None,
            })
    }

    /// 投影成内核只认的三目录纯数据：缓存 / 工作 / 数据根；`cutile` 不读环境变量（ADR 0023）。
    pub fn dirs(&self) -> cutile::dirs::Dirs {
        cutile::dirs::Dirs::new(
            self.cache_dir(),
            self.work_dir(),
            self.data_root().map(Path::to_path_buf),
        )
    }

    /// 逐键打印"值 ← 来源"，供 `doctor` 与故障排查使用。
    pub fn describe(&self) -> String {
        let mut out = String::from("[配置] 默认 < 配置文件 < 环境变量 < 命令行\n");
        for key in Key::KEYS {
            out.push_str(&self.row(key));
        }
        out
    }

    /// 一个键的一行；没有默认值又没设置时写"未设置"。
    fn row(&self, key: Key) -> String {
        let (value, origin) = match self.resolved(key) {
            Some((value, origin)) => (value.to_string(), origin.to_string()),
            None => ("-".to_string(), "未设置".to_string()),
        };
        format!("  {:<26} = {:<28} ← {origin}\n", key.dotted(), value)
    }

    /// 某个键的最终取值与来源：设置过的用它，没设置的看 [`Key::default_value`]。
    fn resolved(&self, key: Key) -> Option<(SettingValue, Origin)> {
        if let Some(slot) = &self.slots[key.index()] {
            return Some((slot.value.clone(), slot.origin.clone()));
        }
        key.default_value(&self.roots)
            .map(|value| (value, Origin::Default))
    }

    /// 从环境变量层收集：正式名优先，兼容名次之；相对路径按启动目录解析。
    fn read_env(&mut self, lookup: impl Fn(&str) -> Option<String>) -> Result<()> {
        for key in Key::KEYS {
            let found = key
                .env_names()
                .into_iter()
                .find_map(|name| lookup(&name).map(|text| (name, text)));
            let Some((name, text)) = found else {
                continue;
            };
            let origin = Origin::Env(name);
            let value = value_of_text(key, &origin, &text)?;
            let value = rebase_value(value, self.roots.base());
            self.store(key, value, origin);
        }
        Ok(())
    }

    /// 配置文件层发现（ADR 0048）：`CUTILE_CONFIG` → `./cutile.toml` → 平台用户目录；
    /// 只读第一份命中，文件不存在不是错误。
    fn read_config(&mut self, lookup: impl Fn(&str) -> Option<String>) -> Result<()> {
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
    fn read_config_at(&mut self, path: &Path) -> Result<()> {
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
        let value = rebase_value(value, self.roots.base());
        self.store(key, value, origin);
        Ok(())
    }

    /// 写入一个值；只有优先级不低于已有来源时才生效。
    fn store(&mut self, key: Key, value: SettingValue, origin: Origin) {
        let slot = &mut self.slots[key.index()];
        if slot
            .as_ref()
            .is_none_or(|existing| origin.rank() >= existing.origin.rank())
        {
            *slot = Some(Resolved { value, origin });
        }
    }
}

/// 配置文件发现顺序里的候选路径（不含显式 `--config` / `CUTILE_CONFIG`）。
fn discover_files() -> Vec<PathBuf> {
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

/// 点号名 → 键；未登记的名字带清单报错。
fn dotted_key(name: &str) -> Result<Key> {
    Key::parse(name).ok_or_else(|| {
        invalid(format!(
            "键 `{name}` 未登记（登记的是：{}）",
            Key::dotted_names().join("、")
        ))
    })
}

/// 按键声明的值类型解析文本；空串与纯空白视为没写。
fn value_of_text(key: Key, origin: &Origin, text: &str) -> Result<SettingValue> {
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
fn rebase_value(value: SettingValue, base: &Path) -> SettingValue {
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
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with("cli::settings")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nothing(_: &str) -> Option<String> {
        None
    }

    /// 测试用的用户根：必须是当前平台认可的绝对路径，否则 `is_absolute` 在 Windows 上为假。
    #[cfg(unix)]
    fn test_home() -> PathBuf {
        PathBuf::from("/home/u")
    }

    #[cfg(windows)]
    fn test_home() -> PathBuf {
        PathBuf::from(r"C:\Users\u")
    }

    #[test]
    fn defaults_are_absolute_user_level_paths() {
        let roots = Roots::new(test_home());
        let settings = Settings::defaults(&roots);
        assert_eq!(settings.cache_dir(), roots.default_cache_dir(APP));
        assert_eq!(settings.work_dir(), roots.default_work_dir(APP));
        assert!(settings.cache_dir().is_absolute());
        assert_eq!(settings.data_root(), None);
    }

    #[test]
    fn an_empty_environment_uses_defaults() {
        let roots = Roots::new("/home/u");
        let settings = Settings::load_with(&roots, nothing).expect("没有环境变量不是错误");
        assert_eq!(settings.cache_dir(), roots.default_cache_dir(APP));
        assert_eq!(settings.work_dir(), roots.default_work_dir(APP));
        assert_eq!(settings.data_root(), None);
    }

    #[test]
    fn env_wins_over_defaults_and_rebases_relative_paths() {
        let roots = Roots::new("/home/u").with_base("/cwd");
        let settings = Settings::load_with(&roots, |name| match name {
            "CUTILE_CACHE_DIR" => Some("/env/cache".to_string()),
            "CUTILE_WORK_DIR" => Some("env-work".to_string()),
            "CUTILE_DATA_DIR" => Some("data".to_string()),
            _ => None,
        })
        .expect("配置可加载");

        assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
        // 环境变量里的相对路径按启动目录解析，不按缓存根。
        assert_eq!(settings.work_dir(), PathBuf::from("/cwd/env-work"));
        assert_eq!(settings.data_root(), Some(Path::new("/cwd/data")));
    }

    #[test]
    fn the_legacy_cache_variable_is_still_honoured() {
        let roots = Roots::new("/home/u");
        let settings = Settings::load_with(&roots, |name| {
            (name == "CUTILE_CACHE").then(|| "legacy/cache".to_string())
        })
        .expect("配置可加载");
        assert_eq!(settings.cache_dir(), PathBuf::from("/home/u/legacy/cache"));
        assert!(
            settings.describe().contains("CUTILE_CACHE（env）"),
            "来源要写出兼容名：{}",
            settings.describe()
        );
    }

    #[test]
    fn the_formal_variable_beats_the_legacy_one() {
        let roots = Roots::new("/home/u");
        let settings = Settings::load_with(&roots, |name| match name {
            "CUTILE_CACHE_DIR" => Some("/formal".to_string()),
            "CUTILE_CACHE" => Some("/legacy".to_string()),
            _ => None,
        })
        .expect("配置可加载");
        assert_eq!(settings.cache_dir(), PathBuf::from("/formal"));
    }

    #[test]
    fn cli_overrides_beat_everything() {
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, |name| {
            (name == "CUTILE_CACHE_DIR").then(|| "/env/cache".to_string())
        })
        .expect("配置可加载");
        assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
        settings
            .apply_override("cache.dir=/cli/cache")
            .expect("覆盖");
        assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
        assert!(settings.apply_override("cache.dir").is_err());
    }

    #[test]
    fn arguments_are_split_into_overrides_and_payload() {
        let roots = Roots::new("/home/u");
        let mut settings = Settings::defaults(&roots);
        let rest = settings
            .apply_arguments(&[
                "--set".to_string(),
                "cache.dir=/cli/cache".to_string(),
                "doctor".to_string(),
                "--device".to_string(),
                "cpu".to_string(),
            ])
            .expect("合法覆盖");
        assert_eq!(rest, ["doctor", "--device", "cpu"]);
        assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
    }

    #[test]
    fn an_override_without_a_value_is_an_error() {
        let roots = Roots::new("/home/u");
        let mut settings = Settings::defaults(&roots);
        let error = settings
            .apply_arguments(&["--set".to_string()])
            .expect_err("`--set` 后面什么都没有");
        assert_eq!(error.code(), Code::Invalid);
    }

    #[test]
    fn empty_values_and_unknown_keys_are_rejected() {
        let roots = Roots::new("/home/u");
        let mut settings = Settings::defaults(&roots);
        let unknown = settings
            .apply_override("cache.dirr=/tmp")
            .expect_err("键未登记");
        assert!(
            format!("{unknown}").contains("cache.dir、work.dir、data.root"),
            "报错要列出登记过的键：{unknown}"
        );
        let empty = settings.apply_override("cache.dir=  ").expect_err("空值");
        assert!(format!("{empty}").contains("值为空"), "{empty}");
    }

    #[test]
    fn describe_lists_every_key_with_its_origin() {
        let roots = Roots::new("/home/u");
        let text = Settings::defaults(&roots).describe();
        assert!(
            text.starts_with("[配置] 默认 < 配置文件 < 环境变量 < 命令行"),
            "{text}"
        );
        for key in Key::KEYS {
            assert!(text.contains(key.dotted()), "{text}");
        }
        assert!(text.contains("内置默认值"), "{text}");
        assert!(text.contains("未设置"), "没有默认值的键要如实写出：{text}");
    }

    #[test]
    fn describe_marks_the_layer_of_every_override() {
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, |name| {
            (name == "CUTILE_WORK_DIR").then(|| "/env/work".to_string())
        })
        .expect("配置可加载");
        settings
            .apply_override("cache.dir=/cli/cache")
            .expect("覆盖");
        let text = settings.describe();
        assert!(text.contains("CUTILE_WORK_DIR（env）"), "{text}");
        assert!(text.contains("--set cache.dir=/cli/cache（cli）"), "{text}");
    }

    /// 临时配置文件路径：pid + 原子序号保证并行测试不撞名。
    fn temp_config() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "cutile-settings-{}-{}.toml",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[test]
    fn a_config_file_sets_values_and_reports_its_origin() {
        let path = temp_config();
        std::fs::write(
            &path,
            "cache.dir = \"/file/cache\"\ncache.limit_bytes = 1073741824\n",
        )
        .expect("写临时配置");
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
        settings.load_config(&path).expect("合法配置文件");
        assert_eq!(settings.cache_dir(), PathBuf::from("/file/cache"));
        assert_eq!(settings.cache_limit_bytes(), Some(1024 * 1024 * 1024));
        let text = settings.describe();
        assert!(
            text.contains("（file）") && text.contains(&path.display().to_string()),
            "来源要写出文件路径：{text}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_config_file_is_a_missing_error() {
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
        let error = settings
            .load_config(Path::new("/no/such/cutile-config.toml"))
            .expect_err("显式指定但不存在");
        assert_eq!(error.code(), Code::Missing);
    }

    #[test]
    fn a_malformed_config_file_is_rejected() {
        let path = temp_config();
        std::fs::write(&path, "cache.dir = ").expect("写坏配置");
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
        let error = settings.load_config(&path).expect_err("坏 TOML 必须报错");
        assert_eq!(error.code(), Code::Invalid);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unknown_keys_and_wrong_types_in_a_config_file_are_rejected() {
        let roots = Roots::new("/home/u");
        let unknown = temp_config();
        std::fs::write(&unknown, "no.such.key = 1").expect("写未知键配置");
        let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
        let error = settings.load_config(&unknown).expect_err("未知键必须报错");
        assert!(format!("{error}").contains("cache.dir"), "{error}");
        std::fs::remove_file(&unknown).ok();

        let mistyped = temp_config();
        std::fs::write(&mistyped, "cache.limit_bytes = \"abc\"").expect("写类型错配置");
        let error = settings.load_config(&mistyped).expect_err("类型错必须报错");
        assert!(format!("{error}").contains("非负整数"), "{error}");
        std::fs::remove_file(&mistyped).ok();
    }

    #[test]
    fn env_beats_file_and_cli_beats_env() {
        let path = temp_config();
        std::fs::write(&path, "cache.dir = \"/file/cache\"").expect("写临时配置");
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, |name| {
            (name == "CUTILE_CACHE_DIR").then(|| "/env/cache".to_string())
        })
        .expect("配置可加载");
        settings.load_config(&path).expect("合法配置文件");
        assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
        settings
            .apply_override("cache.dir=/cli/cache")
            .expect("覆盖");
        assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn arguments_are_split_into_config_overrides_and_payload() {
        let path = temp_config();
        std::fs::write(&path, "cache.dir = \"/file/cache\"").expect("写临时配置");
        let roots = Roots::new("/home/u");
        let mut settings = Settings::load_with(&roots, nothing).expect("配置可加载");
        let rest = settings
            .apply_arguments(&[
                "--config".to_string(),
                path.display().to_string(),
                "doctor".to_string(),
            ])
            .expect("合法覆盖");
        assert_eq!(rest, ["doctor"]);
        assert_eq!(settings.cache_dir(), PathBuf::from("/file/cache"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_numeric_key_has_its_current_default() {
        let roots = Roots::new("/home/u");
        let settings = Settings::load_with(&roots, nothing).expect("配置可加载");
        assert_eq!(settings.cache_limit_bytes(), Some(2 * 1024 * 1024 * 1024));
        assert!(settings.describe().contains("cache.limit_bytes"));
    }

    #[test]
    fn the_relative_base_can_be_changed_without_moving_the_directories() {
        let roots = Roots::new("/home/u").with_base("/work/demo");
        assert_eq!(roots.base(), Path::new("/work/demo"));
        assert_eq!(
            roots.default_cache_dir(APP),
            PathBuf::from("/home/u/cutile/cache")
        );
    }

    #[test]
    fn discovery_never_anchors_to_the_repository_checkout() {
        let roots = Roots::discover();
        let cache = roots.default_cache_dir(APP);
        assert!(cache.is_absolute(), "{}", cache.display());
        assert!(
            !cache.starts_with(env!("CARGO_MANIFEST_DIR")),
            "{}",
            cache.display()
        );
    }
}
