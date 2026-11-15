//! 调用参数：三个目录开关的**解析策略**——默认值、`CUTILE_*` 环境变量、`--set key=value`。
//!
//! 这是全仓**唯一**读 `CUTILE_*` 与 `--set` 的地方。优先级从低到高：
//!
//! ```text
//! 内置默认  <  环境变量 CUTILE_*  <  命令行 --set key=value
//! ```
//!
//! 每个值都带来源，[`Settings::describe`] 逐键打印"值 ← 来源"，因此"为什么这次用的是这个
//! 缓存目录"永远可以当场回答。
//!
//! 为什么策略住在这里而不是内核里：`cutile` 只知道"我要一个缓存目录、一个工作目录、一个可选的
//! 数据根"（`cutile::dirs::Dirs`），**不该决定它们在哪个盘上**。默认值面向发布后的二进制
//! （`~/Library/Caches/cutile/…`），是**部署策略**；`--set` 是**命令行文法**；两者都属于本层。
//! 内核因此只依赖 `std::path`，可以被任何嵌入方接管（测试给临时目录，服务给容器卷）。
//!
//! 它曾经是独立成员 `llm-config` 的一半（ADR 0006 → 0020），先并进 `cutile`，随后第四层
//! （用户级配置文件）在 ADR 0022 被删除；ADR 0023 再把剩下的策略整块上移到本层——`cutile`
//! 从此只收纯数据，配置不再与内核互相牵连。
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
//! 同一个二进制在任何目录运行都命中同一份缓存；需要偏离默认时用 `CUTILE_*` 或 `--set`。
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

/// 三个键的全部投影一次写完：新增一个键只加一行。
///
/// 名字表只有这一份：点号名、环境变量后缀、说明、槽位下标全部由它派生，`describe` 与
/// "键未登记"报错里列出的清单也是同一张表，因此不会出现"加了一处、忘了另一处"。
macro_rules! keys {
    ($($variant:ident => $dotted:literal, [$($env:literal),*], $doc:literal;)*) => {
        /// 配置键闭集。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Key {
            $($variant,)*
        }

        impl Key {
            /// 全部键，顺序即 `describe` 与报错清单的顺序。
            const KEYS: [Self; <[()]>::len(&[$(keys!(@one $variant)),*])] = [$(Self::$variant,)*];

            /// `--set` 里的点号名。
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

keys! {
    CacheDir => "cache.dir", ["CACHE_DIR", "CACHE"], "编译产物缓存的根目录";
    WorkDir => "work.dir", ["WORK_DIR"], "编译与执行时的临时工作目录";
    DataRoot => "data.root", ["DATA_DIR"], "声明层 / 档案层 / 片段层的替代根；未设置表示用内嵌数据";
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
            Self::Env(_) => 1,
            Self::Cli(_) => 2,
        }
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => formatter.write_str("内置默认值"),
            Self::Env(name) => write!(formatter, "{name}（env）"),
            Self::Cli(text) => write!(formatter, "{text}（cli）"),
        }
    }
}

/// 一个已确定的值与它的来源。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Resolved {
    value: PathBuf,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    roots: Roots,
    slots: [Option<Resolved>; KEY_COUNT],
}

impl Settings {
    /// 生产入口：环境变量 → 默认值；命令行覆盖由 [`Settings::apply_arguments`] 注入。
    ///
    /// 非 UTF-8 的环境变量视为未设置（`std::env::var` 的语义）。
    pub fn load() -> Result<Self> {
        Self::load_with(&Roots::discover(), |name| std::env::var(name).ok())
    }

    /// 可注入入口：`lookup` 决定环境变量从哪读，`roots` 决定默认目录在哪。
    pub fn load_with(roots: &Roots, lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let mut settings = Self::defaults(roots);
        settings.read_env(lookup)?;
        Ok(settings)
    }

    /// 只用默认值（不读环境变量）。
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

    /// 摘出全部 `--set key=value` 并写入设置，返回其余参数。
    ///
    /// # Errors
    ///
    /// `--set` 后面没有值、键未登记、缺 `=`、或值为空时返回 `invalid`。
    pub fn apply_arguments(&mut self, args: &[String]) -> Result<Vec<String>> {
        let mut rest = Vec::with_capacity(args.len());
        let mut index = 0;
        while index < args.len() {
            index = if args[index] == "--set" {
                self.apply_at(args, index)?
            } else {
                rest.push(args[index].clone());
                index + 1
            };
        }
        Ok(rest)
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
        let value = path_of_text(key, &origin, raw)?;
        let value = rebase(value, self.roots.base());
        self.store(key, value, origin);
        Ok(())
    }

    /// 缓存根目录。
    pub fn cache_dir(&self) -> PathBuf {
        self.resolved(Key::CacheDir)
            .map(|(value, _)| value)
            .unwrap_or_else(|| self.roots.default_cache_dir(APP))
    }

    /// 临时工作目录。
    pub fn work_dir(&self) -> PathBuf {
        self.resolved(Key::WorkDir)
            .map(|(value, _)| value)
            .unwrap_or_else(|| self.roots.default_work_dir(APP))
    }

    /// 声明层 / 档案层 / 片段层的替代根目录；`None` 表示用被调用方的内置数据。
    pub fn data_root(&self) -> Option<&Path> {
        self.slots[Key::DataRoot.index()]
            .as_ref()
            .map(|slot| slot.value.as_path())
    }

    /// 逐键打印"值 ← 来源"，供 `doctor` 与故障排查使用。
    pub fn describe(&self) -> String {
        let mut out = String::from("[配置] 默认 < 环境变量 < 命令行\n");
        for key in Key::KEYS {
            out.push_str(&self.row(key));
        }
        out
    }

    /// 一个键的一行；没有默认值又没设置时写"未设置"。
    fn row(&self, key: Key) -> String {
        let (value, origin) = match self.resolved(key) {
            Some((value, origin)) => (value.display().to_string(), origin.to_string()),
            None => ("-".to_string(), "未设置".to_string()),
        };
        format!("  {:<10} = {:<28} ← {origin}\n", key.dotted(), value)
    }

    /// 某个键的最终取值与来源：设置过的用它，没设置的只有 `cache.dir` / `work.dir` 有默认值。
    fn resolved(&self, key: Key) -> Option<(PathBuf, Origin)> {
        if let Some(slot) = &self.slots[key.index()] {
            return Some((slot.value.clone(), slot.origin.clone()));
        }
        default_of(key, &self.roots).map(|value| (value, Origin::Default))
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
            let value = path_of_text(key, &origin, &text)?;
            let value = rebase(value, self.roots.base());
            self.store(key, value, origin);
        }
        Ok(())
    }

    /// 写入一个值；只有优先级不低于已有来源时才生效。
    fn store(&mut self, key: Key, value: PathBuf, origin: Origin) {
        let slot = &mut self.slots[key.index()];
        if slot
            .as_ref()
            .is_none_or(|existing| origin.rank() >= existing.origin.rank())
        {
            *slot = Some(Resolved { value, origin });
        }
    }
}

/// 键的默认值：只有 `cache.dir` 与 `work.dir` 有，`data.root` 没有。
fn default_of(key: Key, roots: &Roots) -> Option<PathBuf> {
    match key {
        Key::CacheDir => Some(roots.default_cache_dir(APP)),
        Key::WorkDir => Some(roots.default_work_dir(APP)),
        Key::DataRoot => None,
    }
}

/// 点号名 → 键；未登记的名字带清单报错。
fn dotted_key(name: &str) -> Result<Key> {
    Key::parse(name).ok_or_else(|| {
        invalid(format!(
            "`--set` 里的键 `{name}` 未登记（登记的是：{}）",
            Key::dotted_names().join("、")
        ))
    })
}

/// 非空路径；空串与纯空白视为没写。
fn path_of_text(key: Key, origin: &Origin, text: &str) -> Result<PathBuf> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "{origin}提供的 `{}`（{}）非法：路径为空",
                key.dotted(),
                key.doc()
            ),
        )
        .with("cli::settings"));
    }
    Ok(PathBuf::from(text))
}

/// 相对路径按 `base` 解析；绝对路径原样返回。
fn rebase(path: PathBuf, base: &Path) -> PathBuf {
    if path.is_relative() {
        base.join(path)
    } else {
        path
    }
}

/// 一条配置错误：上下文固定为 `cli::settings`。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with("cli::settings")
}

#[cfg(test)]
#[path = "../tests/settings_tests.rs"]
mod tests;
