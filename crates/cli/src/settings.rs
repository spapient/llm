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
//! 它曾经是独立成员 `llm-config` 的一半，先并进 `cutile`，随后第四层（用户级配置文件）在
//! `docs/adr/0022-build-time-corpora.md` 被删除；`docs/adr/0023-settings-to-cli.md` 再把剩下的
//! 策略整块上移到本层——`cutile` 从此只收纯数据。ADR 0048 恢复了配置文件层：语料仍在构建期解析，配置文件的 schema 只覆盖
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

use crate::key::APP;
pub use crate::key::Roots;
use crate::key::{
    KEY_COUNT, Key, Origin, Resolved, dotted_key, invalid, rebase_value, value_of_text,
};
use diagnostic::Result;
use std::path::{Path, PathBuf};

pub use crate::key::SettingValue;

/// 解析后的设置：值 + 来源 + 路径根。每个进程加载一次，之后按引用传递。
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    roots: Roots,
    slots: [Option<Resolved>; KEY_COUNT],
}

impl Settings {
    /// 生产入口：配置文件（`CUTILE_CONFIG` 或默认发现）→ 环境变量 → 默认值；
    /// 命令行覆盖由 [`Settings::apply_arguments`] 注入。
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
        self.path(Key::CacheDir)
            .unwrap_or_else(|| self.roots.default_cache_dir(APP))
    }

    /// 临时工作目录。
    pub fn work_dir(&self) -> PathBuf {
        self.path(Key::WorkDir)
            .unwrap_or_else(|| self.roots.default_work_dir(APP))
    }

    /// 槽位里的路径取值；不做来源追踪也不复制来源——这三个访问器在每次启动的必经路径上，
    /// 只有 `describe` 需要来源，因此只在那里走 [`Settings::resolved`]。
    fn path(&self, key: Key) -> Option<PathBuf> {
        self.slots[key.index()]
            .as_ref()
            .and_then(|slot| slot.value.as_path())
            .map(Path::to_path_buf)
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

    /// 写入一个值；只有优先级不低于已有来源时才生效。
    pub(crate) fn store(&mut self, key: Key, value: SettingValue, origin: Origin) {
        let slot = &mut self.slots[key.index()];
        if slot
            .as_ref()
            .is_none_or(|existing| origin.rank() >= existing.origin.rank())
        {
            *slot = Some(Resolved { value, origin });
        }
    }
}

/// 解析策略的单元测试：默认目录、优先级、来源、配置文件层。
#[cfg(test)]
#[path = "../tests/settings.rs"]
mod tests;
