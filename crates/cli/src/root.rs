//! 系统目录与路径根：缓存根、相对路径基准、平台默认目录、配置文件的候选位置。
//!
//! 本模块只回答"东西放在哪"，不回答"用哪个值"：优先级与来源追踪在 [`crate::settings`]，
//! 键闭集与取值类型在 [`crate::key`]。分出来是因为平台分支（macOS / Windows / 其余）与
//! 配置文件发现顺序都是**部署事实**，改动它们不该碰到解析策略。

use std::path::{Path, PathBuf};

/// 应用子目录名：默认目录是 `<cache home>/cutile/{cache,work}`。
pub(crate) const APP: &str = "cutile";

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

pub(crate) fn home() -> Option<PathBuf> {
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
