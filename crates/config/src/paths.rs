//! 目录解析：用户级缓存目录、配置目录，以及相对路径的解析基准。
//!
//! 默认值全部是**用户级绝对路径**，与仓库布局、进程启动目录都无关：装好的二进制在任意目录运行，
//! 缓存与配置都落在同一处，不会在 CWD 里造出 `target/`。
//!
//! | 平台 | 缓存 | 配置 |
//! |---|---|---|
//! | macOS | `~/Library/Caches/cutile/cache` | `~/Library/Application Support/cutile/cutile.toml` |
//! | Linux/BSD | `${XDG_CACHE_HOME:-~/.cache}/cutile/cache` | `${XDG_CONFIG_HOME:-~/.config}/cutile/cutile.toml` |
//! | Windows | `%LOCALAPPDATA%\cutile\cache` | `%APPDATA%\cutile\cutile.toml` |

use std::path::{Path, PathBuf};

/// 路径解析的根。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roots {
    cache_home: PathBuf,
    config_home: PathBuf,
    base: PathBuf,
}

impl Roots {
    /// 用同一个目录充当缓存根、配置根与相对路径基准；不读任何环境变量（测试与嵌入方用这个）。
    pub fn new(home: impl Into<PathBuf>) -> Self {
        let home = home.into();
        Self {
            cache_home: home.clone(),
            config_home: home.clone(),
            base: home,
        }
    }

    /// 生产入口：缓存与配置取平台标准目录，相对路径以进程启动目录为基准。
    pub fn discover() -> Self {
        Self {
            cache_home: cache_home(),
            config_home: config_home(),
            base: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    /// 指定相对路径的解析基准（配置文件里的相对路径仍按文件所在目录解析）。
    pub fn with_base(mut self, base: impl Into<PathBuf>) -> Self {
        self.base = base.into();
        self
    }

    /// 缓存根（其下是 `cutile/cache` 与 `cutile/work`）。
    pub fn cache_home(&self) -> &Path {
        &self.cache_home
    }

    /// 配置根（其下是 `cutile/cutile.toml`）。
    pub fn config_home(&self) -> &Path {
        &self.config_home
    }

    /// 环境变量与 `--set` 里相对路径的解析基准：进程启动目录，符合 shell 习惯。
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// 默认缓存目录：`<cache home>/<app>/cache`。
    ///
    /// `app` 来自 [`crate::App::DIR`]：路径层只负责“平台标准目录 + 一层应用子目录”，
    /// 它不知道应用叫什么，也不该知道。
    pub fn default_cache_dir(&self, app: &str) -> PathBuf {
        self.cache_home.join(app).join("cache")
    }

    /// 默认工作目录：`<cache home>/<app>/work`（可被系统清理，不承载事实数据）。
    pub fn default_work_dir(&self, app: &str) -> PathBuf {
        self.cache_home.join(app).join("work")
    }

    /// 用户级配置文件：`<config home>/<app>/<file>`。
    pub fn user_config_file(&self, app: &str, file: &str) -> PathBuf {
        self.config_home.join(app).join(file)
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

#[cfg(target_os = "macos")]
fn config_home() -> PathBuf {
    home()
        .map(|home| home.join("Library").join("Application Support"))
        .unwrap_or_else(fallback_home)
}

#[cfg(windows)]
fn config_home() -> PathBuf {
    env_dir("APPDATA")
        .or_else(|| home().map(|home| home.join("AppData").join("Roaming")))
        .unwrap_or_else(fallback_home)
}

#[cfg(not(any(target_os = "macos", windows)))]
fn config_home() -> PathBuf {
    env_dir("XDG_CONFIG_HOME")
        .or_else(|| home().map(|home| home.join(".config")))
        .unwrap_or_else(fallback_home)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_user_level_and_cwd_independent() {
        let roots = Roots::new("/home/u");
        assert_eq!(
            roots.default_cache_dir("cutile"),
            PathBuf::from("/home/u/cutile/cache")
        );
        assert_eq!(
            roots.default_work_dir("cutile"),
            PathBuf::from("/home/u/cutile/work")
        );
        assert_eq!(
            roots.user_config_file("cutile", "cutile.toml"),
            PathBuf::from("/home/u/cutile/cutile.toml")
        );
        assert_eq!(roots.base(), Path::new("/home/u"));
    }

    #[test]
    fn the_app_name_is_only_a_subdirectory() {
        let roots = Roots::new("/home/u");
        assert_eq!(
            roots.default_cache_dir("demo"),
            PathBuf::from("/home/u/demo/cache")
        );
        assert!(
            !roots
                .default_cache_dir("demo")
                .to_string_lossy()
                .contains("cutile"),
            "路径层不得自己拼应用名"
        );
    }

    #[test]
    fn the_relative_base_can_be_changed_without_moving_the_directories() {
        let roots = Roots::new("/home/u").with_base("/work/demo");
        assert_eq!(roots.base(), Path::new("/work/demo"));
        assert_eq!(
            roots.default_cache_dir("cutile"),
            PathBuf::from("/home/u/cutile/cache")
        );
    }

    #[test]
    fn discovery_never_anchors_to_the_repository_checkout() {
        let roots = Roots::discover();
        let cache = roots.default_cache_dir("cutile");
        assert!(cache.is_absolute(), "{}", cache.display());
        assert!(
            !cache.starts_with(env!("CARGO_MANIFEST_DIR")),
            "{}",
            cache.display()
        );
        assert!(
            roots
                .user_config_file("cutile", "cutile.toml")
                .is_absolute()
        );
    }
}
