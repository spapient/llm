//! 解析后的运行时设置：每个进程只加载一次，之后按值与引用传递。
//!
//! 读取方（`cutile` 的 CLI、将来的 `llm`）只认 [`Settings`]，不再各自读环境变量或拼默认路径；
//! 这样"同一个值在不同模块取到不同结果"在结构上就不可能发生。

use crate::app::{App, Cutile};
use crate::error::Error;
use crate::key::Key;
use crate::paths::Roots;
use crate::source::{self, Origin, Sources};
use crate::value::Value;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

/// 解析后的设置：值 + 来源 + 路径根 + 应用身份。
///
/// 类型参数只决定**名字**（[`App::DIR`] / [`App::FILE`] / [`App::ENV_PREFIX`]），
/// 不改变分层规则：默认参数是 [`Cutile`]，所以 `Settings::load()` 与
/// `Settings::<Cutile>::load()` 是同一个调用。
pub struct Settings<A: App = Cutile> {
    roots: Roots,
    sources: Sources,
    app: PhantomData<A>,
}

// 手写这三个 impl 而不是 derive：derive 会给类型参数加上 `A: Debug` 这类无关约束，
// 而“应用身份”只需提供常量，不该被它拖带着实现一堆 trait。
impl<A: App> fmt::Debug for Settings<A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Settings")
            .field("roots", &self.roots)
            .field("sources", &self.sources)
            .finish()
    }
}

impl<A: App> Clone for Settings<A> {
    fn clone(&self) -> Self {
        Self {
            roots: self.roots.clone(),
            sources: self.sources.clone(),
            app: PhantomData,
        }
    }
}

impl<A: App> PartialEq for Settings<A> {
    fn eq(&self, other: &Self) -> bool {
        self.roots == other.roots && self.sources == other.sources
    }
}

impl<A: App> Settings<A> {
    /// 生产入口：用户级配置文件 → 环境变量。
    ///
    /// 命令行覆盖由调用方在加载后通过 [`Settings::apply_override`] 注入，因此这里不读 `argv`。
    /// 非 UTF-8 的环境变量视为未设置（`std::env::var` 的语义）。
    pub fn load() -> Result<Self, Error> {
        Self::load_with(&Roots::discover(), |name| std::env::var(name).ok())
    }

    /// 可注入入口：`lookup` 决定环境变量从哪读，`roots` 决定文件与默认目录在哪（测试用）。
    pub fn load_with(
        roots: &Roots,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, Error> {
        let mut sources = Sources::new();
        let user = roots.user_config_file(A::DIR, A::FILE);
        if user.is_file() {
            sources.merge(source::file::<A>(&user)?);
        }
        sources.merge(source::env::<A>(lookup, roots)?);
        Ok(Self::from_sources(roots.clone(), sources))
    }

    /// 只用默认值（不读文件与环境变量）。
    pub fn defaults(roots: &Roots) -> Self {
        Self::from_sources(roots.clone(), Sources::new())
    }

    /// 由已收集的来源构造；来源里的相对路径应已由 [`Value::rebase`] 解析。
    pub fn from_sources(roots: Roots, sources: Sources) -> Self {
        Self {
            roots,
            sources,
            app: PhantomData,
        }
    }

    /// 路径根。
    pub fn roots(&self) -> &Roots {
        &self.roots
    }

    /// 全部来源（诊断用）。
    pub fn sources(&self) -> &Sources {
        &self.sources
    }

    /// 追加一条命令行覆盖（`--set key=value` 的取值部分）。
    ///
    /// # Errors
    ///
    /// 键未登记或取值不符合形状时返回 [`Error::BadValue`]。
    pub fn apply_override(&mut self, text: &str) -> Result<(), Error> {
        let (key, value, origin) = source::cli_override::<A>(text, &self.roots)?;
        self.sources.set(key, value, origin);
        Ok(())
    }

    /// 某个键的最终取值与来源；`None` 表示该键既无默认值也未设置。
    ///
    /// 不属于本应用（[`App::KEYS`]）的键一律返回 `None`：它们本来就不该被读到。
    pub fn entry(&self, key: Key) -> Option<(Value, Origin)> {
        if !key.accepted_by::<A>() {
            return None;
        }
        if let Some(value) = self.sources.get(key) {
            let origin = self
                .sources
                .origin(key)
                .cloned()
                .unwrap_or_else(Origin::defaulted);
            return Some((value.clone(), origin));
        }
        let default = match key {
            Key::CacheDir => Value::Path(self.roots.default_cache_dir(A::DIR)),
            Key::WorkDir => Value::Path(self.roots.default_work_dir(A::DIR)),
            Key::DataRoot => return None,
        };
        Some((default, Origin::defaulted()))
    }

    /// 某个键的来源；无默认值且未设置时返回默认层。
    pub fn origin(&self, key: Key) -> Origin {
        self.entry(key)
            .map(|(_, origin)| origin)
            .unwrap_or_else(Origin::defaulted)
    }

    /// 缓存根目录。
    pub fn cache_dir(&self) -> PathBuf {
        self.path(Key::CacheDir)
            .unwrap_or_else(|| self.roots.default_cache_dir(A::DIR))
    }

    /// 临时工作目录。
    pub fn work_dir(&self) -> PathBuf {
        self.path(Key::WorkDir)
            .unwrap_or_else(|| self.roots.default_work_dir(A::DIR))
    }

    /// 声明层 / 档案层 / 片段层的替代根目录；`None` 表示用被调用方的内置数据。
    pub fn data_root(&self) -> Option<&Path> {
        self.sources.get(Key::DataRoot).and_then(Value::as_path)
    }

    /// 逐键打印"值 ← 来源"，供 `doctor` 与故障排查使用。
    pub fn describe(&self) -> String {
        let mut out = String::from("[配置] 默认 < 用户级文件 < 环境变量 < 命令行\n");
        for key in A::KEYS.iter().copied() {
            let (value, origin) = match self.entry(key) {
                Some((value, origin)) => (value.render(), origin.to_string()),
                None => ("-".to_string(), "未设置".to_string()),
            };
            out.push_str(&format!(
                "  {:<10} = {:<28} ← {}\n",
                key.file_key(),
                value,
                origin
            ));
        }
        out
    }

    fn path(&self, key: Key) -> Option<PathBuf> {
        self.sources.get(key).and_then(|value| match value {
            Value::Path(path) => Some(path.clone()),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Layer;

    #[test]
    fn defaults_are_absolute_user_level_paths() {
        let roots = Roots::new("/home/u");
        let settings = Settings::<Cutile>::defaults(&roots);
        assert_eq!(settings.cache_dir(), PathBuf::from("/home/u/cutile/cache"));
        assert_eq!(settings.work_dir(), PathBuf::from("/home/u/cutile/work"));
        assert!(settings.cache_dir().is_absolute());
        assert_eq!(settings.data_root(), None);
        assert_eq!(settings.origin(Key::CacheDir).layer(), Layer::Default);
    }

    fn temp_roots(name: &str) -> (Roots, PathBuf) {
        let home = std::env::temp_dir().join(format!("llm-config-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&home).expect("建临时目录");
        (Roots::new(&home), home)
    }

    #[test]
    fn a_directory_without_a_config_file_uses_defaults() {
        let (roots, home) = temp_roots("nofile");
        let settings =
            Settings::<Cutile>::load_with(&roots, |_| None).expect("没有配置文件不是错误");
        assert_eq!(settings.cache_dir(), roots.default_cache_dir(Cutile::DIR));
        assert_eq!(settings.origin(Key::CacheDir).layer(), Layer::Default);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn the_user_file_is_read_and_env_wins_over_it() {
        let (roots, home) = temp_roots("layers");
        let user = roots.user_config_file(Cutile::DIR, Cutile::FILE);
        std::fs::create_dir_all(user.parent().expect("父目录")).expect("建目录");
        std::fs::write(
            &user,
            "schema = 1\n[cache]\ndir = \"user-cache\"\n[data]\nroot = \"data\"\n",
        )
        .expect("写");

        let settings = Settings::<Cutile>::load_with(&roots, |name| match name {
            "CUTILE_CACHE_DIR" => Some("/env/cache".to_string()),
            "CUTILE_WORK_DIR" => Some("env-work".to_string()),
            _ => None,
        })
        .expect("配置可加载");

        assert_eq!(settings.cache_dir(), PathBuf::from("/env/cache"));
        assert_eq!(settings.origin(Key::CacheDir).layer(), Layer::Env);
        assert_eq!(settings.work_dir(), home.join("env-work"));
        assert_eq!(settings.origin(Key::WorkDir).layer(), Layer::Env);
        let data = home.join("cutile").join("data");
        assert_eq!(settings.data_root(), Some(data.as_path()));
        assert_eq!(settings.origin(Key::DataRoot).layer(), Layer::File);

        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn cli_overrides_beat_everything() {
        let (roots, home) = temp_roots("cli");
        let mut settings = Settings::<Cutile>::load_with(&roots, |name| {
            (name == "CUTILE_CACHE_DIR").then(|| "/env/cache".to_string())
        })
        .expect("配置可加载");
        settings
            .apply_override("cache.dir=/cli/cache")
            .expect("覆盖");
        assert_eq!(settings.cache_dir(), PathBuf::from("/cli/cache"));
        assert_eq!(settings.origin(Key::CacheDir).layer(), Layer::Cli);
        assert!(settings.apply_override("cache.dir").is_err());
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn describe_lists_every_key_with_its_origin() {
        let roots = Roots::new("/home/u");
        let text = Settings::<Cutile>::defaults(&roots).describe();
        for key in Key::ALL {
            assert!(text.contains(key.file_key()), "{text}");
        }
        assert!(text.contains("内置默认值"), "{text}");
    }

    /// 第二个应用：同一个配置机制，不同的名字。
    struct Demo;

    impl App for Demo {
        const DIR: &'static str = "demo";
        const FILE: &'static str = "demo.toml";
        const ENV_PREFIX: &'static str = "DEMO";
        const KEYS: &'static [Key] = &[Key::CacheDir, Key::WorkDir];
    }

    #[test]
    fn a_second_app_reads_its_own_file_and_env_prefix() {
        let (roots, home) = temp_roots("demo-app");
        let user = roots.user_config_file(Demo::DIR, Demo::FILE);
        assert!(user.ends_with("demo/demo.toml"), "{}", user.display());
        std::fs::create_dir_all(user.parent().expect("父目录")).expect("建目录");
        std::fs::write(&user, "schema = 1\n[cache]\ndir = \"demo-cache\"\n").expect("写");

        let settings = Settings::<Demo>::load_with(&roots, |name| match name {
            "DEMO_WORK_DIR" => Some("demo-work".to_string()),
            // 别的应用的变量不该被当成自己的：前缀即身份。
            "CUTILE_CACHE_DIR" => Some("/cutile-cache".to_string()),
            _ => None,
        })
        .expect("配置可加载");

        assert_eq!(settings.cache_dir(), home.join("demo").join("demo-cache"));
        assert_eq!(settings.origin(Key::CacheDir).layer(), Layer::File);
        assert_eq!(
            settings.work_dir(),
            home.join("demo-work"),
            "环境变量的相对路径按启动目录解析"
        );
        assert_eq!(settings.origin(Key::WorkDir).layer(), Layer::Env);

        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn two_apps_never_share_default_directories() {
        let roots = Roots::new("/home/u");
        assert_eq!(
            Settings::<Cutile>::defaults(&roots).cache_dir(),
            PathBuf::from("/home/u/cutile/cache")
        );
        assert_eq!(
            Settings::<Demo>::defaults(&roots).cache_dir(),
            PathBuf::from("/home/u/demo/cache")
        );
    }

    #[test]
    fn a_key_outside_the_application_set_is_invisible_even_to_diagnostics() {
        struct OnlyData;

        impl App for OnlyData {
            const DIR: &'static str = "only-data";
            const FILE: &'static str = "only-data.toml";
            const ENV_PREFIX: &'static str = "ONLY_DATA";
            const KEYS: &'static [Key] = &[Key::DataRoot];
        }

        let settings = Settings::<OnlyData>::defaults(&Roots::new("/home/u"));
        assert_eq!(
            settings.entry(Key::CacheDir),
            None,
            "没登记的键读不到，也不该冒出“看起来有默认值”的样子"
        );
        let described = settings.describe();
        assert!(described.contains("data.root"), "{described}");
        assert!(!described.contains("cache.dir"), "{described}");
    }
}
