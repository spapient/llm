//! 配置键闭集。
//!
//! 新增配置项的唯一入口：在 [`Key`] 里加一个变体，并在 [`Key::ALL`]、[`Key::file_key`]、
//! [`Key::env_vars`]、[`Key::kind`]、[`Key::doc`] 里各补一行。
//! 配置文件里出现未登记的键会直接报错（`Error::UnknownKey`），因此不存在"某个模块偷偷读了一个
//! 别处不知道的配置"这种情况。
//!
//! 反向约束同样重要：**没有任何消费方的键不许登记**。一个转了不生效的旋钮比没有旋钮更糟，
//! 它会让诊断输出里的"值 ← 来源"变成假承诺。

use crate::app::App;
use crate::value::Kind;

const CACHE_SUFFIXES: &[&str] = &["CACHE_DIR", "CACHE"];
const WORK_SUFFIXES: &[&str] = &["WORK_DIR"];
const DATA_SUFFIXES: &[&str] = &["DATA_DIR"];

/// 配置键闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    /// 编译产物缓存的根目录。
    CacheDir,
    /// 编译与执行时的临时工作目录。
    WorkDir,
    /// 声明层 / 档案层 / 片段层的替代根目录；未设置表示使用 `cutile` 的编译期内嵌数据。
    DataRoot,
}

impl Key {
    /// 闭集全量；遍历它的地方（配置文件、环境变量、诊断输出）永远覆盖全部键。
    pub const ALL: [Self; 3] = [Self::CacheDir, Self::WorkDir, Self::DataRoot];

    /// 配置文件里的键（`[cache] dir = ...` 写成 `cache.dir`）。
    pub const fn file_key(self) -> &'static str {
        match self {
            Self::CacheDir => "cache.dir",
            Self::WorkDir => "work.dir",
            Self::DataRoot => "data.root",
        }
    }

    /// 环境变量名的**后缀**（不含应用前缀）；第一个是正式名，其余是历史兼容名。
    ///
    /// 只登记后缀的原因：键闭集是机制，前缀是身份（[`App::ENV_PREFIX`]），
    /// 把两者拼在一起的写法会让“新增一个应用”变成“复制一遍键表”。
    pub const fn env_suffixes(self) -> &'static [&'static str] {
        match self {
            Self::CacheDir => CACHE_SUFFIXES,
            Self::WorkDir => WORK_SUFFIXES,
            Self::DataRoot => DATA_SUFFIXES,
        }
    }

    /// 某个应用下的完整环境变量名：`<前缀>_<后缀>`，按优先级排序。
    ///
    /// 同一个键闭集因此可以被多个应用复用：`Key::CacheDir.env_vars::<Cutile>()` 得到
    /// `CUTILE_CACHE_DIR`，`::<Llm>()` 得到 `LLM_CACHE_DIR`，而键的定义只有一份。
    pub fn env_vars<A: App>(self) -> Vec<String> {
        self.env_suffixes()
            .iter()
            .map(|suffix| format!("{}_{suffix}", A::ENV_PREFIX))
            .collect()
    }

    /// 取值形状。
    pub const fn kind(self) -> Kind {
        match self {
            Self::CacheDir | Self::WorkDir | Self::DataRoot => Kind::Path,
        }
    }

    /// 一句话说明，供文档与诊断输出使用。
    pub const fn doc(self) -> &'static str {
        match self {
            Self::CacheDir => "编译产物缓存目录",
            Self::WorkDir => "临时工作目录",
            Self::DataRoot => "声明层/档案层/片段层的替代根目录",
        }
    }

    /// 这个键是否属于某个应用（见 [`App::KEYS`]）。
    ///
    /// 文件层、环境变量层与 `--set` 都先过这道判定：不属于本应用的键不是"历史兼容名"，
    /// 而是**写错了应用**，应当直接失败。
    pub fn accepted_by<A: App>(self) -> bool {
        A::KEYS.contains(&self)
    }

    /// 某个应用登记的键（点号形式），供错误信息与诊断列举。
    pub fn file_keys_of<A: App>() -> Vec<&'static str> {
        A::KEYS.iter().map(|key| key.file_key()).collect()
    }

    /// 按配置文件里的键解析（不限应用）。
    pub fn from_file_key(text: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|key| key.file_key() == text.trim())
    }

    /// 按配置文件里的键解析，且限定在某个应用登记的键内。
    pub fn from_file_key_in<A: App>(text: &str) -> Option<Self> {
        A::KEYS
            .iter()
            .copied()
            .find(|key| key.file_key() == text.trim())
    }

    /// 按环境变量名解析（在某个应用的前缀下，且限定在它登记的键内）。
    pub fn from_env_var<A: App>(name: &str) -> Option<Self> {
        A::KEYS.iter().copied().find(|key| {
            key.env_vars::<A>()
                .iter()
                .any(|candidate| candidate == name)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ALL` 是键闭集里唯一没有穷尽 `match` 保护的事实：漏写一行不会编译报错，只会让新键从
    /// 配置文件、环境变量与诊断输出里同时消失。这里把表的内容逐项锁定，加键时必须显式改这里。
    #[test]
    fn the_key_table_lists_every_variant() {
        let file_keys: Vec<&str> = Key::ALL.iter().map(|key| key.file_key()).collect();
        assert_eq!(file_keys, ["cache.dir", "work.dir", "data.root"]);
    }

    #[test]
    fn every_key_round_trips_through_its_file_key() {
        for key in Key::ALL {
            assert_eq!(Key::from_file_key(key.file_key()), Some(key));
        }
    }

    #[test]
    fn every_key_round_trips_through_its_env_names() {
        for key in Key::ALL {
            for name in key.env_vars::<crate::app::Cutile>() {
                assert_eq!(
                    Key::from_env_var::<crate::app::Cutile>(&name),
                    Some(key),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn the_prefix_comes_from_the_app_not_from_the_key() {
        struct Demo;
        impl App for Demo {
            const DIR: &'static str = "demo";
            const FILE: &'static str = "demo.toml";
            const ENV_PREFIX: &'static str = "DEMO";
            const KEYS: &'static [Key] = &Key::ALL;
        }

        assert_eq!(
            Key::CacheDir.env_vars::<Demo>(),
            ["DEMO_CACHE_DIR", "DEMO_CACHE"]
        );
        assert_eq!(
            Key::from_env_var::<Demo>("DEMO_WORK_DIR"),
            Some(Key::WorkDir)
        );
        assert_eq!(
            Key::from_env_var::<Demo>("CUTILE_WORK_DIR"),
            None,
            "别的应用的前缀不是“历史兼容名”"
        );
    }

    #[test]
    fn file_keys_are_unique_and_dotted() {
        let mut seen = Vec::new();
        for key in Key::ALL {
            assert!(key.file_key().contains('.'), "{}", key.file_key());
            assert!(!seen.contains(&key.file_key()), "{}", key.file_key());
            seen.push(key.file_key());
        }
    }

    #[test]
    fn unknown_names_are_not_guessed() {
        use crate::app::Cutile;
        assert_eq!(Key::from_file_key("cache.directory"), None);
        assert_eq!(Key::from_env_var::<Cutile>("CUTILE_CACHE_KIND"), None);
        assert_eq!(Key::from_file_key("plan.dir"), None);
        assert_eq!(Key::from_env_var::<Cutile>("CUTILE_LOG"), None);
    }
}
