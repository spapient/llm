//! 应用身份：目录名、配置文件名、环境变量前缀、键闭集。
//!
//! 配置里有两类东西，必须分开：
//!
//! | | 归谁 | 例子 |
//! |---|---|---|
//! | **机制**：分层、优先级、来源、键闭集、路径解析 | 本 crate | `默认 < 文件 < 环境变量 < --set` |
//! | **身份**：这个应用叫什么、认哪些键 | [`App`] 的实现 | `cutile.toml`、`CUTILE_CACHE_DIR` |
//!
//! 分开之前，`llm` / `train` / `serve` 想读自己的配置只有两条路：寄生在 `CUTILE_*` 下，
//! 或者把分层逻辑复制一遍。把身份做成类型参数后，两者都不必——`Settings::<App>::load()`
//! 就够，且 `docs/ARCHITECTURE.md` 的"配置只有一个来源"依然成立。
//!
//! # 键集也是身份的一部分
//!
//! [`App::KEYS`] 让"另一个应用不该继承 cutile 的旋钮"从约定变成可执行事实：
//! 文件层、环境变量层、`--set` 三层都只认本应用登记过的键，别处的键会以
//! [`crate::Error::UnknownKey`] 直接失败，而不是被静默接受成"看起来设置过"。
//!
//! 没有为每个键单独引入标记 trait（例如 `HasCacheDir`）是刻意的：那会让"某个键存在"这件
//! 事被编码三次（[`crate::Key`] 的变体、`KEYS` 的成员、标记 trait 的实现），而新增一个键
//! 本该只改一处。读默认路径的方法（`cache_dir` 等）对所有应用都成立——路径由 [`App::DIR`]
//! 推出，差异只在于能否被配置覆盖。
//!
//! # 例子
//!
//! ```
//! use llm_config::{App, Cutile, Key, Roots, Settings};
//!
//! struct Demo;
//!
//! impl App for Demo {
//!     const DIR: &'static str = "demo";
//!     const FILE: &'static str = "demo.toml";
//!     const ENV_PREFIX: &'static str = "DEMO";
//!     const KEYS: &'static [Key] = &[Key::DataRoot];
//! }
//!
//! assert!(Key::DataRoot.accepted_by::<Demo>());
//! assert!(!Key::CacheDir.accepted_by::<Demo>(), "别的应用登记的键不会自动继承");
//! assert!(Key::CacheDir.accepted_by::<Cutile>());
//!
//! let settings = Settings::<Demo>::defaults(&Roots::new("/tmp/demo-home"));
//! assert!(settings.describe().contains("data.root"));
//! assert!(!settings.describe().contains("cache.dir"), "没登记的键不进诊断输出");
//! ```

use crate::key::Key;

/// 一个应用的身份：用户级目录、配置文件名、环境变量前缀与键闭集。
///
/// 实现应当是零尺寸的标记类型（`struct Llm;`）：它不携带状态，只提供编译期常量。
pub trait App {
    /// 用户级目录名：`<cache home>/<DIR>/{cache,work}`、`<config home>/<DIR>/<FILE>`。
    const DIR: &'static str;

    /// 配置文件名（位于 `<config home>/<DIR>/` 下）。
    const FILE: &'static str;

    /// 环境变量前缀（不含下划线）；完整变量名是 `<前缀>_<后缀>`，
    /// 后缀由 [`crate::Key::env_suffixes`] 给出。
    const ENV_PREFIX: &'static str;

    /// 本应用接受的配置键；文件、环境变量与 `--set` 三层都只认这里登记的键。
    ///
    /// 顺序即诊断输出（[`crate::Settings::describe`]）的顺序，因此应当是稳定的。
    const KEYS: &'static [Key];
}

/// `cutile` 的身份：也是 [`crate::Settings`] 的默认类型参数。
///
/// 默认参数的意义是"既有调用点不用改"：`Settings::load()` 仍然读 `cutile.toml` 与
/// `CUTILE_*`，只有想换身份的地方才写 `Settings::<其它应用>::load()`。
pub struct Cutile;

impl App for Cutile {
    const DIR: &'static str = "cutile";
    const FILE: &'static str = "cutile.toml";
    const ENV_PREFIX: &'static str = "CUTILE";
    const KEYS: &'static [Key] = &Key::ALL;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_is_a_handful_of_constants_and_nothing_else() {
        assert_eq!(Cutile::DIR, "cutile");
        assert_eq!(Cutile::FILE, "cutile.toml");
        assert_eq!(Cutile::ENV_PREFIX, "CUTILE");
        assert_eq!(std::mem::size_of::<Cutile>(), 0, "身份不携带状态");
    }

    #[test]
    fn the_file_name_lives_inside_the_directory() {
        assert_eq!(Cutile::FILE, format!("{}.toml", Cutile::DIR));
    }

    #[test]
    fn cutile_accepts_every_registered_key() {
        for key in Key::ALL {
            assert!(
                key.accepted_by::<Cutile>(),
                "{} 应当属于 cutile",
                key.file_key()
            );
        }
    }

    #[test]
    fn a_key_is_not_inherited_from_another_application() {
        struct Demo;

        impl App for Demo {
            const DIR: &'static str = "demo";
            const FILE: &'static str = "demo.toml";
            const ENV_PREFIX: &'static str = "DEMO";
            const KEYS: &'static [Key] = &[Key::DataRoot];
        }

        assert!(Key::DataRoot.accepted_by::<Demo>());
        assert!(
            !Key::CacheDir.accepted_by::<Demo>(),
            "cutile 的键不是 demo 的键"
        );
        assert!(!Key::WorkDir.accepted_by::<Demo>());
    }

    #[test]
    fn an_application_may_register_no_keys_at_all() {
        struct Bare;

        impl App for Bare {
            const DIR: &'static str = "bare";
            const FILE: &'static str = "bare.toml";
            const ENV_PREFIX: &'static str = "BARE";
            const KEYS: &'static [Key] = &[];
        }

        assert!(Bare::KEYS.is_empty());
        for key in Key::ALL {
            assert!(
                !key.accepted_by::<Bare>(),
                "{} 不该属于 bare",
                key.file_key()
            );
        }
    }
}
