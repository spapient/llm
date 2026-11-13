//! 配置基座：本 workspace **唯一**的"值从哪来"与"目录在哪"的来源。
//!
//! 依赖方向固定为 `llm → cutile → {op, prof, prim, llm-config}`：本 crate 只依赖 `std`，不反向依赖
//! 任何成员，也不解释任何领域语义（`ops/` 属于 `crates/op`，`devices/` 属于 `crates/prof`，
//! `prim/` 属于 `crates/prim`）。
//!
//! # 分层
//!
//! 优先级从低到高：
//!
//! ```text
//! 内置默认  <  用户级文件  <  环境变量 CUTILE_*  <  命令行 --set key=value
//! ```
//!
//! 每个值都带 [`source::Origin`]（层 + 文件路径 / 变量名 / 命令行），
//! [`Settings::describe`] 逐键打印"值 ← 来源"，因此"为什么这次用的是这个缓存目录"永远可以当场回答。
//!
//! # 部署
//!
//! 默认目录全部是**平台用户级目录**，不依赖仓库布局，也不依赖进程启动目录：
//!
//! | 平台 | 缓存 | 配置 |
//! |---|---|---|
//! | macOS | `~/Library/Caches/cutile/cache` | `~/Library/Application Support/cutile/cutile.toml` |
//! | Linux/BSD | `${XDG_CACHE_HOME:-~/.cache}/cutile/cache` | `${XDG_CONFIG_HOME:-~/.config}/cutile/cutile.toml` |
//! | Windows | `%LOCALAPPDATA%\cutile\cache` | `%APPDATA%\cutile\cutile.toml` |
//!
//! 因此同一个二进制在任何目录运行都命中同一份缓存；仓库里不存任何"默认值副本"，
//! 需要偏离默认时用用户级文件、`CUTILE_*` 或 `--set`。
//!
//! # 键闭集
//!
//! 配置项只能登记在 [`Key::ALL`]，读取只能通过 [`Key`] 常量：
//!
//! - 配置文件里出现未登记的键会直接报 [`Error::UnknownKey`]，而不是被静默忽略；
//! - 环境变量名也由 [`Key::env_vars`] 给出，不存在"某个模块自己拼一个变量名"；
//! - 新增一项必须先改本 crate：`file_key` / `env_suffixes` / `kind` / `doc` 都是对 `self` 的
//!   穷尽 `match`，漏一处就编译不过。`Key::ALL` 是唯一没人替你把关的地方（定长数组，加变体不会
//!   报错），由 `the_key_table_lists_every_variant` 逐项锁住内容。
//!
//! # 例子
//!
//! ```
//! use llm_config::{Cutile, Key, Roots, Settings};
//!
//! let roots = Roots::new("/tmp/demo");
//! let mut settings = Settings::<Cutile>::defaults(&roots);
//! assert_eq!(
//!     settings.cache_dir(),
//!     std::path::PathBuf::from("/tmp/demo/cutile/cache")
//! );
//!
//! // 命令行覆盖：相对路径按进程启动目录解析（`Roots::new` 里就是构造时给的目录）。
//! settings.apply_override("cache.dir=build/cache").expect("合法覆盖");
//! assert_eq!(
//!     settings.cache_dir(),
//!     std::path::PathBuf::from("/tmp/demo/build/cache")
//! );
//! assert_eq!(
//!     settings.origin(Key::CacheDir).layer(),
//!     llm_config::source::Layer::Cli
//! );
//! ```
//!
//! # 什么不该放进来
//!
//! - 构建与工具链配置（`Cargo.toml`、`clippy.toml`、`rustfmt.toml`、`rust-toolchain.toml`、CI）：
//!   这些由工具按路径向上查找，搬进来等于静默失效。
//! - 领域数据（算子声明、设备档案、原语片段的内容）：数据与它的 schema 必须同源，
//!   本 crate 只提供解析器 [`toml`] 与目录位置 [`Roots`]。
//!
//! [`cutile`]: https://docs.rs/cutile

#![deny(unsafe_code)]

pub mod app;
pub mod error;
pub mod key;
pub mod paths;
pub mod settings;
pub mod source;
pub mod toml;
pub mod value;

pub use app::{App, Cutile};
pub use error::Error;
pub use key::Key;
pub use paths::Roots;
pub use settings::Settings;
pub use source::{Layer, Origin, Sources};
pub use value::{Kind, Value};

/// 配置文件里 `schema` 字段的期望值；键集合或语义变化时递增。
pub const SCHEMA_VER: u16 = 1;

/// "cutile" 的配置文件名。
///
/// 其它应用用 [`App::FILE`]（或读 [`Cutile::FILE`]）；这个常量只为既有调用点保留，
/// 新增代码不应再直接引用它，否则身份又会被写死回一处。
pub const FILE_NAME: &str = Cutile::FILE;
