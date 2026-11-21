//! 三个目录：编译产物缓存、临时工作目录、可选的数据根。
//!
//! 这里是**纯数据**，不读环境变量、不拼平台默认路径、不认识命令行：机制知道"我要一个缓存目录"，
//! 但**不该决定它在哪**。默认值、`CUTILE_*` 环境变量与 `--set key=value` 的优先级都住在命令行
//! 前端 `cli::dirs`（ADR 0023），于是任何嵌入方都能接管，本 crate 只依赖 `std::path`。
//!
//! # 例子
//!
//! ```
//! use cutile::dirs::Dirs;
//!
//! let dirs = Dirs::under("/tmp/demo");
//! assert_eq!(dirs.cache(), std::path::Path::new("/tmp/demo/cache"));
//! assert_eq!(dirs.data(), None);
//! ```

use prim::Layout;
use std::path::{Path, PathBuf};

/// 一次调用要用的三个目录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dirs {
    cache: PathBuf,
    work: PathBuf,
    data: Option<PathBuf>,
}

impl Dirs {
    /// 显式给出三个目录：缓存根、工作目录、可选的数据根。
    #[must_use]
    pub fn new(cache: PathBuf, work: PathBuf, data: Option<PathBuf>) -> Self {
        Self { cache, work, data }
    }

    /// `<root>/{cache,work}`、没有数据根：测试与嵌入方的便利布局。
    #[must_use]
    pub fn under(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self::new(root.join("cache"), root.join("work"), None)
    }

    /// 编译产物缓存根目录。
    #[must_use]
    pub fn cache(&self) -> &Path {
        &self.cache
    }

    /// 编译与执行时的临时工作目录。
    #[must_use]
    pub fn work(&self) -> &Path {
        &self.work
    }

    /// 声明层 / 档案层 / 片段层的替代根；`None` 表示用编译期内嵌数据。
    #[must_use]
    pub fn data(&self) -> Option<&Path> {
        self.data.as_deref()
    }

    /// 片段层：有数据根时读 `<data.root>/prim/`，否则读六台设备各自的内置 crate。
    #[must_use]
    pub fn prim(&self) -> Layout<'_> {
        self.data().map_or_else(Layout::builtin, Layout::at)
    }
}
