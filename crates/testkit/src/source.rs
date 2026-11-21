//! 源码事实面：仓根、`.rs` 清单与文件字节。
//!
//! 九个门禁扫的是同一批目录、读的是同一批文件，所以清单与字节各缓存一次。缓存键是**目录**与
//! **文件**本身，不是"某次扫描的结果"；命中零拷贝（`Arc`），不让每个门禁各复制一遍全文。
//!
//! 未命中时在**锁外**遍历/读盘再发布——首跑的 I/O 不会把其余门禁堵在锁上。只读：门禁不写
//! `crates/`，一次测试运行里同一路径的内容不会变，因此不存在读到过期事实的窗口。

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// 目录 → `dir` 下的全部 `.rs` 文件。
static FILES: OnceLock<Mutex<BTreeMap<PathBuf, Arc<[PathBuf]>>>> = OnceLock::new();

/// 文件 → 内容；读失败记 `None`，让"缺失"本身也可缓存（否则每次调用都要再 `open` 一次）。
static TEXTS: OnceLock<Mutex<BTreeMap<PathBuf, Option<Arc<str>>>>> = OnceLock::new();

/// 仓根：本 crate 住在 `crates/<crate>/`，仓根是清单目录上两级。
///
/// 常量折叠：不读盘、不分配，`strip_prefix` 与 `join` 都能直接用。
pub(crate) fn root() -> &'static Path {
    const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    Path::new(ROOT)
}

/// 递归列出 `dir` 下的全部 `.rs` 文件；顺序是目录遍历序，只用于扫描。
///
/// 目录不存在或读不到时返回空：扫描集派生方必须另守"至少扫到一个文件"，否则删成员会让守卫
/// 静默通过。
pub(crate) fn rust_files(dir: &Path) -> Arc<[PathBuf]> {
    cached(&FILES, dir, |dir| {
        let mut found = Vec::new();
        walk(dir, &mut found);
        found.into()
    })
}

/// 读取文件内容；读不到时 `None`（缺失也进缓存）。
pub(crate) fn text(path: &Path) -> Option<Arc<str>> {
    cached(&TEXTS, path, |path| {
        std::fs::read_to_string(path).ok().map(Arc::from)
    })
}

/// 命中即返回，未命中在锁外派生后发布。
///
/// `Q: Ord` 的借用查询让命中路径零分配：`&Path` 直接查 `BTreeMap<PathBuf, _>`。
fn cached<K, V, Q>(slot: &'static OnceLock<Mutex<BTreeMap<K, V>>>, key: &Q, derive: impl FnOnce(&Q) -> V) -> V
where
    K: Borrow<Q> + Ord,
    Q: Ord + ToOwned<Owned = K> + ?Sized,
    V: Clone,
{
    if let Some(hit) = locked(slot).get(key) {
        return hit.clone();
    }
    let fresh = derive(key);
    locked(slot).insert(key.to_owned(), fresh.clone());
    fresh
}

/// 取锁；中毒按"恢复"处理——门禁断言不在持锁期间触发，中毒只可能来自别的线程失败，
/// 不该把一个已经失败的用例扩散成其余门禁的缓存错误。
fn locked<K, V>(slot: &'static OnceLock<Mutex<BTreeMap<K, V>>>) -> MutexGuard<'static, BTreeMap<K, V>> {
    slot.get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// 深度优先收集 `.rs`；不跟符号链接（`crates/` 下没有链接，跟了反而可能成环）。
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

use std::borrow::Borrow;
