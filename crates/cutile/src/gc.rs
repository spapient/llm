//! 缓存治理的命令策略：`cache gc` 的上限与回收。
//!
//! 删哪些条目全由缓存自己决定（时间戳与锁文件在 [`crate::cache`] 里），这里只决定"上限是多少"
//! ——它是命令策略，不进内核，也不塞进 `cache.rs`（那个模块只管机制）。

use crate::cache::Cache;
use crate::dirs::Dirs;

/// 缓存容量上限的缺省值（README §10.3）。
pub const CACHE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// `cache gc`：把缓存按 LRU 回收到上限以内；`limit` 为 `None` 时用 [`CACHE_LIMIT`]。
///
/// # Errors
///
/// 删除某个条目失败时返回带路径的说明（缓存目录不可写之类）。
pub fn cache_gc(dirs: &Dirs, limit: Option<u64>) -> diagnostic::Result<crate::cache::Gc> {
    Cache::from_dirs(dirs).gc(limit.unwrap_or(CACHE_LIMIT))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立缓存目录：GC 会真删东西，不能与其他用例共用。
    fn cache_dirs(name: &str) -> Dirs {
        let root = std::env::temp_dir().join(format!("cutile-gc-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Dirs::under(root)
    }

    #[test]
    fn cache_gc_reclaims_through_the_cache_directory() {
        let dirs = cache_dirs("cache-gc");
        let catalogue = crate::builtin::Catalogue::try_builtin().expect("内置数据");
        let report = crate::doctor::doctor(&catalogue, &dirs, "gemv_i4", "cpu").expect("doctor");
        assert!(report.source.is_some(), "doctor 应已落盘");
        let before = cache_gc(&dirs, None).expect("可回收");
        assert_eq!(before.limit, CACHE_LIMIT, "缺省上限是命令策略");
        assert!(before.before > 0, "doctor 刚落过盘：{before:?}");
        assert_eq!(before.removed, 0, "未超上限时不删：{before:?}");

        // 条目个数不再是"doctor 只落 1 条"：它现在还要把见证程序登记进缓存。
        let entries = Cache::from_dirs(&dirs).entries().len();
        assert!(entries > 1, "内核源之外还有见证程序：{entries}");
        let after = cache_gc(&dirs, Some(0)).expect("可回收");
        assert_eq!(after.removed, entries, "{after:?}");
        assert_eq!(after.after, 0, "{after:?}");
        let _ = std::fs::remove_dir_all(dirs.cache());
    }
}
