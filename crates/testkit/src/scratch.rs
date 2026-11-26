//! 测试临时目录：按标签取一个进程隔离的空目录。

use std::path::{Path, PathBuf};

/// 一个已创建的空临时目录，标签用于区分用例。
pub fn scratch_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("llm-testkit-{}-{tag}", std::process::id()));
    if let Err(error) = std::fs::create_dir_all(&path) {
        eprintln!("testkit: 无法创建临时目录 `{}`：{error}", path.display());
    }
    path
}

/// 先删掉旧目录再建一个空目录：进入用例时目录里不该带上一次运行的残留。
pub fn scratch_clean(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("llm-testkit-clean-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    if let Err(error) = std::fs::create_dir_all(&path) {
        eprintln!("testkit: 无法创建临时目录 `{}`：{error}", path.display());
    }
    path
}

/// RAII 临时目录：落地即建空目录，离开作用域时自动清理。
///
/// 与 [`scratch_clean`] 的差别是**生命周期归作用域管**：用例提前 panic 也会走 `Drop` 清场，
/// 不留残留目录给下一次运行。同标签多次构造会各自先清空再建，语义同 `scratch_clean`。
///
/// ```
/// let dir = llm_testkit::ScratchDir::new("demo");
/// std::fs::write(dir.path().join("out.txt"), "x").expect("写入");
/// assert!(dir.path().join("out.txt").exists());
/// // 作用域结束，目录自动删除。
/// ```
#[derive(Debug)]
pub struct ScratchDir(PathBuf);

impl ScratchDir {
    /// 按标签建一个空的 RAII 临时目录；同标签的旧残留先清空。
    #[must_use]
    pub fn new(tag: &str) -> Self {
        Self(scratch_clean(tag))
    }

    /// 目录路径。
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// 拼接目录内的子路径（不创建）。
    #[must_use]
    pub fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::test_cases!(directories {
        the_same_tag_reuses_one_directory => || {
            let first = scratch_dir("a");
            assert!(first.is_dir());
            assert_eq!(first, scratch_dir("a"));
            assert_ne!(first, scratch_dir("b"));
            let _ = std::fs::remove_dir_all(&first);
            let _ = std::fs::remove_dir_all(scratch_dir("b"));
        },
        scratch_clean_wipes_prior_content => || {
            let dir = scratch_clean("clean");
            std::fs::write(dir.join("leftover.txt"), "x").expect("写入残留");
            assert!(dir.join("leftover.txt").exists());
            let again = scratch_clean("clean");
            assert!(again.is_dir());
            assert!(!again.join("leftover.txt").exists());
            let _ = std::fs::remove_dir_all(&again);
        },
        scratch_dir_raii_builds_joins_and_cleans_up_on_drop => || {
            let path = {
                let dir = ScratchDir::new("raii");
                assert!(dir.path().is_dir());
                let child = dir.join("nested.txt");
                std::fs::write(&child, "x").expect("写入子文件");
                assert!(child.exists());
                dir.path().to_path_buf()
            };
            assert!(!path.exists());
        },
    }, |case: fn()| case());
}
