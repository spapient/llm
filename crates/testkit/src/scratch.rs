//! 测试临时目录：按标签取一个进程隔离的空目录。

use std::path::PathBuf;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_tag_reuses_one_directory() {
        let first = scratch_dir("a");
        let second = scratch_dir("a");
        assert!(first.is_dir());
        assert_eq!(first, second, "同标签同进程应复用同一目录");
        assert_ne!(first, scratch_dir("b"), "不同标签必须各自一个目录");
        let _ = std::fs::remove_dir_all(&first);
        let _ = std::fs::remove_dir_all(scratch_dir("b"));
    }

    #[test]
    fn scratch_clean_wipes_prior_content() {
        let dir = scratch_clean("clean");
        std::fs::write(dir.join("leftover.txt"), "x").expect("写入残留");
        assert!(dir.join("leftover.txt").exists());
        let again = scratch_clean("clean");
        assert!(again.is_dir());
        assert!(
            !again.join("leftover.txt").exists(),
            "第二次取同标签必须先清空残留"
        );
        let _ = std::fs::remove_dir_all(&again);
    }
}
