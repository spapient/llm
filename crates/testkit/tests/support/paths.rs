//! 集成测试共享的仓根定位与源码遍历助手。
//!
//! 多个守卫测试都要从 `crates/<crate>/tests/` 定位仓根、递归收集 `.rs`；各写一遍会让目录
//! 布局调整时漏改其中一处。这里只放与被测领域无关的文件系统遍历。

use std::path::{Path, PathBuf};

/// 仓根：本测试住在 `crates/<crate>/tests/`，仓根是清单目录上两级。
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 递归列出 `dir` 下的全部 `.rs` 文件（顺序无关，只用于扫描）。
///
/// 目录不存在或读不到时返回空：扫描集派生方（如层表）必须另守"至少扫到一个文件"，
/// 否则合并/删除成员会让守卫静默通过。
pub fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    walk(dir, &mut found);
    found
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
