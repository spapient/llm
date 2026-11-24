//! 真机门控与跳过登记：测试专用的环境前提开关，不是产品配置。

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 真机门控环境变量名（测试专用，不是产品配置）。取值是逗号分隔的 vendor 列表
/// （`cpu,nvidia`），`all` 表示全部打开；未设置表示全部关闭。
const DEVICE_ENV: &str = "LLM_TEST_DEVICE";

/// 进程内跳过计数与原因清单：标准测试框架没有正式 skip 信号，集中登记后才能在报告里
/// 区分"通过"与"前提不满足而未运行"。
static SKIPPED: AtomicUsize = AtomicUsize::new(0);
static SKIP_REASONS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// 这台设备的真机用例是否被显式打开。
pub fn device_enabled(vendor: &str) -> bool {
    match std::env::var(DEVICE_ENV) {
        Ok(spec) => matches(&spec, vendor),
        Err(_) => false,
    }
}

/// 登记一条"环境前提不满足"的跳过；恒返回 `true`，调用方据此立即 `return`。
pub fn skip(reason: impl Into<String>) -> bool {
    let reason = reason.into();
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut reasons) = SKIP_REASONS.lock() {
        reasons.push(reason.clone());
    }
    eprintln!("llm-testkit[skip]: {reason}");
    if let Some(dir) = std::env::var_os("LLM_TEST_SKIP_DIR") {
        let dir = PathBuf::from(dir);
        if std::fs::create_dir_all(&dir).is_ok() {
            let path = dir.join("skips.log");
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                use std::io::Write;
                let _ = writeln!(file, "{reason}");
            }
        }
    }
    true
}

/// 登记一条跳过并返回 `None`：供 `Option` 链式表达式（如 `ok().or_else(...)`）内联使用，
/// 避免在测试臂里再开一个语句块；类型 `T` 由上下文推断。
#[must_use]
pub fn skip_none<T>(reason: impl Into<String>) -> Option<T> {
    skip(reason);
    None
}

/// 本进程已登记的跳过条数。
#[must_use]
pub fn skip_count() -> usize {
    SKIPPED.load(Ordering::Relaxed)
}

/// 本进程已登记的跳过原因（按登记顺序）。
#[must_use]
pub fn skipped_reasons() -> Vec<String> {
    SKIP_REASONS.lock().map_or_default(|guard| guard.clone())
}

/// 真机门控与跳过登记合一：设备未显式打开时登记原因并返回 `true`（调用方立即 `return`）；
/// 已打开返回 `false`。等价于 `if !device_enabled(v) { skip(reason) }`，省掉重复的成对写法。
pub fn device_or_skip(vendor: &str, reason: impl Into<String>) -> bool {
    if device_enabled(vendor) {
        false
    } else {
        skip(reason);
        true
    }
}

/// 逗号列表匹配：`all` 全开，空白被忽略，空列表等于没开。
fn matches(spec: &str, vendor: &str) -> bool {
    spec.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .any(|item| item == "all" || item == vendor)
}

#[cfg(test)]
#[path = "../tests/gating_tests.rs"]
mod gating_tests;
