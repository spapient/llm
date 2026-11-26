//! 真机门控与跳过登记：测试专用的环境前提开关，不是产品配置。

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// 真机门控环境变量名（测试专用，不是产品配置）。取值是逗号分隔的 vendor 列表。
const DEVICE_ENV: &str = "LLM_TEST_DEVICE";

/// 进程内跳过计数与原因清单：标准测试框架没有正式 skip 信号，集中登记后才能审计。
static SKIPPED: AtomicUsize = AtomicUsize::new(0);
static SKIP_REASONS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// 这台设备的真机用例是否被显式打开。
pub fn device_enabled(vendor: &str) -> bool {
    std::env::var(DEVICE_ENV).is_ok_and(|spec| matches(&spec, vendor))
}

/// 原因锁只保护 Vec 操作，持锁期间不会 panic；若外部已中毒，恢复数据比清空审计轨迹更安全。
fn reasons() -> MutexGuard<'static, Vec<String>> {
    SKIP_REASONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// 登记一条"环境前提不满足"的跳过；恒返回 `true`，调用方据此立即 `return`。
pub fn skip(reason: impl Into<String>) -> bool {
    let reason = reason.into();
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    eprintln!("llm-testkit[skip]: {reason}");
    append_skip_log(&reason);
    reasons().push(reason);
    true
}

/// 按配置把跳过原因追加到日志；日志是辅助产物，失败不掩盖原始跳过事实。
fn append_skip_log(reason: &str) {
    let Some(dir) = std::env::var_os("LLM_TEST_SKIP_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("skips.log"))
    else {
        return;
    };
    use std::io::Write;
    let _ = writeln!(file, "{reason}");
}

/// 登记一条跳过并返回 `None`：供 `Option` 链式表达式内联使用。
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
    reasons().clone()
}

/// 真机门控与跳过登记合一：设备未显式打开时登记原因并返回 `true`。
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
mod tests {
    use super::*;

    crate::test_cases!(registration {
        gating_accepts_all_lists_and_ignores_blank_entries => || {
            assert!(matches("all", "nvidia"));
            assert!(matches("cpu, nvidia", "nvidia"));
            assert!(matches(" cpu ", "cpu"));
            assert!(!matches("", "cpu"));
            assert!(!matches("cpu", "nvidia"));
        },
        skip_registers_reason_and_count => || {
            let before = skip_count();
            assert!(skip("测试用临时跳过原因"));
            assert!(skip_count() > before);
            assert!(skipped_reasons().iter().any(|reason| reason == "测试用临时跳过原因"));
        },
        device_or_skip_registers_when_disabled => || {
            let before = skip_count();
            assert!(device_or_skip("nvidia", "测试：未启用 nvidia"));
            assert!(skip_count() > before);
        },
    }, |case: fn()| case());
}
