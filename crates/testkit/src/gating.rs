//! 真机门控与跳过登记：测试专用的环境前提开关，不是产品配置。
//!
//! 这里的真机门控变量**不进 `cli::settings` 的键表**，生产代码不得读取它。真机用例的形态固定为
//! "默认跳过、显式开门才跑"：前提不满足的用例必须走 [`skip`] 登记，禁止裸 `return` 空转——
//! 跳过原因进进程内计数与结构化日志，设置 `LLM_TEST_SKIP_DIR` 时还会追加到该目录的
//! `skips.log`，供 CI 汇总。

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
///
/// 每条跳过写一行结构化记录 `llm-testkit[skip]: <原因>`，并计入进程内计数；若设置了
/// `LLM_TEST_SKIP_DIR`，同时追加到该目录的 `skips.log`，跨测试二进制汇总时按文件读取。
/// 原因要写清缺什么（设备 / 编译器 / 样本目录 / 旧基线二进制），不留"不知道为什么没跑"。
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
    SKIP_REASONS
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default()
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
mod tests {
    use super::*;

    #[test]
    fn gating_accepts_all_lists_and_ignores_blank_entries() {
        assert!(matches("all", "nvidia"));
        assert!(matches("cpu, nvidia", "nvidia"));
        assert!(matches(" cpu ", "cpu"));
        assert!(!matches("", "cpu"));
        assert!(!matches("cpu", "nvidia"));
    }

    #[test]
    fn skip_registers_reason_and_count() {
        let before = skip_count();
        assert!(skip("测试用临时跳过原因"));
        assert!(skip_count() > before, "跳过计数必须单调递增");
        assert!(
            skipped_reasons()
                .iter()
                .any(|reason| reason == "测试用临时跳过原因")
        );
    }

    #[test]
    fn device_or_skip_registers_when_disabled() {
        let before = skip_count();
        assert!(device_or_skip("nvidia", "测试：未启用 nvidia"));
        assert!(skip_count() > before, "跳过计数必须单调递增");
    }
}
