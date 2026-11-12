//! 测试约定：浮点对拍、真机门控、临时目录。**只作 dev-dependency**。
//!
//! **拥有**：`approx_eq`、`device_enabled`、`scratch_dir`。**不拥有**：任何生产逻辑——它是
//! 测试夹具，不进任何二进制。**依赖**：无（夹具不该把被测 crate 拉进来）。**被依赖**：各成员
//! 的 `[dev-dependencies]`。
//!
//! 这里的 [`DEVICE_ENV`] 是**测试门控变量，不是产品配置**：它不进 `llm-config` 的键闭集，
//! 生产代码不得读取它。真机用例的形态固定为"默认跳过、显式开门才跑"：
//!
//! ```
//! if llm_testkit::device_enabled("nvidia") {
//!     // 需要 nvcc + 真卡的用例
//! }
//! ```

#![deny(unsafe_code)]

use std::path::PathBuf;

/// 真机门控环境变量（测试专用，不是产品配置）。
///
/// 取值是逗号分隔的 vendor 列表（`cpu,nvidia`），`all` 表示全部打开；未设置表示全部关闭。
pub const DEVICE_ENV: &str = "LLM_TEST_DEVICE";

/// 逐个元素比较，`NaN` 与 `NaN` 视为相等（对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
///
/// 长度不同直接返回 `false`：对拍的首要条件是形状一致。
pub fn approx_eq(left: &[f32], right: &[f32], tolerance: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right).all(|(a, b)| {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() <= tolerance
    })
}

/// 这台设备的真机用例是否被显式打开。
pub fn device_enabled(vendor: &str) -> bool {
    match std::env::var(DEVICE_ENV) {
        Ok(spec) => matches(&spec, vendor),
        Err(_) => false,
    }
}

/// 逗号列表匹配：`all` 全开，空白被忽略，空列表等于没开。
fn matches(spec: &str, vendor: &str) -> bool {
    spec.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .any(|item| item == "all" || item == vendor)
}

/// 一个已创建的空临时目录，标签用于区分用例。
///
/// 目录名带进程号，避免并行测试互相踩。
pub fn scratch_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("llm-testkit-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&path).expect("创建测试临时目录");
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_is_elementwise_and_length_sensitive() {
        assert!(approx_eq(&[1.0, 2.0], &[1.0, 2.0], 0.0));
        assert!(approx_eq(&[1.0, 2.0], &[1.0, 2.0004], 1e-3));
        assert!(!approx_eq(&[1.0, 2.0], &[1.0, 2.01], 1e-3));
        assert!(!approx_eq(&[1.0], &[1.0, 1.0], 1.0), "长度不同必须为假");
    }

    #[test]
    fn nan_matches_nan() {
        assert!(approx_eq(&[f32::NAN], &[f32::NAN], 0.0));
    }

    #[test]
    fn gating_accepts_all_lists_and_ignores_blank_entries() {
        assert!(matches("all", "nvidia"));
        assert!(matches("cpu, nvidia", "nvidia"));
        assert!(matches(" cpu ", "cpu"));
        assert!(!matches("", "cpu"));
        assert!(!matches("cpu", "nvidia"));
    }

    #[test]
    fn scratch_directories_are_created_and_distinct() {
        let first = scratch_dir("a");
        let second = scratch_dir("a");
        assert!(first.is_dir());
        assert_eq!(first, second, "同标签同进程应复用同一目录");
        let other = scratch_dir("b");
        assert_ne!(first, other);
        let _ = std::fs::remove_dir_all(&first);
        let _ = std::fs::remove_dir_all(&other);
    }
}
