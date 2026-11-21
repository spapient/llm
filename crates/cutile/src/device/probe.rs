//! 设备探测：只如实探测，不做判断（README §5.3、§13 的 `probe`）。
//!
//! P0 只有 CPU 是真实可用的；GPU 档案保留下来，但探测不到设备时协商会拒绝它们。

use crate::prof::{Detected, Profile};

/// 探测本机 CPU。
///
/// 指令集用 `is_x86_feature_detected!` / `is_aarch64_feature_detected!` 在**运行期**问 CPU，
/// 不用 `cfg!(target_feature)`：后者回答的是“这个二进制按什么指令集编的”。默认 baseline 构建
/// 下它永远报不出 `avx2`，而被 `-C target-cpu=native` 编出来的二进制拷到别的机器又会虚报能力
/// ——指纹要描述这台机器，不是这个产物。
pub fn detect_cpu() -> Detected {
    let arch = std::env::consts::ARCH;
    let mut features = vec![arch.to_string()];
    #[cfg(target_arch = "x86_64")]
    {
        for (name, present) in [
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("avx512", std::arch::is_x86_feature_detected!("avx512f")),
        ] {
            if present {
                features.push(name.to_string());
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon".to_string());
        }
    }
    Detected::new()
        .with("algorithm", arch)
        .with("features", features.join("+"))
}

/// 探测到的全部设备（P0 只有 CPU）。
pub fn detect() -> Vec<Detected> {
    vec![detect_cpu()]
}

/// 按指纹挑档案：匹配、具体度排序与平局判定都归 `prof::select`，本模块只做本机探测。
pub fn select<'a>(profiles: &'a [Profile], detected: &Detected) -> diagnostic::Result<&'a Profile> {
    crate::prof::select(profiles, detected)
}

#[cfg(test)]
#[path = "../../tests/device_probe_tests.rs"]
mod tests;
