//! 设备探测：只如实探测，不做判断（README §5.3、§13 的 `probe`）。
//!
//! P0 只有 CPU 是真实可用的；GPU 档案保留下来，但探测不到设备时协商会拒绝它们。

use crate::prof::{Detected, Profile};
use diagnostic::{Code, Error};

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
        .with("arch", arch)
        .with("features", features.join("+"))
}

/// 探测到的全部设备（P0 只有 CPU）。
pub fn detect() -> Vec<Detected> {
    vec![detect_cpu()]
}

/// 按指纹挑档案：命中的规则条数多者胜出；与**最优者**同等具体才算配置错误。
///
/// 只在最优档比并列：两份更宽的档案撞在一起不影响“选哪一份”，那是 `Catalogue::validate`
/// 在装载期回答的问题；这里只回答这次该用谁。
pub fn select<'a>(profiles: &'a [Profile], detected: &Detected) -> diagnostic::Result<&'a Profile> {
    let best = profiles
        .iter()
        .enumerate()
        .filter(|(_, profile)| profile.matches(detected))
        .max_by_key(|(_, profile)| profile.specificity());
    let Some((index, profile)) = best else {
        return Err(Error::new(
            Code::Unsupported,
            format!("没有档案匹配该指纹：{detected}"),
        ));
    };
    match rival(profiles, detected, index, profile.specificity()) {
        Some(other) => Err(Error::new(
            Code::Invalid,
            format!(
                "`{}` 与 `{}` 的指纹同等具体，无法决定用哪一份（指纹：{detected}）",
                profile.meta.vendor, other.meta.vendor
            ),
        )),
        None => Ok(profile),
    }
}

/// 与最优者同等具体的另一份档案：它让“选谁”没有唯一答案。
fn rival<'a>(
    profiles: &'a [Profile],
    detected: &Detected,
    best: usize,
    top: usize,
) -> Option<&'a Profile> {
    profiles
        .iter()
        .enumerate()
        .find(|(index, profile)| {
            *index != best && profile.specificity() == top && profile.matches(detected)
        })
        .map(|(_, profile)| profile)
}

#[cfg(test)]
#[path = "../tests/probe_tests.rs"]
mod tests;
