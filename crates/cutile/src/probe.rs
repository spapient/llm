//! 设备探测：只如实探测，不做判断（DESIGN §5.3、§13 的 `probe`）。
//!
//! P0 只有 CPU 是真实可用的；GPU 档案保留下来，但探测不到设备时协商会拒绝它们。

use crate::prof::{Detected, MatchRule, Profile};
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

/// 构造一个假指纹：用于模拟器、回归测试与 `--device` 显式指定。
///
/// 指纹**从档案自己的 `[fingerprint]` 反查**，不另写一张厂商表：规则语义只在 `prof` 里定义
/// 一次，这里只负责给每条规则填一个能满足它的值（`Exact` 取原文、`Range` 取下界、
/// `Contains` 取第一个子串、`Set` 取前缀加全部成员、`OneOf` 取第一个分支）。新增一台设备因此
/// 只需加一份 TOML——手写表的时代，漏掉这里不会让任何编译或测试失败，只会在运行期得到
/// “没有档案匹配该指纹”。
///
/// 档案里没有该 vendor（拼错、或这台机器没带它）时退回 `arch = <vendor>`，
/// [`select`] 照常报“没有档案匹配”，不静默给一个能命中的指纹。
pub fn fake(vendor: &str) -> Detected {
    let Some(profile) = ::prof::builtin()
        .iter()
        .filter_map(|archive| Profile::parse(archive.text()).ok())
        .find(|profile| profile.meta.vendor == vendor)
    else {
        return Detected::new().with("arch", vendor);
    };
    profile
        .fingerprint
        .iter()
        .fold(Detected::new(), |detected, (key, rule)| {
            detected.with(key, example_of(rule))
        })
}

/// 一条匹配规则的一个满足值。
fn example_of(rule: &MatchRule) -> String {
    match rule {
        MatchRule::Exact(text) => text.clone(),
        MatchRule::Range { lo, .. } => lo.clone(),
        MatchRule::Contains(tokens) => tokens.first().cloned().unwrap_or_default(),
        MatchRule::Set { base, members } => format!("{base}{}", members.join("")),
        MatchRule::OneOf(branches) => branches.first().map_or_else(String::new, example_of),
    }
}

/// 按指纹挑档案：命中的规则条数多者胜出；同等具体多个视为配置错误。
pub fn select<'a>(profiles: &'a [Profile], detected: &Detected) -> diagnostic::Result<&'a Profile> {
    let mut best: Option<usize> = None;
    for (index, profile) in profiles.iter().enumerate() {
        if !profile.matches(detected) {
            continue;
        }
        match best {
            None => best = Some(index),
            Some(current) => {
                let current_specificity = profiles[current].specificity();
                if profile.specificity() > current_specificity {
                    best = Some(index);
                } else if profile.specificity() == current_specificity {
                    return Err(Error::new(
                        Code::Invalid,
                        format!(
                            "`{}` 与 `{}` 的指纹同等具体，无法决定用哪一份（指纹：{detected}）",
                            profiles[current].meta.vendor, profile.meta.vendor
                        ),
                    ));
                }
            }
        }
    }
    best.map(|index| &profiles[index])
        .ok_or_else(|| Error::new(Code::Unsupported, format!("没有档案匹配该指纹：{detected}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::Catalogue;

    #[test]
    fn cpu_detection_matches_the_cpu_profile() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let detected = detect_cpu();
        assert_eq!(detected.get("arch"), Some(std::env::consts::ARCH));
        let profile = select(catalogue.profiles(), &detected).expect("必须命中");
        assert_eq!(profile.meta.vendor, "cpu");
    }

    #[test]
    fn fake_fingerprints_come_from_the_profiles_themselves() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        // 逐份档案反查：加一台设备不需要改这里的任何一行。
        for profile in catalogue.profiles() {
            let detected = fake(&profile.meta.vendor);
            let chosen = select(catalogue.profiles(), &detected).expect("必须命中");
            assert_eq!(
                chosen.meta.vendor, profile.meta.vendor,
                "`{}` 的假指纹落到了别的档案：{detected}",
                profile.meta.vendor
            );
        }
        // 值取自 `[fingerprint]` 的规则，而不是手写表：改档案里的上下界，假指纹跟着变。
        let metal = fake("metal");
        assert_eq!(metal.get("name"), Some("Apple M"), "contains 取第一个子串");
        assert_eq!(metal.get("gpu_cores"), Some("10"), "range 取下界");
    }

    #[test]
    fn cpu_features_are_probed_at_runtime_not_at_build_time() {
        let detected = detect_cpu();
        let features = detected.get("features").expect("features 必须有值");
        assert_eq!(features.split('+').next(), Some(std::env::consts::ARCH));
        // 与 cpuid 对拍：把 `cfg!(target_feature)` 换回来会让本机有 avx2 的 baseline 构建变红。
        #[cfg(target_arch = "x86_64")]
        assert_eq!(
            features.split('+').any(|feature| feature == "avx2"),
            std::arch::is_x86_feature_detected!("avx2"),
            "指纹必须描述这台机器的 cpuid：{features}"
        );
    }

    #[test]
    fn unknown_fingerprint_is_an_error_not_a_silent_fallback() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let error = select(catalogue.profiles(), &fake("sparc")).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("没有档案匹配"), "{error}");
    }

    #[test]
    fn equally_specific_profiles_are_ambiguous() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let mut profiles = catalogue.profiles().to_vec();
        let cpu = profiles
            .iter()
            .find(|profile| profile.meta.vendor == "cpu")
            .expect("内置 cpu 档案")
            .clone();
        profiles.push(cpu);
        let error = select(&profiles, &detect_cpu()).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("同等具体"), "{error}");
    }
}
