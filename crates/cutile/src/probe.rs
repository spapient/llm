//! 设备探测：只如实探测，不做判断（DESIGN §5.3、§13 的 `probe`）。
//!
//! P0 只有 CPU 是真实可用的；GPU 档案保留下来，但探测不到设备时协商会拒绝它们。

use crate::prof::{Detected, Profile};
use diagnostic::{Code, Error};

/// 探测本机 CPU。
pub fn detect_cpu() -> Detected {
    let arch = std::env::consts::ARCH;
    let mut features = vec![arch.to_string()];
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(target_feature = "avx2") {
            features.push("avx2".to_string());
        }
        if cfg!(target_feature = "avx512f") {
            features.push("avx512".to_string());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        features.push("neon".to_string());
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
pub fn fake(vendor: &str) -> Detected {
    match vendor {
        "nvidia" => Detected::new()
            .with("cc", "10.0")
            .with("name", "NVIDIA B200"),
        "intel" => Detected::new()
            .with("name", "Intel Arc A770")
            .with("arch", "gen12"),
        "ascend" => Detected::new().with("soc", "ascend910b"),
        other => Detected::new().with("arch", other),
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
    fn fake_fingerprints_select_the_matching_profile() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let nvidia = select(catalogue.profiles(), &fake("nvidia")).expect("必须命中");
        assert_eq!(nvidia.meta.vendor, "nvidia");
        let intel = select(catalogue.profiles(), &fake("intel")).expect("必须命中");
        assert_eq!(intel.meta.vendor, "intel");
        let ascend = select(catalogue.profiles(), &fake("ascend")).expect("必须命中");
        assert_eq!(ascend.meta.vendor, "ascend");
        assert_eq!(ascend.meta.emit, crate::prof::Emit::Ascendc);
        assert_eq!(ascend.units.len(), 5);
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
