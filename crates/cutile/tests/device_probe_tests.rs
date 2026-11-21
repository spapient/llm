//! `cutile::device::probe` 的用例：探测与降级。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;
use crate::builtin::Catalogue;
use crate::prof::MatchRule;
use diagnostic::Code;

/// 假指纹：回归测试替身。指纹从档案自己的 `[fingerprint]` 反查，不另写厂商表——规则语义只在
/// `prof` 里定义一次，这里只填一个能满足规则的值；档案里没有该 vendor 时退回 `algorithm = <vendor>`，
/// [`select`] 照常报"没有档案匹配"，不静默给能命中的指纹。
fn fake(vendor: &str) -> Detected {
    let Some(profile) = ::prof::names()
        .iter()
        .filter_map(|name| ::prof::embedded(name))
        .filter_map(|table| Profile::parse(table).ok())
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
