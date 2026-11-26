//! 职责：设备探测：只如实探测，不做判断（README §5.3、§13 的 `probe`）。

use crate::prof::{Detected, Emit, Profile};
use diagnostic::{Code, Error};
use std::process::Command;

/// 探测本机 CPU。
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

/// 解析 `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits`。
/// 用最后一个逗号拆字段，合法设备名里即使含逗号也不会截断；空行和残缺行直接跳过。
fn parse_nvidia_smi(text: &str) -> Vec<Detected> {
    text.lines()
        .filter_map(|line| {
            let (name, cc) = line.trim().rsplit_once(',')?;
            let name = name.trim().trim_matches('"');
            let cc = cc.trim().trim_matches('"');
            if name.is_empty() || cc.is_empty() {
                return None;
            }
            Some(
                Detected::new()
                    .with("name", name)
                    .with("cc", cc)
                    .with("kind", "gpu")
                    .with("api", "cuda"),
            )
        })
        .collect()
}

/// 逐块探测 NVIDIA GPU；没有工具、驱动或设备时返回空表，不把调用失败伪装成设备型号。
pub fn detect_nvidia() -> Vec<Detected> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout))
}

/// 探测到的全部设备：CPU 始终在场，其后是实际枚举到的 NVIDIA GPU。
pub fn detect() -> Vec<Detected> {
    let mut found = vec![detect_cpu()];
    found.extend(detect_nvidia());
    found
}

/// 逐台设备取唯一档案；同型号多卡重复命中同一档案时只保留一次。
fn matching_profiles<'a>(
    profiles: &'a [Profile],
    devices: &[Detected],
) -> diagnostic::Result<Vec<&'a Profile>> {
    let mut matched: Vec<&Profile> = Vec::new();
    for device in devices {
        push_match(profiles, device, &mut matched)?;
    }
    Ok(matched)
}

/// 单台设备命中一个档案：不支持的型号静默跳过，其余错误上抛。
fn push_match<'a>(
    profiles: &'a [Profile],
    device: &Detected,
    matched: &mut Vec<&'a Profile>,
) -> diagnostic::Result<()> {
    match select(profiles, device) {
        Ok(profile) => {
            if !matched
                .iter()
                .any(|current| std::ptr::eq(*current, profile))
            {
                matched.push(profile);
            }
        }
        Err(error) if error.code() == Code::Unsupported => {}
        Err(error) => return Err(error),
    }
    Ok(())
}

/// 在多台设备的指纹间选择：加速器优先，再按具体度排序；同分多档案必须显式指定设备。
pub fn select_detected<'a>(
    profiles: &'a [Profile],
    devices: &[Detected],
) -> diagnostic::Result<&'a Profile> {
    let mut matched = matching_profiles(profiles, devices)?;
    prefer_non_c(&mut matched);
    matched.sort_by_key(|profile| std::cmp::Reverse(profile.specificity()));
    best_unique(&matched)
}

/// 加速器优先：只要匹配集里有非 C 档案，就只保留非 C 档案。
fn prefer_non_c(matched: &mut Vec<&Profile>) {
    if matched.iter().any(|profile| profile.meta.emit != Emit::C) {
        matched.retain(|profile| profile.meta.emit != Emit::C);
    }
}

/// 返回具体度最高且无并列的档案；空集不支持，第二名同分即判平局。
fn best_unique<'a>(matched: &[&'a Profile]) -> diagnostic::Result<&'a Profile> {
    let best = matched
        .first()
        .copied()
        .ok_or_else(|| Error::new(Code::Unsupported, "没有档案匹配任何探测到的设备"))?;
    if matched
        .get(1)
        .is_some_and(|profile| profile.specificity() == best.specificity())
    {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "多个设备以同等具体度命中 `{}`；请显式指定设备",
                best.meta.id
            ),
        ));
    }
    Ok(best)
}

/// 按指纹挑档案：匹配、具体度排序与平局判定都归 `prof::select`，本模块只做本机探测。
pub fn select<'a>(profiles: &'a [Profile], detected: &Detected) -> diagnostic::Result<&'a Profile> {
    crate::prof::select(profiles, detected)
}

#[cfg(test)]
mod tests {
    //! `cutile::probe` 的用例：探测与降级。
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

    testkit::test_cases! {
    fn cpu_detection_matches_the_cpu_profile() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let detected = detect_cpu();
        assert_eq!(detected.get("arch"), Some(std::env::consts::ARCH));
        let profile = select(catalogue.profiles(), &detected).expect("必须命中");
        assert_eq!(profile.meta.vendor, "cpu");
    }

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

    fn unknown_fingerprint_is_an_error_not_a_silent_fallback() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let error = select(catalogue.profiles(), &fake("sparc")).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("没有档案匹配"), "{error}");
    }

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

    fn a3000() -> Detected {
        Detected::new()
            .with("cc", "8.6")
            .with("name", "NVIDIA RTX A3000 12GB Laptop GPU")
    }

    testkit::test_cases! {
    fn nvidia_smi_parser_preserves_names_and_rejects_incomplete_rows() {
        let parsed = parse_nvidia_smi(
            "NVIDIA RTX A3000 12GB Laptop GPU, 8.6\n\"RTX, Test\",9.0\nmissing comma\n,\n",
        );
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].get("cc"), Some("8.6"));
        assert_eq!(parsed[0].get("kind"), Some("gpu"));
        assert_eq!(parsed[0].get("api"), Some("cuda"));
        assert_eq!(parsed[1].get("name"), Some("RTX, Test"));
        assert_eq!(parsed[1].get("cc"), Some("9.0"));
    }

    fn automatic_selection_falls_back_to_cpu_without_an_accelerator() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let chosen = select_detected(catalogue.profiles(), &[detect_cpu(), fake("sparc")])
            .expect("无加速器时必须回退 CPU");
        assert_eq!(chosen.meta.vendor, "cpu");
    }

    fn automatic_selection_prefers_a3000_and_deduplicates_equal_cards() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let chosen = select_detected(catalogue.profiles(), &[detect_cpu(), a3000(), a3000()])
            .expect("CPU 与同型号双卡不能形成歧义");
        assert_eq!(chosen.meta.vendor, "nvidia_a3000");
        assert_eq!(chosen.meta.id, "sm86");
    }

    fn equally_specific_different_gpus_require_an_explicit_device() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let error = select_detected(catalogue.profiles(), &[fake("nvidia"), a3000()])
            .expect_err("T600 与 A3000 同分时不应随机选卡");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("显式指定设备"), "{error}");
    }

    fn no_matching_detected_device_is_an_explicit_error() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let error =
            select_detected(catalogue.profiles(), &[fake("sparc")]).expect_err("未知设备不能静默回退");
        assert_eq!(error.code(), Code::Unsupported);
    }
    }
}
