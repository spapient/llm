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
#[path = "../tests/device_probe_tests.rs"]
mod tests;
