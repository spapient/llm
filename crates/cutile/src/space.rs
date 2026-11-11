//! 三级存储：语义名映射、容量校验、对齐校验（DESIGN §6.3）。

use crate::atom::Space;
use crate::prof::Profile;

/// 一个存储层的用量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceUse {
    /// 语义存储层。
    pub space: Space,
    /// 字节数。
    pub bytes: u64,
}

impl SpaceUse {
    /// 构造。
    pub const fn new(space: Space, bytes: u64) -> Self {
        Self { space, bytes }
    }
}

/// 语义名映射到本设备的物理名；档案缺项时退回语义名本身。
pub fn physical(profile: &Profile, space: Space) -> &str {
    profile.physical_space(space).unwrap_or(space.as_str())
}

/// 容量校验：`FAST` 与 `REG` 的用量不得超过档案声明的字节数。
pub fn check_capacity(profile: &Profile, uses: &[SpaceUse]) -> Result<(), String> {
    for usage in uses {
        let limit = match usage.space {
            Space::Fast => profile.mem.fast_bytes,
            Space::Reg => profile.mem.reg_bytes,
            Space::Gm => continue,
        };
        if limit > 0 && usage.bytes > limit {
            return Err(format!(
                "{} 需要 {} B，档案只声明 {} B（物理层 `{}`）",
                usage.space,
                usage.bytes,
                limit,
                physical(profile, usage.space)
            ));
        }
    }
    Ok(())
}

/// 对齐校验：swizzle 的 bank 粒度不得超过档案声明的 bank 数。
pub fn check_alignment(profile: &Profile, swizzle_bits: u8) -> Result<(), String> {
    let bank = profile.mem.fast_bank;
    if bank == 0 {
        return Ok(());
    }
    if (1u32 << swizzle_bits) > bank {
        return Err(format!(
            "swz<{swizzle_bits},…> 需要 {} 个 bank，档案只有 {bank} 个",
            1u32 << swizzle_bits
        ));
    }
    Ok(())
}

/// 一条存储用量的人读描述。
pub fn describe(profile: &Profile, usage: SpaceUse) -> String {
    format!(
        "{}={} ({} B)",
        usage.space,
        physical(profile, usage.space),
        usage.bytes
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> Profile {
        Profile::parse(include_str!("../devices/cpu.toml")).expect("CPU 档案可解析")
    }

    #[test]
    fn maps_semantic_names_to_physical_ones() {
        let profile = profile();
        assert_eq!(physical(&profile, Space::Gm), "main");
        assert_eq!(physical(&profile, Space::Fast), "L2");
        assert_eq!(physical(&profile, Space::Reg), "stack");
    }

    #[test]
    fn capacity_is_enforced_against_profile() {
        let profile = profile();
        assert!(check_capacity(&profile, &[SpaceUse::new(Space::Fast, 1024)]).is_ok());
        let error = check_capacity(&profile, &[SpaceUse::new(Space::Fast, 1 << 30)]).unwrap_err();
        assert!(error.contains("L2"), "{error}");
    }

    #[test]
    fn global_memory_is_unbounded_in_the_profile() {
        let profile = profile();
        assert!(check_capacity(&profile, &[SpaceUse::new(Space::Gm, u64::MAX / 2)]).is_ok());
    }

    #[test]
    fn alignment_uses_declared_bank_count() {
        let profile = profile();
        assert!(check_alignment(&profile, 4).is_ok());
        assert!(check_alignment(&profile, 6).is_err());
    }

    #[test]
    fn describe_uses_physical_names() {
        let profile = profile();
        assert_eq!(
            describe(&profile, SpaceUse::new(Space::Fast, 256)),
            "FAST=L2 (256 B)"
        );
    }
}
