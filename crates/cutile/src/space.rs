//! 三级存储：语义名映射、容量校验、对齐校验（README §6.3）。

use crate::atom::Space;
use crate::prof::Profile;
use diagnostic::{Code, Error};

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

/// 档案声明的容量；`None` 表示这一层不受档案限制（`GM`，或档案没有声明字节数）。
///
/// 容量只有这一个出处：`check_capacity` 用它判定，报告用它打印，两者不会各算一份。
pub fn capacity(profile: &Profile, space: Space) -> Option<u64> {
    let bytes = match space {
        Space::Fast => profile.mem.fast_bytes,
        Space::Reg => profile.mem.reg_bytes,
        Space::Gm => return None,
    };
    (bytes > 0).then_some(bytes)
}

/// 容量校验：`FAST` 与 `REG` 的用量不得超过档案声明的字节数。
///
/// 超出容量不是“实现写得不好”，而是“这份声明与这台设备的组合不存在”——类别是
/// [`Code::Unsupported`]，与协商拒绝同一个意思。
pub fn check_capacity(profile: &Profile, uses: &[SpaceUse]) -> diagnostic::Result<()> {
    for usage in uses {
        let Some(limit) = capacity(profile, usage.space) else {
            continue;
        };
        if usage.bytes > limit {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "{} 需要 {} B，档案只声明 {} B（物理层 `{}`）",
                    usage.space,
                    usage.bytes,
                    limit,
                    physical(profile, usage.space)
                ),
            ));
        }
    }
    Ok(())
}

/// 对齐校验：swizzle 的 bank 粒度不得超过档案声明的 bank 数。
pub fn check_alignment(profile: &Profile, swizzle_bits: u8) -> diagnostic::Result<()> {
    let bank = profile.mem.fast_bank;
    if bank == 0 {
        return Ok(());
    }
    // `LayoutSpec::parse` 接受 `u8` 到 255，所以 `swz<40,…>` 是数据能写出来的。
    // 先挡住移位溢出，再比较：2^32 个 bank 不存在，任何 ≥ 32 的粒度都必然超限。
    let Some(needed) = 1u64.checked_shl(u32::from(swizzle_bits)) else {
        return Err(Error::new(
            Code::Unsupported,
            format!("swz<{swizzle_bits},…> 的 bank 粒度超出可表示范围，档案只有 {bank} 个 bank"),
        ));
    };
    if needed > u64::from(bank) {
        return Err(Error::new(
            Code::Unsupported,
            format!("swz<{swizzle_bits},…> 需要 {needed} 个 bank，档案只有 {bank} 个"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
use super::*;

fn profile() -> Profile {
    Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("CPU 档案可解析")
}

#[test]
fn maps_semantic_names_to_physical_ones() {
    let profile = profile();
    assert_eq!(physical(&profile, Space::Gm), "main");
    assert_eq!(physical(&profile, Space::Fast), "L2");
    assert_eq!(physical(&profile, Space::Reg), "stack");
}

#[test]
fn capacity_is_the_archive_declaration() {
    let profile = profile();
    assert_eq!(capacity(&profile, Space::Fast), Some(524_288));
    assert_eq!(capacity(&profile, Space::Reg), Some(65_536));
    assert_eq!(capacity(&profile, Space::Gm), None, "GM 不受限额");
}

#[test]
fn capacity_is_enforced_against_profile() {
    let profile = profile();
    assert!(check_capacity(&profile, &[SpaceUse::new(Space::Fast, 1024)]).is_ok());
    let error = check_capacity(&profile, &[SpaceUse::new(Space::Fast, 1 << 30)]).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("L2"), "{error}");
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
fn alignment_rejects_granularities_that_overflow() {
    // `swz<40,…>` 是声明里能写出来的，必须报错而不是 panic。
    let error = check_alignment(&profile(), 40).unwrap_err();
    assert!(error.to_string().contains("40"), "{error}");
}
}

