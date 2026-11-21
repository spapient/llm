//! `cutile::space` 的用例：空间核算。
//!
//! 生产文件只留执行与判据，用例集中于此。

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
