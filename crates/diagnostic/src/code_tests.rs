use super::*;
use std::collections::BTreeSet;

#[test]
fn a_code_is_a_closed_stable_set() {
    let names: Vec<&str> = Code::ALL.iter().map(|code| code.as_str()).collect();
    assert_eq!(
        names,
        ["unsupported", "invalid", "missing", "io", "internal"]
    );
    let unique = BTreeSet::from_iter(names.iter().copied());
    assert_eq!(unique.len(), Code::ALL.len(), "类别名必须唯一");
}

#[test]
fn every_code_has_its_own_exit_code() {
    let codes: Vec<u8> = Code::ALL.iter().map(|code| code.exit_code()).collect();
    assert_eq!(codes, [3, 2, 4, 5, 70]);
    let unique = BTreeSet::from_iter(codes.iter().copied());
    assert_eq!(unique.len(), codes.len(), "退出码必须两两不同");
    assert!(!codes.contains(&0), "0 只表示成功");
}
