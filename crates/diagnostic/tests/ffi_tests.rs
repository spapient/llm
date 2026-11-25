use super::*;
use std::collections::BTreeSet;

fn sig(name: &str, params: &[FfiParam]) -> FfiSignature {
    FfiSignature {
        name: name.to_owned(),
        params: params.to_vec(),
    }
}

#[test]
fn matching_signatures_pass() {
    let expected: BTreeSet<&str> = ["a", "b"].into_iter().collect();
    let left = [
        sig("a", &[FfiParam::PtrConst, FfiParam::Size]),
        sig("b", &[FfiParam::Size]),
    ];
    let right = [
        sig("a", &[FfiParam::PtrConst, FfiParam::Size]),
        sig("b", &[FfiParam::Size]),
    ];
    assert!(ffi_align(&expected, &left, &right).is_ok());
}

#[test]
fn shape_mismatch_is_rejected() {
    let expected: BTreeSet<&str> = ["a"].into_iter().collect();
    let left = [sig("a", &[FfiParam::PtrConst])];
    let right = [sig("a", &[FfiParam::PtrMut])];
    let err = ffi_align(&expected, &left, &right).unwrap_err();
    assert_eq!(err.code(), Code::Invalid);
    assert!(err.to_string().contains("形状不一致"));
}

#[test]
fn missing_on_either_side_is_rejected() {
    let expected: BTreeSet<&str> = ["a", "b"].into_iter().collect();
    let left = [sig("a", &[])];
    let right = [sig("b", &[])];
    let err = ffi_align(&expected, &left, &right).unwrap_err();
    assert_eq!(err.code(), Code::Missing);
}

#[test]
fn unregistered_symbol_is_rejected() {
    let expected: BTreeSet<&str> = ["a"].into_iter().collect();
    let left = [sig("a", &[]), sig("z", &[])];
    let right = [sig("a", &[])];
    let err = ffi_align(&expected, &left, &right).unwrap_err();
    assert!(err.to_string().contains("未登记"));
}
