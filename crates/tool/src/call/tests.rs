use super::*;

#[test]
fn bare_unwrap_expect_and_panic_are_caught() {
    let src = "fn f() { let a = x.unwrap(); let b = y.expect(\"m\"); panic!(\"p\"); }";
    let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect, Kind::Panic]);
}

/// `unreachable!` 是 panic 路径但没有 lint 覆盖：漏掉它，「零 panic」就只覆盖一部分。
#[test]
fn unreachable_is_caught_but_its_plain_identifier_is_not() {
    let src = "fn f() { let unreachable = 1; g(unreachable); unreachable!(\"x\"); }";
    let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![Kind::Unreachable], "只有带 `!` 的那次是宏调用");
}

#[test]
fn unwrap_inside_a_string_is_not_flagged() {
    let src = "let s = \"x.unwrap()\";";
    assert!(scan(src).unwrap().is_empty());
}

#[test]
fn result_returning_code_is_clean() {
    let src = "fn f() -> Result<u32, E> { let v = ok()?; Ok(v) }";
    assert!(scan(src).unwrap().is_empty());
}
