use super::*;

#[test]
fn strips_test_items_with_attached_attributes() {
    let src = "#[cfg(target_os = \"macos\")]\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn keep() {}\n";
    let stripped = strip_tests(src).expect("strip");
    assert_eq!(stripped.lines().count(), src.lines().count());
    assert!(!stripped.contains("mod tests"));
    assert!(!stripped.contains("target_os"));
    assert!(stripped.contains("fn keep"));
}

#[test]
fn keeps_not_test_items() {
    let src = "#[cfg(not(test))]\nfn real() {}\n";
    assert!(strip_tests(src).expect("strip").contains("fn real"));
}

#[test]
fn detects_compound_test_cfg() {
    assert!(is_test_body("all(test, feature = \"x\")"));
    assert!(is_test_body("any(test, feature = \"x\")"));
    assert!(!is_test_body("not(test)"));
    assert!(!is_test_body("any(not(test), feature)"));
    assert!(!is_test_body("feature = \"testing\""));
}

#[test]
fn strips_inner_cfg_test_attribute() {
    let src = "#![cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn keep() {}\n";
    let stripped = strip_tests(src).expect("strip");
    assert_eq!(stripped.lines().count(), src.lines().count());
    assert!(!stripped.contains("mod tests"));
    assert!(stripped.contains("fn keep"));
}

#[test]
fn braces_inside_strings_do_not_break_strip() {
    let src = "#[cfg(test)]\nmod tests {\n    let t = \"void f() { {{x}} }\";\n}\nfn keep() {}\n";
    let stripped = strip_tests(src).expect("strip");
    assert!(!stripped.contains("void f"));
    assert!(stripped.contains("fn keep"));
}
