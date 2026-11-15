use super::*;

fn cc(src: &str) -> Vec<(String, u32)> {
    analyze(src)
        .expect("analyze")
        .into_iter()
        .map(|f| (f.name, f.ccn))
        .collect()
}

#[test]
fn counts_conditions_like_lizard() {
    let src = r#"
fn branch(a: bool, b: bool) -> u32 {
    if a && b { 1 } else { 2 }
}
fn choose(x: Option<u32>) -> u32 {
    match x {
        Some(0) => 0,
        Some(v) => v,
        None => 1,
    }
}
fn fallible(v: Option<u32>) -> Result<u32, ()> {
    let Some(x) = v else { return Err(()) };
    Ok(x?)
}
"#;
    assert_eq!(
        cc(src),
        vec![
            ("branch".into(), 3),
            ("choose".into(), 3),
            ("fallible".into(), 3)
        ]
    );
}

#[test]
fn function_pointer_types_are_not_definitions() {
    let src = "fn real(f: fn(u8) -> u8) -> u8 { f(1) }";
    let findings = analyze(src).expect("analyze");
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].name, "real");
}

#[test]
fn nested_functions_keep_their_own_complexity() {
    let src = "fn outer() { fn inner(a: bool) { if a { } } }";
    let by_name: BTreeMap<String, u32> = analyze(src)
        .expect("analyze")
        .into_iter()
        .map(|f| (f.name, f.ccn))
        .collect();
    assert_eq!(by_name["inner"], 2);
    assert_eq!(by_name["outer"], 1);
}

#[test]
fn two_top_level_functions_are_both_collected() {
    let src = "fn first() {}\nfn second() {}\n";
    assert_eq!(cc(src).len(), 2);
}
