use super::*;

/// §16 的片段判据是“同一对数与两个门槛比”：上限管成败，内部目标只是余量。
#[test]
fn the_fragment_verdict_reads_the_p0_limits() {
    assert_eq!(FRAGMENT_LIMIT, 400, "README §16 的上限");
    assert_eq!(FRAGMENT_GOAL, 240, "README §16 的内部目标");
    assert!(Fragment::new(FRAGMENT_LIMIT).within_limit());
    assert!(!Fragment::new(FRAGMENT_LIMIT + 1).within_limit());
    assert!(Fragment::new(FRAGMENT_GOAL).within_goal());
    assert!(
        !Fragment::new(FRAGMENT_GOAL + 1).within_goal(),
        "超出目标是余量不足，不是失败"
    );
}

#[test]
fn fnv_is_stable_and_distinguishes_inputs() {
    assert_eq!(fnv1a("abc"), fnv1a("abc"));
    assert_ne!(fnv1a("abc"), fnv1a("abd"));
    assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
}

#[test]
fn assemble_puts_provenance_in_the_header() {
    let profile = Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案");
    let declaration = &prof::catalogue::named("gemv_i4").expect("内置");
    let op = OpDecl::parse(declaration.declaration(), declaration.template()).expect("声明");
    let variant = &op.variants[0];
    let combo = crate::decl::expand(&op).expect("展开").remove(0);
    let source = assemble(&profile, &op, variant, &combo, "int x;\n");
    assert!(source.text.contains("op=gemv_i4"));
    assert!(source.text.contains("device=cpu(portable)"));
    assert!(source.text.contains("variant=pipe2"));
    assert!(
        source.text.contains("#include <stdint.h>"),
        "前导声明要带进来"
    );
    assert!(source.text.ends_with("int x;\n"));
    assert_eq!(source.lang, Emit::C);
}
