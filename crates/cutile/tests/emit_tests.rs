use super::*;

testkit::test_cases! {
/// §16 的片段判据是“同一对数与两个门槛比”：上限管成败，内部目标只是余量。
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

fn fnv_is_stable_and_distinguishes_inputs() {
    assert_eq!(fnv1a("abc"), fnv1a("abc"));
    assert_ne!(fnv1a("abc"), fnv1a("abd"));
    assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
}

fn assemble_puts_provenance_in_the_header() {
    let profile = Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案");
    let op = prof::catalogue::named("gemv_i4")
        .expect("内置")
        .to_decl()
        .expect("声明");
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

fn assemble_defines_the_c_float4_type_for_template_kernels() {
    let profile = Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案");
    let op = prof::catalogue::named("add_f32x4")
        .expect("向量算子")
        .to_decl()
        .expect("声明");
    let variant = &op.variants[0];
    let combo = crate::decl::expand(&op).expect("展开").remove(0);
    let source = assemble(&profile, &op, variant, &combo, "float4 value;\n");
    assert_eq!(
        source
            .text
            .matches("typedef float float4 __attribute__((vector_size(16)));")
            .count(),
        1
    );
    assert!(source.text.ends_with("float4 value;\n"));
    let source = assemble(&profile, &op, variant, &combo, &source.text);
    assert_eq!(
        source
            .text
            .matches(crate::printer::C_FLOAT4_TYPEDEF)
            .count(),
        1
    );
}
}
