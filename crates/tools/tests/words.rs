//! 一概念一词的判定面：标识符、外来 API、字面量与注释的边界。
//!
//! 只断言「哪一类算违规、哪一类不算」：词表本身在 `llm_tools::words` 里，换词表不该动这里。

use llm_tools::words::{FOREIGN, scan};

/// 违规标识符的原文，按出现顺序。
fn found(src: &str) -> Vec<String> {
    scan(src)
        .expect("用例源码可解析")
        .into_iter()
        .map(|hit| hit.found)
        .collect()
}

#[test]
fn a_full_spelling_is_flagged_with_its_canonical_form() {
    let hits = scan("fn f(buffer: usize, index: usize) {}").expect("可解析");
    let pairs: Vec<(&str, &str)> = hits
        .iter()
        .map(|hit| (hit.found.as_str(), hit.canon))
        .collect();
    assert_eq!(pairs, vec![("buffer", "buf"), ("index", "idx")]);
}

#[test]
fn the_canonical_spelling_itself_is_silent() {
    assert_eq!(found("let idx = 0; let buf = vec![];"), Vec::<String>::new());
}

#[test]
fn a_plural_folds_onto_the_same_canonical_form() {
    let hits = scan("let buffers = bufs_of(0);").expect("可解析");
    assert_eq!(hits.len(), 1, "复数要命中原名：{hits:?}");
    assert_eq!(hits[0].canon, "bufs");
}

#[test]
fn a_derived_word_is_its_own_concept() {
    // `buffered` 是"已缓冲的量"，不是 `buffer` 的另一种拼法：折成 `buf` 会让两个概念共用一词。
    assert_eq!(found("let buffered = 0;"), Vec::<String>::new());
}

#[test]
fn a_camel_name_is_split_into_words_before_judging() {
    // 本项目自造的驼峰名要拆段：`MetalBuffer` 没有豁免，`MTLBuffer` 有。
    assert_eq!(found("struct MetalBuffer;"), vec!["MetalBuffer"]);
    assert_eq!(found("struct MTLBuffer;"), Vec::<String>::new());
}

#[test]
fn foreign_api_names_are_never_renamed() {
    for name in FOREIGN {
        let src = format!("fn f() {{ let _ = {name}; }}");
        assert_eq!(found(&src), Vec::<String>::new(), "外来名 {name} 不该命中");
    }
}

#[test]
fn device_source_inside_a_literal_is_not_rust_identifiers() {
    // Metal 选择器与 OpenCL 形参只存在于字符串里：门禁看标识符，看不见它们。
    let src = r##"
        const KERNEL: &str = "kernel void k(device float* buffer [[buffer(0)]]) {}";
        const RAW: &str = r#"index buffer;"#;
    "##;
    assert_eq!(found(src), Vec::<String>::new());
}

#[test]
fn a_comment_is_not_a_definition() {
    let src = "// 这里的 buffer 与 index 只是历史提法\n/* buffer */\nfn f() {}\n";
    assert_eq!(found(src), Vec::<String>::new());
}

#[test]
fn a_foreign_member_name_is_exempt_only_after_a_dot() {
    // `at.index(3)` 是 proptest 的成员；我们自己定义的 `index` 仍要改。
    assert_eq!(found("let idx = at.index(3);"), Vec::<String>::new());
    assert_eq!(found("fn index(&self) {}"), vec!["index"]);
    assert_eq!(found("let index = 0;"), vec!["index"]);
}

#[test]
fn a_hit_carries_the_line_it_appeared_on() {
    let hits = scan("fn f() {\n    let index = 1;\n}\n").expect("可解析");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].line, 2);
    assert!(hits[0].describe().contains("`idx`"));
}
