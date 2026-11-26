//! 源分析原语的集成测试：词法、预处理、调用扫描、圈复杂度、文档链接、行数。
//! 只用公开面，不读实现即可换掉实现。

use diagnostic::Code;
use llm_tools::call::{Kind, scan};
use llm_tools::ccn::analyze;
use llm_tools::docs::{Link, links};
use llm_tools::dup::file_lines;
use llm_tools::gate::Gate;
use llm_tools::lex::{Token, parse};
use llm_tools::strip::{strip_tests, test_modules};
use std::collections::BTreeMap;

/// 前 N 个顶层 token 的调试文本。
fn outline(tokens: &[Token], max: usize) -> String {
    tokens
        .iter()
        .take(max)
        .map(outline_piece)
        .collect::<Vec<_>>()
        .join(" ")
}

/// 单个 token 的调试文本。
fn outline_piece(token: &Token) -> String {
    match token {
        Token::Ident { text, .. } | Token::Literal { text, .. } => text.clone(),
        Token::Punct { ch, .. } => ch.to_string(),
        Token::Group { delim, .. } => format!("{}..{}", delim.open(), delim.close()),
    }
}

testkit::test_cases! {
fn strips_comments_and_strings_before_analysis() {
    let src = "// if ? &&\nfn s() -> u8 { let x = \"if ? &&\"; 1 }\n/* if ? */\n";
    let tokens = parse(src).expect("parse");
    let rendered = outline(&tokens, 20);
    assert!(rendered.contains("fn"));
    assert!(!rendered.contains("if ?"));
    assert!(!rendered.contains("/*"));
}

fn nested_groups_keep_order_and_positions() {
    let src = "fn f() { if a { b } else { c } }";
    let tokens = parse(src).expect("parse");
    assert_eq!(tokens.first().map(Token::start_line), Some(1));
    let group = tokens
        .iter()
        .find(|t| matches!(t, Token::Group { .. }))
        .expect("body group");
    assert_eq!(group.start_line(), 1);
    assert_eq!(group.end_line(), 1);
}

fn raw_and_char_literals_are_single_tokens() {
    let src = "r#\"a ? b\"#; 'c'; 'static;";
    let tokens = parse(src).expect("parse");
    assert!(matches!(&tokens[0], Token::Literal { .. }));
    assert!(matches!(&tokens[2], Token::Literal { .. }));
    // 生命周期：' 是标点，static 是标识符。
    assert!(matches!(&tokens[4], Token::Punct { ch: '\'', .. }));
    assert!(matches!(&tokens[5], Token::Ident { text, .. } if text == "static"));
}

fn unclosed_group_is_an_invalid_error() {
    let err = parse("fn f() { ").expect_err("must fail");
    assert_eq!(err.code(), Code::Invalid);
}

fn boundary_inputs_reach_a_stable_verdict() {
    let cases: &[(&str, bool)] = &[
        ("", true),
        ("   \n\t ", true),
        ("fn f() { ", false),
        ("let x = [1, 2", false),
        ("/* 注释没完", false),
        ("\"字符串没完", false),
        ("a = 1; )", true),
        ("'static", true),
    ];
    for (src, expect_ok) in cases {
        let verdict = parse(src);
        assert_eq!(verdict.is_ok(), *expect_ok, "输入：{src:?}");
        if let Err(err) = &verdict {
            assert_eq!(err.code(), Code::Invalid, "输入：{src:?}");
            assert!(!err.message().is_empty(), "拒绝必须带理由");
        }
    }
    // 饱和长标识符：单个 token，既不截断也不 panic。
    let long = "x".repeat(256);
    let tokens = parse(&long).expect("长标识符可解析");
    assert_eq!(tokens.len(), 1);
}
}

// ---------------------------------------------------------------------------
// strip：剔除测试条目，保留行号与真实代码
// ---------------------------------------------------------------------------

testkit::test_cases! {
fn strips_test_items_with_attached_attributes() {
    let src = "#[cfg(target_os = \"macos\")]\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn keep() {}\n";
    let stripped = strip_tests(src).expect("strip");
    assert_eq!(stripped.lines().count(), src.lines().count());
    assert!(!stripped.contains("mod tests"));
    assert!(!stripped.contains("target_os"));
    assert!(stripped.contains("fn keep"));
}

fn keeps_not_test_items() {
    let src = "#[cfg(not(test))]\nfn real() {}\n";
    assert!(strip_tests(src).expect("strip").contains("fn real"));
}

/// 通过公开的 `strip_tests` 验证复合 `cfg(test)` 判定：只有真正含 `test` 的条目被剔除。
fn detects_compound_test_cfg() {
    let stripped = |body: &str| {
        let src = format!("#[cfg({body})]\nfn real() {{}}\n");
        !strip_tests(&src).expect("strip").contains("fn real")
    };
    assert!(stripped("all(test, feature = \"x\")"));
    assert!(stripped("any(test, feature = \"x\")"));
    assert!(!stripped("not(test)"));
    assert!(!stripped("any(not(test), feature)"));
    assert!(!stripped("feature = \"testing\""));
}

fn strips_inner_cfg_test_attribute() {
    let src = "#![cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn keep() {}\n";
    let stripped = strip_tests(src).expect("strip");
    assert_eq!(stripped.lines().count(), src.lines().count());
    assert!(!stripped.contains("mod tests"));
    assert!(stripped.contains("fn keep"));
}

fn braces_inside_strings_do_not_break_strip() {
    let src = "#[cfg(test)]\nmod tests {\n    let t = \"void f() { {{x}} }\";\n}\nfn keep() {}\n";
    let stripped = strip_tests(src).expect("strip");
    assert!(!stripped.contains("void f"));
    assert!(stripped.contains("fn keep"));
}

/// 分号形式挂载的模块是独立文件，`strip_tests` 看不到其内容，只能由 `test_modules` 点名。
fn names_only_the_modules_mounted_under_cfg_test() {
    let src = "#[cfg(test)]\nmod mirror;\n#[cfg(test)]\npub(crate) mod boundary;\nmod production;\n#[cfg(not(test))]\nmod bench;\n#[cfg(test)]\nmod inline {\n    fn t() {}\n}\n";
    let mods: Vec<String> = test_modules(src)
        .expect("mods")
        .into_iter()
        .map(|mount| mount.name)
        .collect();
    assert_eq!(mods, ["mirror".to_owned(), "boundary".to_owned()]);
}
}

// ---------------------------------------------------------------------------
// call：禁止形态扫描
// ---------------------------------------------------------------------------

testkit::test_cases! {
fn bare_unwrap_expect_and_panic_are_caught() {
    let src = "fn f() { let a = x.unwrap(); let b = y.expect(\"m\"); panic!(\"p\"); }";
    let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect, Kind::Panic]);
}

fn unreachable_is_caught_but_its_plain_identifier_is_not() {
    let src = "fn f() { let unreachable = 1; g(unreachable); unreachable!(\"x\"); }";
    let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![Kind::Unreachable], "只有带 `!` 的那次是宏调用");
}

fn unwrap_inside_a_string_is_not_flagged() {
    let src = "let s = \"x.unwrap()\";";
    assert!(scan(src).unwrap().is_empty());
}

fn ufcs_unwrap_expect_are_caught() {
    let src = "fn f() { let a = Option::unwrap(x); let b = Result::expect(y, \"m\"); }";
    let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect]);
}

fn ordinary_associated_function_is_not_flagged() {
    let src = "fn f() { let v = Type::new(x); }";
    assert!(scan(src).unwrap().is_empty());
}

fn result_returning_code_is_clean() {
    let src = "fn f() -> Result<u32, E> { let v = ok()?; Ok(v) }";
    assert!(scan(src).unwrap().is_empty());
}
}

// ---------------------------------------------------------------------------
// ccn：圈复杂度
// ---------------------------------------------------------------------------

fn cc(src: &str) -> Vec<(String, u32)> {
    analyze(src)
        .expect("analyze")
        .into_iter()
        .map(|f| (f.name, f.ccn))
        .collect()
}

testkit::test_cases! {
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

fn function_pointer_types_are_not_definitions() {
    let src = "fn real(f: fn(u8) -> u8) -> u8 { f(1) }";
    let findings = analyze(src).expect("analyze");
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].name, "real");
}

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

fn two_top_level_functions_are_both_collected() {
    let src = "fn first() {}\nfn second() {}\n";
    assert_eq!(cc(src).len(), 2);
}
}

// ---------------------------------------------------------------------------
// docs：文档相对链接
// ---------------------------------------------------------------------------

testkit::test_cases! {
fn relative_links_are_collected_and_anchors_stripped() {
    let doc = "见 [架构](ARCHITECTURE.md#二) 与 [代码](../crates/tools/src/lib.rs)。";
    assert_eq!(
        links(doc),
        vec![
            Link {
                target: "ARCHITECTURE.md".to_string(),
                line: 1
            },
            Link {
                target: "../crates/tools/src/lib.rs".to_string(),
                line: 1
            },
        ]
    );
}

fn external_and_pure_anchor_links_are_skipped() {
    let doc = "[site](https://x.io/a) [top](#section) [mail](mailto:a@b.c)";
    assert!(links(doc).is_empty());
}

fn links_carry_line_numbers_across_multiple_lines() {
    let doc = "第一行无链接。\n第二行 [a](guide.md)。\n第三行 [b](note.md#top)。";
    assert_eq!(
        links(doc),
        vec![
            Link {
                target: "guide.md".to_string(),
                line: 2
            },
            Link {
                target: "note.md".to_string(),
                line: 3
            },
        ]
    );
}

fn empty_and_unclosed_links_are_skipped() {
    let doc = "[]()\n[open](note.md 未闭合";
    assert!(links(doc).is_empty());
}
}

// ---------------------------------------------------------------------------
// dup：文件行数闸门
// ---------------------------------------------------------------------------

/// 行数上限故意收紧到 10，便于用短文本验证边界。
fn line_gate() -> Gate {
    Gate::from_view(&llm_tools::GateView {
        ccn_max: 5,
        nloc_max: 60,
        file_max_lines: 10,
        file_min_lines: 1,
        test_max_lines: 10,
        debt: &[],
        min_lines_baseline: &[],
    })
    .expect("测试阈值合法")
}

testkit::test_cases! {
fn an_oversized_file_is_flagged() {
    let text = (0..12).map(|_| "x").collect::<Vec<_>>().join("\n");
    assert_eq!(file_lines(&text, &line_gate()), Some(12));
}

fn a_file_within_budget_is_silent() {
    let text = (0..8).map(|_| "x").collect::<Vec<_>>().join("\n");
    assert_eq!(file_lines(&text, &line_gate()), None);
}
}
