//! 词法器集成测试：只用公开面，不读实现即可换掉实现。

use diagnostic::Code;
use llm_tools::lex::{Token, parse};

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

#[test]
fn strips_comments_and_strings_before_analysis() {
    let src = "// if ? &&\nfn s() -> u8 { let x = \"if ? &&\"; 1 }\n/* if ? */\n";
    let tokens = parse(src).expect("parse");
    let rendered = outline(&tokens, 20);
    assert!(rendered.contains("fn"));
    assert!(!rendered.contains("if ?"));
    assert!(!rendered.contains("/*"));
}

#[test]
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

#[test]
fn raw_and_char_literals_are_single_tokens() {
    let src = "r#\"a ? b\"#; 'c'; 'static;";
    let tokens = parse(src).expect("parse");
    assert!(matches!(&tokens[0], Token::Literal { .. }));
    assert!(matches!(&tokens[2], Token::Literal { .. }));
    // 生命周期：' 是标点，static 是标识符。
    assert!(matches!(&tokens[4], Token::Punct { ch: '\'', .. }));
    assert!(matches!(&tokens[5], Token::Ident { text, .. } if text == "static"));
}

#[test]
fn unclosed_group_is_an_invalid_error() {
    let err = parse("fn f() { ").expect_err("must fail");
    assert_eq!(err.code(), Code::Invalid);
}

#[test]
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
