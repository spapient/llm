use super::*;

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
