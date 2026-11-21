use super::*;
use diagnostic::Code;

#[test]
fn splice_substitutes_positional_and_named_holes() {
    let args = [
        Arg::Expr("32"),
        Arg::Named {
            key: "acc",
            value: "out",
        },
    ];
    let out = splice("g=$0; s=$acc", &args, 1).expect("占位符可解析");
    assert_eq!(out, "g=32; s=out");
}

#[test]
fn splice_repeats_the_same_hole() {
    let args = [Arg::Expr("x")];
    let out = splice("$0 -> $0", &args, 1).expect("重复占位");
    assert_eq!(out, "x -> x");
}

#[test]
fn missing_positional_arg_is_an_error() {
    let args = [Arg::Expr("x")];
    let err = splice("$1", &args, 7).expect_err("缺位置参数");
    assert!(err.to_string().contains("$1"), "{err}");
    assert_eq!(err.code(), Code::Invalid, "缺绑定是模板/片段契约不合法");
}

#[test]
fn missing_named_arg_is_an_error() {
    let err = splice("$nope", &[], 7).expect_err("缺命名参数");
    assert!(err.to_string().contains("nope"), "{err}");
    assert_eq!(err.code(), Code::Invalid, "缺绑定是模板/片段契约不合法");
}

#[test]
fn isolated_dollar_stays_literal() {
    let out = splice("cost $ 5", &[], 1).expect("孤立美元号不是占位符");
    assert_eq!(out, "cost $ 5");
}

#[test]
fn place_strips_trailing_semicolon_and_aligns_lines() {
    let mut text = String::from("    call(");
    place(&mut text, "a;\nb;");
    // 缩进对齐到当前整行宽度（`    call(` 共 9 个字符），第二行续到同宽。
    assert_eq!(text, "    call(a;\n         b");
}

#[test]
fn place_without_indent_keeps_flat_text() {
    let mut text = String::new();
    place(&mut text, "only;");
    assert_eq!(text, "only");
}
