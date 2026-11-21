//! 运行时文法的单元测试：洞的顺序、跨键冲突、每个错误路径与目标类型判据。

use super::*;
use crate::fixture::lex;

/// 断言 `result` 的报错里出现 `needle`，并把错误交回给调用方断位置。
fn reject<T: std::fmt::Debug>(result: Result<T, Error>, needle: &str) -> Error {
    let error = result.expect_err("不合文法");
    assert!(
        error.message.contains(needle),
        "{needle} 不在提示里：{error:?}"
    );
    error
}

/// 手工构造 token：夹具只认方括号组，字段体要按真实词法的花括号组写。
fn ident(word: &str) -> Token {
    Token::Ident(word.to_string())
}

fn punct(mark: char) -> Token {
    Token::Punct(mark)
}

/// 运行时形态：每个值都是 `$参数`，轴序原样保留，重复轴宽合法，角色可选。
#[test]
fn a_runtime_annotation_carries_parameter_names_in_order() {
    let text = r#"name = $name, out = [$tokens, $dim], ops = [$op], role = $role"#;
    let declaration = parse_runtime(&lex(text)).expect("合法");
    assert_eq!(declaration.name, "name");
    assert_eq!(declaration.out, ["tokens", "dim"]);
    assert_eq!(declaration.ops, ["op"]);
    assert_eq!(declaration.role.as_deref(), Some("role"));

    let square =
        parse_runtime(&lex(r#"name = $n, out = [$dim, $dim], ops = [$op]"#)).expect("合法");
    assert_eq!(square.out, ["dim", "dim"], "同一轴宽可以出现两次");
    assert_eq!(square.role, None, "没写角色走默认值");
    assert!(is_runtime(&lex(text)), "`$` 是运行时形态的唯一判据");
    assert!(!is_runtime(&lex(r#"name = "x", out = [1], ops = ["o"]"#)));
}

/// 运行时形态的每一条错误：字面量、裸 `$`、缺键、重复、跨类复用、`in` 不在表里。
#[test]
fn every_bad_runtime_annotation_is_rejected_with_an_actionable_message() {
    let cases = [
        (r#"name = "x", out = [$a], ops = [$o]"#, "`$`"),
        (r#"name = $, out = [$a], ops = [$o]"#, "参数名"),
        (r#"name = $n, out = [$a], ops = []"#, "至少要引用一个"),
        (r#"name = $n, out = [$a], ops = [$o, $o]"#, "出现了两次"),
        (r#"name = $n, out = [$n], ops = [$o]"#, "两种类型"),
        (r#"name = $n, out = [$a], ops = [$o], role = "norm""#, "`$`"),
        (r#"name = $n, out = [$a], ops = [$o], in = 4"#, "in"),
        (r#"out = [$a], ops = [$o]"#, "name"),
        (r#"name = $n, ops = [$o]"#, "out"),
        (r#"name = $n, out = [$a]"#, "ops"),
        (r#"name = $n, name = $m, out = [$a], ops = [$o]"#, "重复"),
        (r#"name = $n, out = [$a $b], ops = [$o]"#, "`,`"),
    ];
    for (text, needle) in cases {
        reject(parse_runtime(&lex(text)), needle);
    }
}

/// 运行时形态只标命名字段 struct：枚举、元组结构体与缺字段体都在注入前被拒。
#[test]
fn a_runtime_target_must_be_a_named_field_struct() {
    let named_fields = [
        ident("pub"),
        ident("struct"),
        ident("Layer"),
        Token::Group(
            Bracket::Brace,
            vec![ident("width"), punct(':'), ident("usize")],
        ),
    ];
    assert_eq!(
        runtime_target(&named_fields, "spec").expect("命名字段"),
        "Layer"
    );
    reject(
        runtime_target(&lex("pub enum Layer { A }"), "spec"),
        "struct",
    );
    reject(runtime_target(&lex("pub struct Layer;"), "spec"), "字段体");
    reject(
        runtime_target(&lex("pub struct Layer(usize);"), "spec"),
        "字段体",
    );
    reject(
        runtime_target(&lex("struct Layer<T> { x: T }"), "spec"),
        "泛型",
    );
}
