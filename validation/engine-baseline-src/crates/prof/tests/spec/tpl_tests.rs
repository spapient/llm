use super::*;

#[test]
fn text_slots_and_calls_are_split_without_copying() {
    let template = "void f() {\n    float d[{{GMAX}}];\n    @dot(acc, d, xs, gsz);\n}\n";
    let parsed = Template::parse(template);

    assert_eq!(
        parsed.slots().collect::<Vec<_>>(),
        [(2, "GMAX")],
        "槽位名借用正文，不去空白之外做任何改写"
    );
    let calls: Vec<_> = parsed.calls().map(|(line, prim, _)| (line, prim)).collect();
    assert_eq!(calls, [(3, Prim::Dot)]);
    // 文本段拼回去必须还原原文：切分只拿走两种构造，不能吃掉任何字符。
    let text_only: String = parsed
        .segments()
        .iter()
        .filter_map(|segment| match segment {
            Segment::Text(text) => Some(*text),
            _ => None,
        })
        .collect();
    assert_eq!(
        text_only,
        template
            .replace("{{GMAX}}", "")
            .replace("@dot(acc, d, xs, gsz)", "")
    );
    assert_eq!(lex(template), parsed.into_segments());
}

/// 行号必须跟着被跳过的构造一起前进：否则跨行调用的下一条诊断指错行。
#[test]
fn the_line_number_survives_a_multi_line_call() {
    let template = "void f() {\n    @bar(\n        tok,\n    );\n    {{typo}}\n}\n";
    let parsed = Template::parse(template);
    assert_eq!(parsed.slots().collect::<Vec<_>>(), [(5, "typo")]);
    let (line, prim, _) = parsed.calls().next().expect("一条调用");
    assert_eq!(prim, Prim::Bar);
    assert_eq!(line, 2);
}

#[test]
fn named_and_positional_arguments_keep_their_text() {
    let parsed = Template::parse("@cvt(v, acc, sat: off, along: 32)");
    let (_, prim, args) = parsed.calls().next().expect("一条调用");
    assert_eq!(prim, Prim::Cvt);
    assert_eq!(
        args,
        [
            Arg::Expr("v"),
            Arg::Expr("acc"),
            Arg::Named {
                key: "sat",
                value: "off"
            },
            Arg::Named {
                key: "along",
                value: "32"
            },
        ]
    );
}

#[test]
fn an_unknown_primitive_stays_text_instead_of_being_guessed() {
    let parsed = Template::parse("@nosuchprim(y, n, out);");
    assert!(parsed.calls().next().is_none(), "闭集外不猜");
    assert!(matches!(
        parsed.segments(),
        [Segment::Text("@nosuchprim(y, n, out);")]
    ));
}

#[test]
fn an_unclosed_slot_or_call_is_left_alone() {
    for template in ["{{GMAX", "@dot(acc, d"] {
        let parsed = Template::parse(template);
        assert_eq!(
            parsed.segments(),
            [Segment::Text(template)],
            "写坏了应当退回文本，而不是产生半个构造"
        );
    }
}
