//! 注解文法的单元测试：键顺序、必填键、闭集、错误位置与词法边角。
//!
//! 与 `syntax.rs` 分开放，是为了让生产文件只含生产代码——两边的行数各自可读，也各自受门禁约束。

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

/// 与真实词法相关的两件事：`0x10` 是一个字面量 token，`"a\nb"` 里的反斜杠是内容而不是转义。
fn lexical_corners() {
    let hex = Token::Literal("0x10".to_string());
    let escaped = Token::Literal(r#""a\nb""#.to_string());
    reject(unsigned(&hex), "十进制");
    reject(string(&escaped), "转义");
}

/// 键顺序随意、尾随逗号允许、`in` 可选、轴序不被重排。
#[test]
fn a_layer_annotation_carries_name_shape_input_and_ops() {
    let text = r#"out = [2, 3], name = "q", ops = ["gemv_i4"],"#;
    let declaration = parse(&lex(text)).expect("合法");
    assert_eq!(declaration.name, "q");
    assert_eq!(declaration.out, [2, 3], "轴序不能被重排");
    assert_eq!(declaration.ops, ["gemv_i4"]);
    assert_eq!(declaration.input, None, "没有 `in` 不是文法错误");
    let scalar = parse(&lex(r#"name = "loss", out = [], ops = ["dot"]"#)).expect("合法");
    assert!(scalar.out.is_empty(), "空形状即标量输出");
    let projected = parse(&lex(r#"out = [8], name = "q", in = 32, ops = ["dot"]"#));
    assert_eq!(projected.expect("合法").input, Some(32), "`in` 原样保留");
}

/// 层注解的每一条错误：`needle` 是提示里必须出现的那一段。
#[test]
fn every_bad_layer_annotation_is_rejected_with_an_actionable_message() {
    let cases = [
        (
            r#"name = "q", out = [2, 3], in = 4, ops = ["dot"]"#,
            "单轴输出",
        ),
        (r#"name = "q", out = [8], in = 0, ops = ["dot"]"#, "大于 0"),
        (r#"name = "x", out = [0], ops = ["dot"]"#, "大于 0"),
        (r#"name = 42, out = [1], ops = ["dot"]"#, "字符串字面量"),
        (r#"name = "x", out = [1], ops = []"#, "至少要引用一个"),
        (r#"name = "x", out = [1], ops = ["a", "a"]"#, "出现了两次"),
        (r#"name = "x", out = [1], ops = ["dot", 7]"#, "字符串字面量"),
        (r#"name = "a b", out = [1], ops = ["dot"]"#, "空白"),
        (r#"name = "", out = [1], ops = ["dot"]"#, "空字符串"),
        (
            r#"name = "a", out = [1], ops = ["dot"], dtype = "f32""#,
            "dtype",
        ),
        (r#"out = [1], ops = ["dot"]"#, "name"),
        (r#"name = "a", ops = ["dot"]"#, "out"),
        (r#"name = "a", out = [1]"#, "ops"),
        (r#"name = "a", name = "b""#, "重复"),
        (r#"out = [1], out = [2]"#, "重复"),
        (r#"out = [1], in = 4, in = 8"#, "重复"),
        (r#"ops = ["a"], ops = ["a"]"#, "重复"),
        (r#"name = "a", out, ops = ["dot"]"#, "键与值之间的 `=`"),
        (r#"name = "a", out = 3"#, "方括号"),
        (r#"name = "a", out = [1], ops = "#, "的方括号"),
        (r#"name = "a", out = [1], in = "#, "的值"),
        (r#"name = "a" out = [1]"#, "`,`"),
    ];
    for (text, needle) in cases {
        reject(parse(&lex(text)), needle);
    }
    // 未知键要把整个闭集列出来，用户不用去翻源码。
    let unknown = parse(&lex(r#"name = "a", out = [1], ops = ["d"], nope = 1"#));
    let error = reject(unknown, "nope");
    for key in KEYS {
        assert!(error.message.contains(key), "{error:?} 应列出 {key}");
    }
    lexical_corners();
}

/// 报错指在写错的那一处：缺分隔符、未知键、方括号组各一例。
#[test]
fn an_error_points_at_the_token_the_user_wrote() {
    let error = reject(parse(&lex(r#"name = "a" out = [1]"#)), "`,`");
    assert_eq!(error.position, 3, "缺的那个逗号在下标 3 处");
    let tokens = lex(r#"name = "a", out = [1], ops = ["dot"], bogus = "x""#);
    let error = reject(parse(&tokens), "bogus");
    assert_eq!(error.position, 12, "`bogus` 是自己那一处，不是注解开头");
    assert_eq!(tokens[error.position], Token::Ident("bogus".to_string()));
    let tokens = lex(r#"name = "a", out = [1], ops = ["dot", 7]"#);
    let error = reject(parse(&tokens), "字符串字面量");
    assert_eq!(error.position, 10, "错误指向 `ops` 的方括号组");
    assert!(matches!(tokens[error.position], Token::Group(..)));
}

/// 前端注解：三个键都是必填，键顺序不影响结果，闭集之外一律拒收。
#[test]
fn a_frontend_annotation_carries_three_keys_from_the_closed_sets() {
    let text = r#"tag = "w", emit = "metal", ext = "metal""#;
    let frontend = parse_frontend(&lex(text)).expect("合法前端");
    assert_eq!(
        (frontend.emit, frontend.ext, frontend.tag),
        ("metal".to_string(), "metal".to_string(), "w".to_string())
    );
    for (text, needle) in [
        (r#"emit = "metal", ext = "metal""#, "tag"),
        (r#"emit = "metal", tag = "w""#, "ext"),
        (r#"ext = "metal", tag = "w""#, "emit"),
        (r#"emit = "vulkan", ext = "spv", tag = "w""#, "vulkan"),
        (r#"emit = "metal", emit = "metal""#, "重复"),
        (r#"emit = "metal", nope = "x""#, "nope"),
    ] {
        reject(parse_frontend(&lex(text)), needle);
    }
}

/// 行为不是配置：两道类型注解只标具体类型，泛型与行为都在这里被拒；`#[model]` 只标别名。
#[test]
fn only_a_concrete_type_is_annotated() {
    assert_eq!(
        target_name(&lex("pub struct OutputHead;"), "spec").expect("struct"),
        "OutputHead"
    );
    assert_eq!(
        target_name(&lex("pub(crate) enum Head { A }"), "spec").expect("enum"),
        "Head"
    );
    for (text, keyword) in [
        ("fn f() {}", "fn"),
        ("trait T {}", "trait"),
        ("impl T {}", "impl"),
        ("mod m {}", "mod"),
        ("union U { a: u8 }", "union"),
        ("type A = u8;", "type"),
    ] {
        let error = reject(target_name(&lex(text), "spec"), keyword);
        assert!(error.message.contains(keyword), "{text}: {error:?}");
    }
    let error = reject(target_name(&lex("struct Head<T> { x: T }"), "spec"), "泛型");
    assert_eq!(error.position, 1, "错误指向类型名");
    reject(target_name(&[], "spec"), "找不到类型定义");
    assert_eq!(
        model_target(&lex("pub type I4 = Quantized<32>;"), "model").expect("别名"),
        "I4"
    );
    reject(model_target(&lex("pub struct Head;"), "model"), "type");
    reject(model_target(&lex("pub type Head<T> = T;"), "model"), "泛型");
}

/// 语义表注解：四个键齐全、骨架缺省 `gemv`、`gemm` 只认三个注册名。
#[test]
fn a_model_declaration_carries_name_shape_dtype_and_skeleton() {
    let text = r#"name = "gemv_i4", shape = [4, 4096], dtype = "f32""#;
    let model = parse_model(&lex(text)).expect("合法注解");
    assert_eq!(model.name, "gemv_i4");
    assert_eq!(model.shape, [4, 4096]);
    assert_eq!(model.dtype, "f32");
    assert_eq!(model.skeleton, "gemv", "缺省值必须等于生成器实际走的分支");
    for name in GEMM_SKELETON_NAMES {
        let text = format!(r#"name = "{name}", shape = [4, 4], dtype = "f32", skeleton = "gemm""#);
        assert_eq!(parse_model(&lex(&text)).expect("合法").skeleton, "gemm");
    }
}

/// 语义表注解的每一条错误：缺键点名、形状恰好两个轴、闭集之外、重复键。
#[test]
fn every_bad_model_annotation_is_rejected_with_an_actionable_message() {
    let cases = [
        (r#"shape = [4, 4], dtype = "f32""#, "name"),
        (r#"name = "a", dtype = "f32""#, "shape"),
        (r#"name = "a", shape = [4, 4]"#, "dtype"),
        (r#"name = "a", shape = [4, 4, 4], dtype = "f32""#, "两个轴"),
        (r#"name = "a", shape = [4], dtype = "f32""#, "两个轴"),
        (r#"name = "a", shape = [4, 0], dtype = "f32""#, "大于 0"),
        (r#"name = "a", shape = [4, 4], dtype = "f64""#, "闭集"),
        (
            r#"name = "a", shape = [4, 4], dtype = "f32", x = 1"#,
            "未知的注解键",
        ),
        (
            r#"name = "a", name = "b", shape = [4, 4], dtype = "f32""#,
            "重复",
        ),
        (
            r#"name = "a", shape = [4, 4], dtype = "f32", skeleton = "x""#,
            "闭集",
        ),
        (
            r#"name = "gemv_f32", shape = [4, 4], dtype = "f32", skeleton = "gemm""#,
            "gemm_f32",
        ),
    ];
    for (text, needle) in cases {
        reject(parse_model(&lex(text)), needle);
    }
}

/// 手工构造 token 的小工厂：词法边角（非字面量出现在值/键位、括号组出现在键位）夹具模拟不出来。
fn ident(word: &str) -> Token {
    Token::Ident(word.to_string())
}

fn literal(text: &str) -> Token {
    Token::Literal(text.to_string())
}

fn punct(mark: char) -> Token {
    Token::Punct(mark)
}

/// 键位必须是标识符：字面量、标点、括号组各自被读成可读的 token，错误指在那一处。
#[test]
fn a_key_position_must_be_an_identifier() {
    let tokens = [literal("42"), punct('='), ident("x")];
    let error = reject(parse(&tokens), "期望键名");
    assert!(matches!(tokens[error.position], Token::Literal(..)));
    let group = Token::Group(Bracket::Square, Vec::new());
    let error = reject(parse(&[group, punct('=')]), "期望键名");
    assert!(
        error.message.contains("`[..]`"),
        "括号组要按括号种类读：{error:?}"
    );
}

/// 字符串与十进制整数只接受对应字面量；非字面量、缺引号、非数字、超 `usize` 各有一句判据。
#[test]
fn string_and_unsigned_reject_every_non_literal_shape() {
    reject(string(&ident("name")), "字符串字面量");
    reject(unsigned(&ident("42")), "整数字面量");
    reject(string(&literal("42")), "普通字符串");
    reject(unsigned(&literal("4.2")), "十进制");
    reject(unsigned(&literal("1_000")), "十进制");
    reject(unsigned(&literal("99999999999999999999999")), "usize");
}

/// 类型注解的形状判据：缺关键字、`type` 后无名、既不是 `=` 也不是 `<`、`struct`/`enum` 后无名。
#[test]
fn type_target_rejects_every_malformed_shape() {
    reject(model_target(&[], "model"), "找不到类型定义");
    reject(model_target(&[ident("type"), punct(';')], "model"), "别名");
    reject(
        model_target(&[ident("type"), ident("A"), punct(';')], "model"),
        "具体格式",
    );
    reject(
        target_name(&[ident("struct"), punct(';')], "spec"),
        "类型名",
    );
}

/// 值访问器只认自己的变体：其余变体一律回落到空，避免把别的槽位的值错当成这一键。
#[test]
fn value_accessors_are_empty_outside_their_own_variant() {
    assert_eq!(Value::Empty.text(), "");
    assert!(Value::Empty.names().is_empty());
    let (axes, at) = Value::Empty.axes();
    assert!(axes.is_empty() && at == 0);

    let width = Value::Width(3, 2);
    assert_eq!(width.text(), "");
    assert!(width.names().is_empty());
    assert!(width.axes().0.is_empty());
}

/// 名字既不能是空白，也不能含控制字符；两类坏字符各有一句指向名字本身的提示。
#[test]
fn a_name_rejects_whitespace_and_control_characters() {
    reject(checked_name(" ", 0), "空白");
    reject(checked_name("a\u{0}b", 0), "控制字符");
}

/// 每种括号在错误信息里都有可读写法；透明组没有可见括号，要写成「不可见分组」。
#[test]
fn every_bracket_describes_a_readable_group() {
    assert!(Bracket::Paren.describe().contains("("));
    assert!(Bracket::Brace.describe().contains("{"));
    assert!(Bracket::Square.describe().contains("["));
    assert!(Bracket::Invisible.describe().contains("不可见"));
}

/// 确定性伪随机：整数混合、无随机源、无全局状态，同一 (`seed`, `case`) 在任何机器上得到同一串文本。
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    fn pick(&mut self, pool: &[char]) -> char {
        pool[(self.next() as usize) % pool.len()]
    }
}

/// 一趟确定性模糊回归的三条契约（与 `llm-fuzz` 同一判据，在本 crate 内自包含：注解层不允许拉
/// `diagnostic` / `prof` 依赖边）：不 panic、同一输入两次结论一致、拒绝时消息非空。
fn sweep_one<D: std::fmt::Debug + PartialEq>(
    parse: fn(&[Token]) -> Result<D, Error>,
    text: &str,
    case: u64,
) {
    let tokens = lex(text);
    let first = std::panic::catch_unwind(|| parse(&tokens));
    let second = std::panic::catch_unwind(|| parse(&tokens));
    assert!(first.is_ok(), "用例 {case} 在 `{text:?}` 上 panic");
    let first = first.unwrap();
    let second = second.unwrap();
    assert_eq!(first, second, "用例 {case} 两次结论不一致：`{text:?}`");
    if let Err(error) = &first {
        assert!(
            !error.message.trim().is_empty(),
            "用例 {case} 被静默拒绝：`{text:?}`"
        );
    }
}

/// 128 个确定性用例分别打三道文法；括号组夹具已由手写 token 用例覆盖，这里只压标识符 / 数字 /
/// 标点 / 空白 / 转义的乱序组合，看门在任何输入下都给出可复现的结论。
#[test]
fn a_deterministic_sweep_keeps_every_parser_resilient_and_stable() {
    const POOL: [char; 16] = [
        'a', 'b', 'm', 'n', '0', '1', '9', '_', '=', ',', '"', '<', '>', ';', ' ', '\\',
    ];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for case in 0..128 {
        let length = (rng.next() % 12) as usize;
        let text: String = (0..length).map(|_| rng.pick(&POOL)).collect();
        sweep_one(parse, &text, case);
        sweep_one(parse_frontend, &text, case);
        sweep_one(parse_model, &text, case);
    }
}
