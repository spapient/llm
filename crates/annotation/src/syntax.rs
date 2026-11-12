//! 注解文法：与 `proc_macro` 解耦的纯解析层。
//!
//! 分成这一层的理由是可测性：`proc_macro::TokenStream` 只在过程宏展开期可用，若把文法直接写在
//! 它上面，每一条错误路径都"只有宏展开时才走得到"，等于永远没有测试。本模块只依赖下面这个
//! 最小 token 模型，因此 `cargo test` 能逐条覆盖正常路径与全部错误路径；上一层
//! [`crate::layer`] 只负责把 `TokenStream` 翻译成 [`Token`]，并把错误定位回 span。
//!
//! 文法就是一句话：`键 = 值 (, 键 = 值)*`，键是 [`KEYS`] 里的闭集，值只接受字符串字面量或
//! 方括号数组，键顺序随意、尾随逗号允许、每个键只能出现一次。

use std::fmt;

/// token 在输入里的下标。错误用它指回用户写的那一处，再由上层换成 span。
pub(crate) type Position = usize;

/// 括号种类：不直接用 `proc_macro::Delimiter`，否则本模块又绑回过程宏 API。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bracket {
    /// `(..)`。
    Paren,
    /// `[..]`。
    Square,
    /// `{..}`。
    Brace,
    /// `proc_macro::Delimiter::None`：不显示的分组，按 `(..)` 之外的第三种处理。
    Invisible,
}

impl Bracket {
    /// 错误信息里怎么称呼它。
    pub(crate) fn text(self) -> &'static str {
        match self {
            Self::Paren => "`(..)`",
            Self::Square => "`[..]`",
            Self::Brace => "`{..}`",
            Self::Invisible => "不可见分组",
        }
    }
}

/// 一个 token：够文法用即可，不复制 `proc_macro` 的全部种类。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    /// 标识符或关键字，例如 `name`、`struct`。
    Ident(String),
    /// 单字符标点，例如 `=`、`,`、`<`。
    Punct(char),
    /// 字面量的原始文本，例如 `"lm_head"`、`4096`。
    Literal(String),
    /// 带括号的组。
    Group(Bracket, Vec<Token>),
}

impl Token {
    /// 错误信息里怎么读它。
    fn describe(&self) -> String {
        match self {
            Self::Ident(word) => format!("`{word}`"),
            Self::Punct(punct) => format!("`{punct}`"),
            Self::Literal(text) => format!("字面量 `{text}`"),
            Self::Group(bracket, _) => bracket.text().to_string(),
        }
    }
}

/// 文法错误：出错位置 + 一句可执行的提示。
///
/// `position` 是输入 token 的下标，上层用它取出真实 span，因此错误能指在用户写错的那个词上，
/// 而不是笼统地指在被标注的类型上。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Error {
    position: Position,
    message: String,
}

impl Error {
    fn at(position: Position, message: impl Into<String>) -> Self {
        Self {
            position,
            message: message.into(),
        }
    }

    /// 把组内错误指回组本身：组内 token 没有独立 span 可指，指向方括号最清楚。
    fn rebased(self, position: Position) -> Self {
        Self {
            position,
            message: self.message,
        }
    }

    /// 出错 token 的下标。
    pub(crate) fn position(&self) -> Position {
        self.position
    }

    /// 一句可执行的提示。
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// 注解解析后的四件事实：名字、输出形状、输入宽度、算子列表。
///
/// `input` 是可选件：没有它，层仍然是一份合法的声明（名字 + 输出 + 算子），只是算不出下界——
/// 算子语义表要的是 `(规约轴, 输出轴)`，缺一半就宁可不给数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Declaration {
    name: String,
    out: Vec<usize>,
    input: Option<usize>,
    ops: Vec<String>,
}

impl Declaration {
    /// 层名（与权重条目前缀一致）。
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// 输出形状；空切片表示标量输出。
    pub(crate) fn out(&self) -> &[usize] {
        &self.out
    }

    /// 输入（规约轴）宽度；`None` 表示这一层没有可计价的形状。
    pub(crate) fn input(&self) -> Option<usize> {
        self.input
    }

    /// 要引用的算子名，已去重且非空。
    pub(crate) fn ops(&self) -> &[String] {
        &self.ops
    }
}

/// 注解允许的键；顺序即错误信息里的提示顺序。
pub(crate) const KEYS: [&str; 4] = ["name", "out", "in", "ops"];

/// 解析注解本体；任何一处不合文法都只产出一条带位置的错误。
pub(crate) fn parse(tokens: &[Token]) -> Result<Declaration, Error> {
    let mut cursor = Cursor::new(tokens);
    let mut name: Option<String> = None;
    let mut out: Option<Vec<usize>> = None;
    let mut input: Option<(usize, Position)> = None;
    let mut ops: Option<Vec<String>> = None;

    while !cursor.is_empty() {
        let key_position = cursor.position();
        let key = cursor.ident("键名")?;
        cursor.punct('=', "键与值之间的 `=`")?;
        match key.as_str() {
            "name" => {
                if name.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `name` 重复：名字只能声明一次",
                    ));
                }
                let token = cursor.next("`name` 的值")?;
                let text = string(token).map_err(|error| error.rebased(key_position))?;
                checked_name(&text).map_err(|error| error.rebased(key_position))?;
                name = Some(text);
            }
            "out" => {
                if out.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `out` 重复：输出形状只能声明一次",
                    ));
                }
                let (group, array) = cursor.group(Bracket::Square, "`out = [..]` 的方括号")?;
                let mut dims = Vec::with_capacity(array.len());
                for item in elements(group, array, "`out` 的元素")? {
                    let dimension = unsigned(item).map_err(|error| error.rebased(group))?;
                    if dimension == 0 {
                        return Err(Error::at(
                            group,
                            "输出形状的每个轴都必须大于 0：零宽轴不是一个可计算的形状",
                        ));
                    }
                    dims.push(dimension);
                }
                out = Some(dims);
            }
            "in" => {
                if input.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `in` 重复：输入宽度只能声明一次",
                    ));
                }
                let token = cursor.next("`in` 的值")?;
                let width = unsigned(token).map_err(|error| error.rebased(key_position))?;
                if width == 0 {
                    return Err(Error::at(
                        key_position,
                        "输入宽度必须大于 0：零宽的规约轴没有下界",
                    ));
                }
                input = Some((width, key_position));
            }
            "ops" => {
                if ops.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `ops` 重复：算子列表只能声明一次",
                    ));
                }
                let (group, array) = cursor.group(Bracket::Square, "`ops = [\"..\"]` 的方括号")?;
                let mut names: Vec<String> = Vec::with_capacity(array.len());
                for item in elements(group, array, "`ops` 的元素")? {
                    let text = string(item).map_err(|error| error.rebased(group))?;
                    checked_name(&text).map_err(|error| error.rebased(group))?;
                    if names.contains(&text) {
                        return Err(Error::at(
                            group,
                            format!("算子 `{text}` 在同一个层里出现了两次：多写的这一次不会生效"),
                        ));
                    }
                    names.push(text);
                }
                ops = Some(names);
            }
            other => {
                return Err(Error::at(
                    key_position,
                    format!("未知的注解键 `{other}`（可用：{}）", KEYS.join("、")),
                ));
            }
        }
        if cursor.is_empty() {
            break;
        }
        cursor.punct(',', "键之间的 `,`")?;
    }

    let name = name.ok_or_else(|| Error::at(0, "缺少必填键 `name = \"..\"`"))?;
    let out = out.ok_or_else(|| Error::at(0, "缺少必填键 `out = [..]`"))?;
    let ops = ops.ok_or_else(|| Error::at(0, "缺少必填键 `ops = [\"..\"]`"))?;
    if ops.is_empty() {
        return Err(Error::at(
            tokens.len().saturating_sub(1),
            "`ops` 不能为空：一个层至少要引用一个已声明的算子",
        ));
    }
    if let Some((width, position)) = input
        && out.len() != 1
    {
        // `(m, n)` 只描述“一条规约轴 + 一条输出轴”。多轴输出需要形状上下文，
        // 而不是把某一根轴冒充 `n`；宁可让注解在这里失败，也不要生成一个假的下界。
        return Err(Error::at(
            position,
            format!(
                "`in = {width}` 只对单轴输出（一次投影）有意义：`out` 现在有 {} 个轴",
                out.len()
            ),
        ));
    }
    Ok(Declaration {
        name,
        out,
        input: input.map(|(width, _)| width),
        ops,
    })
}

/// 从被标注的项里取出类型名；顺带拒绝无法用一份注解覆盖的形态。
pub(crate) fn type_name(tokens: &[Token]) -> Result<String, Error> {
    for (position, token) in tokens.iter().enumerate() {
        let Token::Ident(word) = token else {
            continue;
        };
        match word.as_str() {
            "fn" | "trait" | "impl" | "mod" | "union" | "type" | "static" | "const" => {
                return Err(Error::at(
                    position,
                    format!("`#[layer]` 只能标注 struct / enum，得到 `{word}`：层是类型，不是行为"),
                ));
            }
            "struct" | "enum" => {
                let Some(Token::Ident(name)) = tokens.get(position + 1) else {
                    return Err(Error::at(position, format!("`{word}` 后应当紧跟类型名")));
                };
                if matches!(tokens.get(position + 2), Some(Token::Punct('<'))) {
                    return Err(Error::at(
                        position + 1,
                        format!("`#[layer]` 不支持泛型参数：`{name}` 请改成形状固定的具体类型"),
                    ));
                }
                return Ok(name.clone());
            }
            _ => {}
        }
    }
    Err(Error::at(
        0,
        "`#[layer]` 只能标注 struct / enum：找不到类型定义",
    ))
}

/// 数组元素：`元素 (, 元素)*`，允许尾随逗号；任何错误都指回方括号组。
fn elements<'a>(group: Position, tokens: &'a [Token], what: &str) -> Result<Vec<&'a Token>, Error> {
    let mut cursor = Cursor::new(tokens);
    let mut elements = Vec::new();
    while !cursor.is_empty() {
        let item = cursor.next(what).map_err(|error| error.rebased(group))?;
        elements.push(item);
        if cursor.is_empty() {
            break;
        }
        cursor
            .punct(',', "数组元素之间的 `,`")
            .map_err(|error| error.rebased(group))?;
    }
    Ok(elements)
}

/// 名字（层名与算子名）的公共约束：非空、不含空白与控制字符。
fn checked_name(name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::at(0, "名字不能是空字符串"));
    }
    if let Some(bad) = name
        .chars()
        .find(|character| character.is_whitespace() || character.is_control())
    {
        return Err(Error::at(
            0,
            format!("名字不能包含空白或控制字符（发现 `{bad:?}`）：`{name}`"),
        ));
    }
    Ok(())
}

/// 只接受普通字符串字面量：转义会让"注解里写的名字"与"实际名字"不再逐字相同。
fn string(token: &Token) -> Result<String, Error> {
    let Token::Literal(text) = token else {
        return Err(Error::at(
            0,
            format!("期望字符串字面量（`\"..\"`），得到 {}", token.describe()),
        ));
    };
    let inner = text
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .ok_or_else(|| {
            Error::at(
                0,
                format!("期望普通字符串字面量（`\"..\"`），得到 `{text}`"),
            )
        })?;
    if inner.contains('\\') {
        return Err(Error::at(
            0,
            format!("注解不支持转义序列，请直接写字面内容：`{text}`"),
        ));
    }
    Ok(inner.to_string())
}

/// 只接受十进制无符号整数字面量：形状是数值，`0x10` 与 `4usize` 会让注解出现两套写法。
fn unsigned(token: &Token) -> Result<usize, Error> {
    let Token::Literal(text) = token else {
        return Err(Error::at(
            0,
            format!("期望十进制无符号整数字面量，得到 {}", token.describe()),
        ));
    };
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Error::at(
            0,
            format!("期望十进制无符号整数字面量（不要 `_`、小数点或后缀），得到 `{text}`"),
        ));
    }
    text.parse()
        .map_err(|_| Error::at(0, format!("`{text}` 超出 `usize` 范围")))
}

/// 注解参数的最小游标：只认标识符、`=`、`,` 与括号组。
struct Cursor<'a> {
    tokens: &'a [Token],
    index: usize,
}

impl<'a> Cursor<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, index: 0 }
    }

    fn is_empty(&self) -> bool {
        self.index >= self.tokens.len()
    }

    fn position(&self) -> Position {
        self.index
    }

    fn next(&mut self, what: &str) -> Result<&'a Token, Error> {
        let position = self.index;
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| Error::at(position, format!("缺少{what}")))?;
        self.index += 1;
        Ok(token)
    }

    fn ident(&mut self, what: &str) -> Result<String, Error> {
        let position = self.index;
        match self.next(what)? {
            Token::Ident(ident) => Ok(ident.clone()),
            other => Err(Error::at(
                position,
                format!("期望{what}（标识符），得到 {}", other.describe()),
            )),
        }
    }

    fn punct(&mut self, expected: char, what: &str) -> Result<(), Error> {
        let position = self.index;
        match self.next(what)? {
            Token::Punct(punct) if *punct == expected => Ok(()),
            other => Err(Error::at(
                position,
                format!("期望 `{expected}`（{what}），得到 {}", other.describe()),
            )),
        }
    }

    fn group(&mut self, expected: Bracket, what: &str) -> Result<(Position, &'a [Token]), Error> {
        let position = self.index;
        match self.next(what)? {
            Token::Group(bracket, items) if *bracket == expected => Ok((position, items)),
            other => Err(Error::at(
                position,
                format!("期望{what}，得到 {}", other.describe()),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::lex;

    fn parse_text(text: &str) -> Result<Declaration, Error> {
        parse(&lex(text))
    }

    #[test]
    fn a_full_annotation_parses_in_any_order() {
        let declaration =
            parse_text(r#"out = [4096], name = "lm_head", ops = ["gemv_i4"]"#).expect("合法注解");
        assert_eq!(declaration.name(), "lm_head");
        assert_eq!(declaration.out(), [4096]);
        assert_eq!(declaration.ops(), ["gemv_i4"]);
    }

    #[test]
    fn a_trailing_comma_and_empty_shape_are_accepted() {
        let declaration = parse_text(r#"name = "loss", out = [], ops = ["dot"],"#)
            .expect("尾随逗号与标量输出都合法");
        assert!(declaration.out().is_empty(), "空形状即标量输出");
        assert_eq!(declaration.ops(), ["dot"]);
    }

    #[test]
    fn a_rank_two_shape_keeps_its_axis_order() {
        let declaration =
            parse_text(r#"name = "q", out = [2, 3], ops = ["gemv_i4"]"#).expect("合法注解");
        assert_eq!(declaration.out(), [2, 3], "轴序不能被重排");
    }

    #[test]
    fn an_input_width_is_optional_and_kept_as_given() {
        let declaration =
            parse_text(r#"name = "q", out = [8], in = 32, ops = ["gemv_i4"]"#).expect("合法注解");
        assert_eq!(declaration.input(), Some(32));
        assert_eq!(
            parse_text(r#"name = "q", out = [8], ops = ["gemv_i4"]"#)
                .expect("合法注解")
                .input(),
            None,
            "没有 `in` 只是没有可计价的形状，不是文法错误"
        );
    }

    #[test]
    fn an_input_width_only_fits_a_single_axis_output() {
        let error = parse_text(r#"name = "q", out = [2, 3], in = 32, ops = ["gemv_i4"]"#)
            .expect_err("多轴输出没有单一的输出轴");
        assert!(error.message().contains("单轴输出"), "{error}");
    }

    #[test]
    fn a_zero_input_width_is_rejected() {
        let error = parse_text(r#"name = "q", out = [8], in = 0, ops = ["gemv_i4"]"#)
            .expect_err("零宽规约轴");
        assert!(error.message().contains("大于 0"), "{error}");
    }

    #[test]
    fn an_unknown_key_lists_the_closed_set() {
        let error = parse_text(r#"name = "x", out = [1], ops = ["dot"], dtype = "f32""#)
            .expect_err("未知键");
        assert!(error.message().contains("dtype"), "{error}");
        for key in KEYS {
            assert!(error.message().contains(key), "{error} 应列出 {key}");
        }
    }

    #[test]
    fn a_repeated_key_is_rejected_for_every_key() {
        for text in [
            r#"name = "a", name = "b", out = [1], ops = ["dot"]"#,
            r#"name = "a", out = [1], out = [2], ops = ["dot"]"#,
            r#"name = "a", out = [1], in = 4, in = 8, ops = ["dot"]"#,
            r#"name = "a", out = [1], ops = ["dot"], ops = ["dot"]"#,
        ] {
            let error = parse_text(text).expect_err("重复键");
            assert!(error.message().contains("重复"), "{text}: {error}");
        }
    }

    #[test]
    fn a_missing_key_says_which_one() {
        let cases = [
            (r#"out = [1], ops = ["dot"]"#, "name"),
            (r#"name = "a", ops = ["dot"]"#, "out"),
            (r#"name = "a", out = [1]"#, "ops"),
        ];
        for (text, missing) in cases {
            let error = parse_text(text).expect_err("缺键");
            assert!(error.message().contains(missing), "{text}: {error}");
        }
    }

    #[test]
    fn a_non_string_name_is_rejected() {
        let error = parse_text(r#"name = 42, out = [1], ops = ["dot"]"#).expect_err("非字符串");
        assert!(error.message().contains("字符串字面量"), "{error}");
    }

    #[test]
    fn an_escaped_name_is_rejected_because_it_would_stop_being_literal() {
        let tokens = vec![
            Token::Ident("name".to_string()),
            Token::Punct('='),
            Token::Literal(r#""a\nb""#.to_string()),
            Token::Punct(','),
            Token::Ident("out".to_string()),
            Token::Punct('='),
            Token::Group(Bracket::Square, vec![Token::Literal("1".to_string())]),
            Token::Punct(','),
            Token::Ident("ops".to_string()),
            Token::Punct('='),
            Token::Group(
                Bracket::Square,
                vec![Token::Literal(r#""dot""#.to_string())],
            ),
        ];
        let error = parse(&tokens).expect_err("转义");
        assert!(error.message().contains("转义"), "{error}");
    }

    #[test]
    fn a_hex_dimension_is_not_a_decimal() {
        let tokens = vec![
            Token::Ident("name".to_string()),
            Token::Punct('='),
            Token::Literal(r#""x""#.to_string()),
            Token::Punct(','),
            Token::Ident("out".to_string()),
            Token::Punct('='),
            Token::Group(Bracket::Square, vec![Token::Literal("0x10".to_string())]),
            Token::Punct(','),
            Token::Ident("ops".to_string()),
            Token::Punct('='),
            Token::Group(
                Bracket::Square,
                vec![Token::Literal(r#""dot""#.to_string())],
            ),
        ];
        let error = parse(&tokens).expect_err("十六进制");
        assert!(error.message().contains("十进制"), "{error}");
    }

    #[test]
    fn a_zero_width_axis_is_rejected() {
        let error = parse_text(r#"name = "x", out = [0], ops = ["dot"]"#).expect_err("零宽轴");
        assert!(error.message().contains("大于 0"), "{error}");
    }

    #[test]
    fn an_empty_operator_list_is_rejected() {
        let error = parse_text(r#"name = "x", out = [1], ops = []"#).expect_err("空算子集");
        assert!(error.message().contains("至少要引用一个"), "{error}");
    }

    #[test]
    fn a_duplicated_operator_is_rejected() {
        let error =
            parse_text(r#"name = "x", out = [1], ops = ["dot", "dot"]"#).expect_err("重复算子");
        assert!(error.message().contains("出现了两次"), "{error}");
    }

    #[test]
    fn a_name_with_whitespace_is_rejected() {
        let error = parse_text(r#"name = "a b", out = [1], ops = ["dot"]"#).expect_err("空白");
        assert!(error.message().contains("空白"), "{error}");
        let error = parse_text(r#"name = "", out = [1], ops = ["dot"]"#).expect_err("空名字");
        assert!(error.message().contains("空字符串"), "{error}");
    }

    #[test]
    fn a_missing_separator_is_reported_at_the_offending_token() {
        let error = parse_text(r#"name = "a" out = [1], ops = ["dot"]"#).expect_err("缺逗号");
        assert_eq!(error.position(), 3, "缺的那个逗号在下标 3 处");
        assert!(error.message().contains("`,`"), "{error}");
    }

    #[test]
    fn an_error_points_at_the_token_the_user_wrote() {
        let tokens = lex(r#"name = "a", out = [1], ops = ["dot"], bogus = "x""#);
        let error = parse(&tokens).expect_err("未知键");
        assert_eq!(error.position(), 12, "`bogus` 是自己那一处，不是注解开头");
        assert_eq!(
            &tokens[error.position()],
            &Token::Ident("bogus".to_string())
        );
    }

    #[test]
    fn a_group_error_points_at_the_bracket_group() {
        let tokens = lex(r#"name = "a", out = [1], ops = ["dot", 7]"#);
        let error = parse(&tokens).expect_err("非字符串算子");
        assert_eq!(error.position(), 10, "错误指向 `ops` 的方括号组");
        assert!(matches!(tokens[error.position()], Token::Group(..)));
    }

    #[test]
    fn structures_and_enums_are_both_acceptable_targets() {
        assert_eq!(
            type_name(&lex("pub struct OutputHead;")).expect("struct"),
            "OutputHead"
        );
        assert_eq!(
            type_name(&lex("pub(crate) enum Head { A }")).expect("enum"),
            "Head"
        );
    }

    #[test]
    fn behaviour_is_not_a_layer() {
        for (text, keyword) in [
            ("fn f() {}", "fn"),
            ("trait T {}", "trait"),
            ("impl T {}", "impl"),
            ("mod m {}", "mod"),
            ("union U { a: u8 }", "union"),
            ("type A = u8;", "type"),
        ] {
            let error = type_name(&lex(text)).expect_err(keyword);
            assert!(error.message().contains(keyword), "{text}: {error}");
        }
    }

    #[test]
    fn a_generic_target_is_rejected_with_the_reason() {
        let error = type_name(&lex("struct Head<T> { inner: T }")).expect_err("泛型");
        assert!(error.message().contains("泛型"), "{error}");
        assert_eq!(error.position(), 1, "错误指向类型名");
    }

    #[test]
    fn an_empty_item_is_rejected_rather_than_guessed() {
        let error = type_name(&[]).expect_err("空项");
        assert!(error.message().contains("找不到类型定义"), "{error}");
    }
}
