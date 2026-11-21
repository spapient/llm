//! 文法引擎：token 模型、游标、表驱动的读键机、字面量原子与被标注项判据。
//!
//! 不碰 `proc_macro`，因此每条错误路径都能在 `cargo test` 里跑到。三种声明形态（层 / 语义表 /
//! 运行时）共用这一台读键机：一个键就是 [`KeySpec`] 表里的一行，于是"加一个键"是加一行。
//!
//! 本模块只回答两件事：**注解本体怎么读**（`键 = 值 (, 键 = 值)*`）与**被标注项能不能标**
//! （只有具体类型/别名，不含泛型与行为）。声明解析后的产物与键表在 `decl`，渲染在 `render`。

/// token 在输入里的下标；错误用它指回用户写的那一处，再由适配层换成 span。
pub(crate) type Position = usize;

/// 括号种类：不直接用 `proc_macro::Delimiter`；`Invisible` 是那种不显示的分组。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bracket {
    Paren,
    Square,
    Brace,
    Invisible,
}

impl Bracket {
    /// 错误信息里怎么读一个分组：`Invisible` 没有可读的括号。
    pub(crate) fn describe(self) -> String {
        match self {
            Self::Paren => "`(..)`".to_string(),
            Self::Square => "`[..]`".to_string(),
            Self::Brace => "`{..}`".to_string(),
            Self::Invisible => "不可见分组".to_string(),
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
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Ident(word) => format!("`{word}`"),
            Self::Punct(punct) => format!("`{punct}`"),
            Self::Literal(text) => format!("字面量 `{text}`"),
            Self::Group(bracket, _) => bracket.describe(),
        }
    }
}

/// 文法错误：出错 token 的下标 + 一句可执行的提示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Error {
    pub(crate) position: Position,
    pub(crate) message: String,
}

impl Error {
    pub(crate) fn at(position: Position, message: impl Into<String>) -> Self {
        Self {
            position,
            message: message.into(),
        }
    }

    /// 把组内（或字面量内）的错误指回它所属的那个 token。
    pub(crate) fn rebased(self, position: Position) -> Self {
        Self { position, ..self }
    }
}

/// 一个运行时洞：`$参数` 的参数名 + `$` 在注解里的位置（跨键冲突要指到写下它的那一处）。
pub(crate) type Hole = (String, Position);

/// 一个键解析出的值；`Empty` 只表示可选键没写。
///
/// 运行时形态把单个洞也写成 [`Value::Holes`]（长度 1），于是两种形态共用同一组槽位。
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Value {
    Empty,
    Text(String),
    Width(usize, Position),
    Axes(Vec<usize>, Position),
    Names(Vec<String>),
    Holes(Vec<Hole>, Position),
}

impl Value {
    /// 名字或闭集字面量：取值文法保证类型，落空即空串。
    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Text(text) => text,
            _ => "",
        }
    }

    /// 名字数组；`ops` 之外的静态键没有名字列表，落空即空切片。
    pub(crate) fn names(&self) -> &[String] {
        match self {
            Self::Names(names) => names,
            _ => &[],
        }
    }

    /// 方括号数组的内容与方括号的位置。
    pub(crate) fn axes(&self) -> (&[usize], Position) {
        match self {
            Self::Axes(axes, at) => (axes, *at),
            _ => (&[], 0),
        }
    }

    /// 运行时洞列表与它的位置；其余变体回落到空表。
    pub(crate) fn holes(&self) -> (&[Hole], Position) {
        match self {
            Self::Holes(holes, at) => (holes, *at),
            _ => (&[], 0),
        }
    }

    /// 第一个运行时洞的参数名；其余变体（或空表）回落到空串。
    pub(crate) fn hole(&self) -> &str {
        self.holes().0.first().map_or("", |(name, _)| name.as_str())
    }

    /// 运行时洞的参数名，顺序即写法顺序。
    pub(crate) fn hole_names(&self) -> impl Iterator<Item = &str> {
        self.holes().0.iter().map(|(name, _)| name.as_str())
    }
}

/// 一个键的取值文法：`(游标, 键的位置, 这一行) -> 值`。
pub(crate) type Grammar = fn(&mut Cursor<'_>, Position, &KeySpec<'_>) -> Result<Value, Error>;

/// 一个注解键的全部文法就是表里的一行：名字、取值文法、重复与缺失时的提示。
#[derive(Clone, Copy)]
pub(crate) struct KeySpec<'a> {
    /// 键名；也是「未知键」提示里列出的顺序。
    pub(crate) name: &'a str,
    pub(crate) grammar: Grammar,
    /// 重复时写在「注解键 `x` 重复：」之后的半句。
    dup: &'static str,
    /// 缺少必填键时接在键名后面的写法（` = ".."`）；`None` 表示可选。
    form: Option<&'static str>,
}

/// 表项构造：与 [`KeySpec`] 的字段一一对应，于是每条键占表里的一行。
pub(crate) const fn row<'a>(
    name: &'a str,
    grammar: Grammar,
    dup: &'static str,
    form: Option<&'static str>,
) -> KeySpec<'a> {
    KeySpec {
        name,
        grammar,
        dup,
        form,
    }
}

/// 读完一条注解的全部 `键 = 值`；一次只报一条错误，顺序是「键 → 重复 → 值」。
pub(crate) fn collect<'a>(tokens: &[Token], table: &'a [KeySpec<'a>]) -> Result<Vec<Value>, Error> {
    let mut cursor = Cursor { tokens, index: 0 };
    let mut values = vec![Value::Empty; table.len()];
    while !cursor.is_empty() {
        read_one(&mut cursor, table, &mut values)?;
    }
    Ok(values)
}

/// 读一对 `键 = 值` 并放回槽位：认键、查重、读值、吃分隔符。
fn read_one<'a>(
    cursor: &mut Cursor<'_>,
    table: &'a [KeySpec<'a>],
    values: &mut [Value],
) -> Result<(), Error> {
    let at = cursor.index;
    let key = key_name(cursor, at)?;
    cursor.punct('=', "键与值之间的 `=`")?;
    let index = table
        .iter()
        .position(|spec| spec.name == key)
        .ok_or_else(|| unknown_key(table, key, at))?;
    if !matches!(values[index], Value::Empty) {
        return Err(Error::at(
            at,
            format!("注解键 `{}` 重复：{}", table[index].name, table[index].dup),
        ));
    }
    values[index] = (table[index].grammar)(cursor, at, &table[index])?;
    if cursor.is_empty() {
        return Ok(());
    }
    cursor.punct(',', "键之间的 `,`")
}

/// 读一个键名：非标识符就是一条「期望键名」的错误。
fn key_name<'a>(cursor: &mut Cursor<'a>, at: Position) -> Result<&'a str, Error> {
    match cursor.next("键名")? {
        Token::Ident(key) => Ok(key.as_str()),
        other => Err(Error::at(
            at,
            format!("期望键名（标识符），得到 {}", other.describe()),
        )),
    }
}

/// 不在闭集里的键：点名它并列出这张表的全部可用键。
fn unknown_key(table: &[KeySpec<'_>], key: &str, at: Position) -> Error {
    let available: Vec<&str> = table.iter().map(|spec| spec.name).collect();
    Error::at(
        at,
        format!("未知的注解键 `{key}`（可用：{}）", available.join("、")),
    )
}

/// 必填键一个都不能少；缺了就是一条点名这个键的「缺少必填键」。
pub(crate) fn require(values: &[Value], table: &[KeySpec<'_>]) -> Result<(), Error> {
    for (index, spec) in table.iter().enumerate() {
        let (true, Some(form)) = (matches!(values[index], Value::Empty), spec.form) else {
            continue;
        };
        return Err(Error::at(0, format!("缺少必填键 `{}{form}`", spec.name)));
    }
    Ok(())
}

/// 数组元素：`元素 (, 元素)*`，允许尾随逗号；任何错误都指回方括号组。
pub(crate) fn elements<'a>(
    group: Position,
    tokens: &'a [Token],
    what: &str,
) -> Result<Vec<&'a Token>, Error> {
    let mut cursor = Cursor { tokens, index: 0 };
    let mut elements = Vec::new();
    while !cursor.is_empty() {
        elements.push(cursor.next(what).map_err(|error| error.rebased(group))?);
        if cursor.is_empty() {
            break;
        }
        cursor
            .punct(',', "数组元素之间的 `,`")
            .map_err(|error| error.rebased(group))?;
    }
    Ok(elements)
}

/// 只接受普通字符串字面量：转义会让"注解里的名字"与"实际名字"不再逐字相同。
pub(crate) fn string(token: &Token) -> Result<String, Error> {
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
pub(crate) fn unsigned(token: &Token) -> Result<usize, Error> {
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

/// 名字的公共约束：非空、不含空白与控制字符；名字没有独立 span，指在写下它的键上。
pub(crate) fn checked_name(name: &str, at: Position) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::at(at, "名字不能是空字符串"));
    }
    if let Some(bad) = name
        .chars()
        .find(|character| character.is_whitespace() || character.is_control())
    {
        return Err(Error::at(
            at,
            format!("名字不能包含空白或控制字符（发现 `{bad:?}`）：`{name}`"),
        ));
    }
    Ok(())
}

/// 注解参数的最小游标：只认标识符、`=`、`,` 与括号组；`index` 同时是「错误指在这里」的位置。
pub(crate) struct Cursor<'a> {
    pub(crate) tokens: &'a [Token],
    pub(crate) index: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn is_empty(&self) -> bool {
        self.index >= self.tokens.len()
    }

    pub(crate) fn next(&mut self, what: &str) -> Result<&'a Token, Error> {
        let position = self.index;
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| Error::at(position, format!("缺少{what}")))?;
        self.index += 1;
        Ok(token)
    }

    pub(crate) fn punct(&mut self, expected: char, what: &str) -> Result<(), Error> {
        let position = self.index;
        match self.next(what)? {
            Token::Punct(punct) if *punct == expected => Ok(()),
            other => Err(Error::at(
                position,
                format!("期望 `{expected}`（{what}），得到 {}", other.describe()),
            )),
        }
    }

    pub(crate) fn group(
        &mut self,
        expected: Bracket,
        what: &str,
    ) -> Result<(Position, &'a [Token]), Error> {
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

/// 一项里的关键字闭集：注解只看这些词，`pub`、名字与宏名一律跳过。
const KEYWORDS: [&str; 10] = [
    "struct", "enum", "fn", "trait", "impl", "mod", "union", "type", "static", "const",
];

/// 一项里第一个关键字：判据只认它，位置留着指错。
fn keyword(tokens: &[Token]) -> Option<(Position, &str)> {
    tokens
        .iter()
        .enumerate()
        .find_map(|(position, token)| match token {
            Token::Ident(word) if KEYWORDS.contains(&word.as_str()) => {
                Some((position, word.as_str()))
            }
            _ => None,
        })
}

/// 一道注解能标什么：允许的关键字、提示里的说法、以及"为什么别的不行"。
pub(crate) struct Allowed {
    words: &'static [&'static str],
    /// 「只能标注 …」里的那一句。
    what: &'static str,
    /// 「，得到 `x`：」之后的理由。
    reason: &'static str,
}

/// `#[spec]` 只标具体类型：形状与算子列表在编译期固定，带泛型的类型无法用一份注解覆盖所有实例。
pub(crate) const TYPES: Allowed = Allowed {
    words: &["struct", "enum"],
    what: "struct / enum",
    reason: "注解标的是类型，不是行为",
};

/// `#[model]` 只标别名：语义表绑的是"哪份格式契约"，不是行为。
pub(crate) const ALIAS: Allowed = Allowed {
    words: &["type"],
    what: "`type` 别名",
    reason: "语义表绑的是\"哪份格式契约\"，不是行为",
};

/// 运行时形态只标命名字段 `struct`：`spec` 字段要注入到 `{ .. }` 字段体里。
pub(crate) const FIELDS: Allowed = Allowed {
    words: &["struct"],
    what: "命名字段 struct",
    reason: "字段注入需要命名字段体",
};

/// 被标注项的第一个关键字必须属于 `allowed`：判据一份，注解名与理由由调用方给。
pub(crate) fn target_with<'a>(
    tokens: &'a [Token],
    attribute: &str,
    allowed: &Allowed,
) -> Result<(Position, &'a str), Error> {
    let Some((position, word)) = keyword(tokens) else {
        return Err(Error::at(
            0,
            format!("`#[{attribute}]` 只能标注 {}：找不到类型定义", allowed.what),
        ));
    };
    if allowed.words.contains(&word) {
        return Ok((position, word));
    }
    Err(Error::at(
        position,
        format!(
            "`#[{attribute}]` 只能标注 {}，得到 `{word}`：{}",
            allowed.what, allowed.reason
        ),
    ))
}

/// 关键字后的名字；`word` 只用于「后应当紧跟…」这句提示。
pub(crate) fn declared_name(
    tokens: &[Token],
    position: Position,
    word: &str,
    noun: &str,
) -> Result<String, Error> {
    match tokens.get(position + 1) {
        Some(Token::Ident(name)) => Ok(name.clone()),
        _ => Err(Error::at(position, format!("`{word}` 后应当紧跟{noun}"))),
    }
}

/// 泛型参数被拒：形状必须在编译期固定，一份注解覆盖不了所有实例化。
pub(crate) fn generic(tokens: &[Token], position: Position, message: String) -> Result<(), Error> {
    match tokens.get(position + 2) {
        Some(Token::Punct('<')) => Err(Error::at(position + 1, message)),
        _ => Ok(()),
    }
}

/// `#[spec]` 的被标注项：一个具体类型（`struct` / `enum`）。
pub(crate) fn target_name(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    let (position, word) = target_with(tokens, attribute, &TYPES)?;
    generic(
        tokens,
        position,
        format!("`#[{attribute}]` 不支持泛型参数：请把类型改成形状固定的具体类型"),
    )?;
    declared_name(tokens, position, word, "类型名")
}

/// `#[model]` 的被标注项：`type 名字 = <具体实例>;`。`MODEL` 的入口要写成 `<名字 as Format>`，
/// 名字必须能被当作类型用，而泛型别名不能。
pub(crate) fn model_target(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    let (position, _) = target_with(tokens, attribute, &ALIAS)?;
    generic(
        tokens,
        position,
        format!("`#[{attribute}]` 不支持泛型别名：请先固定成一个具体实例（例如 `Quantized<32>`）"),
    )?;
    let name = declared_name(tokens, position, "type", "别名")?;
    if matches!(tokens.get(position + 2), Some(Token::Punct('='))) {
        return Ok(name);
    }
    Err(Error::at(
        position + 1,
        format!("`{name}` 必须写成 `type {name} = <具体格式>;`：语义表要绑到那个类型上"),
    ))
}
