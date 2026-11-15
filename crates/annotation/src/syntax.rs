//! 注解文法：与 `proc_macro` 解耦的纯解析层。
//!
//! 文法是一句话：`键 = 值 (, 键 = 值)*`，值只接受字符串字面量、十进制整数或方括号数组，键顺序随意、
//! 尾随逗号允许、每个键只能出现一次。两道注解共用同一台读键机：一个键就是 [`KeySpec`] 表里的一行
//! ——名字、取值文法、重复与缺失时的提示，于是"加一个键"是加一行。
//!
//! 分成这一层是为了可测性：本模块不碰 `proc_macro`，`cargo test` 因此能逐条覆盖全部错误路径。

/// token 在输入里的下标；错误用它指回用户写的那一处，再由上层换成 span。
pub(crate) type Position = usize;

/// 括号种类：不直接用 `proc_macro::Delimiter`；`Invisible` 是那种不显示的分组。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bracket {
    Paren,
    Square,
    Brace,
    Invisible,
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

impl Bracket {
    /// 错误信息里怎么读一个分组：括号种类各自的写法，`Invisible` 没有可读的括号。
    fn describe(self) -> String {
        match self {
            Self::Paren => "`(..)`".to_string(),
            Self::Square => "`[..]`".to_string(),
            Self::Brace => "`{..}`".to_string(),
            Self::Invisible => "不可见分组".to_string(),
        }
    }
}

impl Token {
    /// 错误信息里怎么读它。
    fn describe(&self) -> String {
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
    fn at(position: Position, message: impl Into<String>) -> Self {
        Self {
            position,
            message: message.into(),
        }
    }

    /// 把组内（或字面量内）的错误指回它所属的那个 token。
    fn rebased(self, position: Position) -> Self {
        Self { position, ..self }
    }
}

/// 注解解析后的四件事实：名字、输出形状、输入宽度、算子列表。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Declaration {
    /// 层名（与权重条目前缀一致）。
    pub(crate) name: String,
    /// 输出形状；空表示标量输出。
    pub(crate) out: Vec<usize>,
    /// 输入（规约轴）宽度；`None` 表示这一层没有可计价的形状。
    pub(crate) input: Option<usize>,
    /// 要引用的算子名，已去重且非空。
    pub(crate) ops: Vec<String>,
}

/// 注解允许的键；顺序即错误信息里的提示顺序。
pub(crate) const KEYS: [&str; 4] = ["name", "out", "in", "ops"];

/// 一个键解析出的值；`Empty` 只表示可选键没写。
#[derive(Clone, PartialEq, Eq)]
enum Value {
    Empty,
    Text(String),
    Width(usize, Position),
    Axes(Vec<usize>, Position),
    Names(Vec<String>),
}

impl Value {
    /// 名字或闭集字面量：取值文法保证类型，落空即空串。
    fn text(&self) -> &str {
        match self {
            Self::Text(text) => text,
            _ => "",
        }
    }

    /// 名字数组；`ops` 之外的键没有名字列表，落空即空切片。
    fn names(&self) -> &[String] {
        match self {
            Self::Names(names) => names,
            _ => &[],
        }
    }

    /// 方括号数组的内容与方括号的位置。
    fn axes(&self) -> (&[usize], Position) {
        match self {
            Self::Axes(axes, at) => (axes, *at),
            _ => (&[], 0),
        }
    }
}

/// 一个键的取值文法：`(游标, 键的位置, 这一行) -> 值`。
type Grammar = fn(&mut Cursor<'_>, Position, &KeySpec<'_>) -> Result<Value, Error>;

/// 一个注解键的全部文法就是表里的一行：名字、取值文法、重复与缺失时的提示。
#[derive(Clone, Copy)]
struct KeySpec<'a> {
    /// 键名；也是「未知键」提示里列出的顺序。
    name: &'a str,
    grammar: Grammar,
    /// 重复时写在「注解键 `x` 重复：」之后的半句。
    dup: &'static str,
    /// 缺少必填键时接在键名后面的写法（` = \"..\"`）；`None` 表示可选。
    form: Option<&'static str>,
}

/// 表项构造：与 [`KeySpec`] 的字段一一对应，于是每条键占表里的一行。
const fn row<'a>(
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

/// `#[spec(..)]` 的键表，顺序即「未知键」提示里的顺序。
const LAYER: [KeySpec<'static>; 4] = [
    row(KEYS[0], name_value, "名字只能声明一次", Some(" = \"..\"")),
    row(KEYS[1], out_value, "输出形状只能声明一次", Some(" = [..]")),
    row(KEYS[2], in_value, "输入宽度只能声明一次", None),
    row(
        KEYS[3],
        ops_value,
        "算子列表只能声明一次",
        Some(" = [\"..\"]"),
    ),
];

/// 解析注解本体；任何一处不合文法都只产出一条带位置的错误。
pub(crate) fn parse(tokens: &[Token]) -> Result<Declaration, Error> {
    let values = collect(tokens, &LAYER)?;
    require(&values, &LAYER)?;
    let ops = values[3].names().to_vec();
    if ops.is_empty() {
        return Err(Error::at(
            tokens.len().saturating_sub(1),
            "`ops` 不能为空：一个层至少要引用一个已声明的算子",
        ));
    }
    Ok(Declaration {
        name: values[0].text().to_string(),
        out: values[1].axes().0.to_vec(),
        input: input_width(&values)?,
        ops,
    })
}

/// `in` 只对单轴输出（一次投影）有意义：多轴输出把某一根轴冒充 `n`，只会给出假下界。
fn input_width(values: &[Value]) -> Result<Option<usize>, Error> {
    let out = values[1].axes().0;
    let Value::Width(width, at) = &values[2] else {
        return Ok(None);
    };
    if out.len() == 1 {
        return Ok(Some(*width));
    }
    Err(Error::at(
        *at,
        format!(
            "`in = {width}` 只对单轴输出（一次投影）有意义：`out` 现在有 {} 个轴",
            out.len()
        ),
    ))
}

/// 读完一条注解的全部 `键 = 值`；一次只报一条错误，顺序是「键 → 重复 → 值」。
fn collect<'a>(tokens: &[Token], table: &'a [KeySpec<'a>]) -> Result<Vec<Value>, Error> {
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
    let (index, at) = key_index(cursor, table)?;
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

/// 读 `键 =`：返回它在表里的下标与键的位置；不在闭集里就点名并列出可用键。
fn key_index<'a>(
    cursor: &mut Cursor<'_>,
    table: &[KeySpec<'a>],
) -> Result<(usize, Position), Error> {
    let at = cursor.index;
    let key = key_name(cursor, at)?;
    cursor.punct('=', "键与值之间的 `=`")?;
    let found = table.iter().position(|spec| spec.name == key);
    let index = found.ok_or_else(|| unknown_key(table, key, at))?;
    Ok((index, at))
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
fn require(values: &[Value], table: &[KeySpec<'_>]) -> Result<(), Error> {
    for (index, spec) in table.iter().enumerate() {
        let (true, Some(form)) = (matches!(values[index], Value::Empty), spec.form) else {
            continue;
        };
        return Err(Error::at(0, format!("缺少必填键 `{}{form}`", spec.name)));
    }
    Ok(())
}

/// 字符串字面量键值；字面量自身的错误指回键的位置。
fn text_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<String, Error> {
    let token = cursor.next(&format!("`{}` 的值", spec.name))?;
    string(token).map_err(|error| error.rebased(at))
}

/// 十进制无符号整数键值。
fn number_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<usize, Error> {
    let token = cursor.next(&format!("`{}` 的值", spec.name))?;
    unsigned(token).map_err(|error| error.rebased(at))
}

/// 名字类键值：`name` / `ext` / `tag` 共用。
fn name_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let text = text_value(cursor, at, spec)?;
    checked_name(&text, at)?;
    Ok(Value::Text(text))
}

/// 非零轴数组：括号与元素的措辞由键名推出，零宽轴那一句由调用方给。
fn axes(cursor: &mut Cursor<'_>, spec: &KeySpec<'_>, zero: &str) -> Result<Value, Error> {
    let group_what = format!("`{} = [..]` 的方括号", spec.name);
    let (group, array) = cursor.group(Bracket::Square, &group_what)?;
    let mut axes: Vec<usize> = Vec::with_capacity(array.len());
    for item in elements(group, array, &format!("`{}` 的元素", spec.name))? {
        axes.push(axis_value(item, group, zero)?);
    }
    Ok(Value::Axes(axes, group))
}

/// 一个轴：十进制无符号整数，且大于 0——零宽轴不是一个可计算的形状。
fn axis_value(item: &Token, group: Position, zero: &str) -> Result<usize, Error> {
    let axis = unsigned(item).map_err(|error| error.rebased(group))?;
    if axis == 0 {
        return Err(Error::at(group, zero));
    }
    Ok(axis)
}

/// `out = [..]`：输出形状的轴，轴宽随意但不能是 0。
fn out_value(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    axes(
        cursor,
        spec,
        "输出形状的每个轴都必须大于 0：零宽轴不是一个可计算的形状",
    )
}

/// `shape = [..]`：生产形状，两个轴由 [`two_axes`] 再核一次。
fn shape_value(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    axes(
        cursor,
        spec,
        "生产形状的每个轴都必须大于 0：零宽形状不是一个可计算的形状",
    )
}

/// `ops = [".."]`：名字数组，逐项校验、去重；空表由 [`parse`] 单独拦。
fn ops_value(cursor: &mut Cursor<'_>, _: Position, _: &KeySpec<'_>) -> Result<Value, Error> {
    let (group, array) = cursor.group(Bracket::Square, "`ops = [\"..\"]` 的方括号")?;
    let mut names: Vec<String> = Vec::with_capacity(array.len());
    for item in elements(group, array, "`ops` 的元素")? {
        push_name(&mut names, item, group)?;
    }
    Ok(Value::Names(names))
}

/// 一个算子名：字符串字面量 + [`checked_name`]；重复的那一次不会生效，所以在这里就拒。
fn push_name(names: &mut Vec<String>, item: &Token, group: Position) -> Result<(), Error> {
    let text = string(item).map_err(|error| error.rebased(group))?;
    checked_name(&text, group)?;
    if names.contains(&text) {
        return Err(Error::at(
            group,
            format!("算子 `{text}` 在同一个层里出现了两次：多写的这一次不会生效"),
        ));
    }
    names.push(text);
    Ok(())
}

/// `in = ..`：大于 0 的规约轴宽度，位置留着给 [`input_width`] 报跨键的错。
fn in_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let width = number_value(cursor, at, spec)?;
    if width == 0 {
        return Err(Error::at(at, "输入宽度必须大于 0：零宽的规约轴没有下界"));
    }
    Ok(Value::Width(width, at))
}

/// `dtype = ".."`：对拍有阈值的输出类型闭集。
fn dtype_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let text = text_value(cursor, at, spec)?;
    closed(&text, at, spec, "输出类型", &DTYPES)?;
    Ok(Value::Text(text))
}

/// `skeleton = ".."`：骨架闭集。
fn skeleton_value(
    cursor: &mut Cursor<'_>,
    at: Position,
    spec: &KeySpec<'_>,
) -> Result<Value, Error> {
    let text = text_value(cursor, at, spec)?;
    closed(&text, at, spec, "骨架", &SKELETONS)?;
    Ok(Value::Text(text))
}

/// 闭集校验：`what` 是「语言 / 输出类型 / 骨架」，提示里直接列出可用名字。
fn closed(
    text: &str,
    at: Position,
    spec: &KeySpec<'_>,
    what: &str,
    set: &[&str],
) -> Result<(), Error> {
    if set.contains(&text) {
        return Ok(());
    }
    Err(Error::at(
        at,
        format!(
            "`{} = \"{text}\"` 不在{what}闭集内（可用：{}）",
            spec.name,
            set.join("、")
        ),
    ))
}

/// `skeleton` 闭集：决定生成的 `MODEL` 走哪副算子骨架。
///
/// 每个成员在 `codegen` 的骨架表里都有**自己的一行**（`every_skeleton_has_a_branch` 逐项渲染核对）；
/// `gemm` 特殊：模块名用注册名，名字必须先在 [`GEMM_SKELETON_NAMES`] 里。缺省值是 `gemv`：早期版本把
/// 缺省写成 `gemm`，而 `gemm` 又靠「名字不是 `gemm_*` 就落到 gemv」收尾——声明与行为各说一套。
pub(crate) const SKELETONS: [&str; 12] = [
    "gemv",
    "gemm",
    "elementwise",
    "unary",
    "reduce",
    "rmsnorm",
    "softmax",
    "rope",
    "embed",
    "dequant",
    "dequant_i4",
    "attention",
];

/// `gemm` 骨架认识的注册名：三种格式各有 `op::gemm_*` 模块，名字就是模块选择器。
pub(crate) const GEMM_SKELETON_NAMES: [&str; 3] = ["gemm_f32", "gemm_i8", "gemm_i4"];

/// `#[model(..)]` 的 `dtype` 闭集：对拍**有阈值**的那几种输出类型。
///
/// 注解只允许写出**输出**类型（对拍读回的宽度由它定），闭集之外的名字在这里失败，而不是生成一个不
/// 存在的 `DType` 变体。判据不是"手挑三个"：它与 `cutile::reference::tolerance` 给出非零阈值的
/// `DType` 集合**逐项相等**（`tests/contracts.rs`），于是新增一种可对拍的浮点输出时，失败的是"注解
/// 写不出它"，而不是它悄悄地无法被声明。
pub(crate) const DTYPES: [&str; 3] = ["f16", "bf16", "f32"];

/// `#[model(..)]` 解析后的四件事：注册名、生产形状、输出类型、骨架，都是编译期常量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelDecl {
    /// 注册表的键（与 `ops/<name>.toml` 同名）。
    pub(crate) name: String,
    /// 生产形状 `(m, n)`。
    pub(crate) shape: [usize; 2],
    /// 输出类型名（[`DTYPES`] 里的一个）。
    pub(crate) dtype: String,
    /// 骨架名（[`SKELETONS`] 里的一个），缺省为 `gemv`。
    pub(crate) skeleton: String,
}

/// `#[model(..)]` 的键表：注册名、生产形状、输出类型、骨架。
const MODEL: [KeySpec<'static>; 4] = [
    row(
        "name",
        name_value,
        "一条语义只有一个注册名",
        Some(" = \"..\""),
    ),
    row(
        "shape",
        shape_value,
        "生产形状只能声明一次",
        Some(" = [m, n]"),
    ),
    row(
        "dtype",
        dtype_value,
        "输出类型只能声明一次",
        Some(" = \"..\""),
    ),
    row("skeleton", skeleton_value, "一条语义只有一个骨架", None),
];

/// 解析 `#[model(name = "..", shape = [..], dtype = "..")]`。三个键都必填，`shape` 恰好两个轴。
pub(crate) fn parse_model(tokens: &[Token]) -> Result<ModelDecl, Error> {
    let values = collect(tokens, &MODEL)?;
    require(&values, &MODEL)?;
    let name = values[0].text();
    let skeleton = chosen_skeleton(&values[3]);
    check_gemm_name(skeleton, name)?;
    Ok(ModelDecl {
        name: name.to_string(),
        shape: two_axes(&values[1])?,
        dtype: values[2].text().to_string(),
        skeleton: skeleton.to_string(),
    })
}

/// 骨架名的缺省值：没写 `skeleton` 时用 `gemv`——生成器实际走的那条分支。
fn chosen_skeleton(value: &Value) -> &str {
    match value.text() {
        "" => "gemv",
        name => name,
    }
}

/// `shape` 恰好两个轴：`Model::shape` 就是 `(m, n)`。
fn two_axes(value: &Value) -> Result<[usize; 2], Error> {
    let (axes, at) = value.axes();
    let [m, n] = axes else {
        return Err(Error::at(
            at,
            format!(
                "生产形状必须写成两个轴 `[m, n]`，得到 {} 个：`Model::shape` 是 `(m, n)`",
                axes.len()
            ),
        ));
    };
    Ok([*m, *n])
}

/// `gemm` 骨架按注册名选模块：名字不在 [`GEMM_SKELETON_NAMES`] 里就没有骨架可用（否则会生成一份
/// 编译得过、却绑错格式的 `MODEL`）。
fn check_gemm_name(skeleton: &str, name: &str) -> Result<(), Error> {
    if skeleton != "gemm" || GEMM_SKELETON_NAMES.contains(&name) {
        return Ok(());
    }
    Err(Error::at(
        0,
        format!(
            "`skeleton = \"gemm\"` 只适用于 {} 这几种注册名（gemm 骨架按名字选模块）；`{name}` 请显式写出它实际用的骨架",
            GEMM_SKELETON_NAMES.join("、")
        ),
    ))
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

/// `#[model]` 标注的是**一个具体格式的别名**：`type 名字 = <具体实例>;`。`MODEL` 的入口要写成
/// `<名字 as Format>`，名字必须能被当作类型用，而泛型别名不能。
pub(crate) fn model_target(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    let Some((position, word)) = keyword(tokens) else {
        return Err(Error::at(
            0,
            format!("`#[{attribute}]` 只能标注 `type` 别名：找不到类型定义"),
        ));
    };
    if word != "type" {
        return Err(Error::at(
            position,
            format!(
                "`#[{attribute}]` 只能标注 `type` 别名，得到 `{word}`：语义表绑的是\"哪份格式契约\"，不是行为"
            ),
        ));
    }
    alias(tokens, position, attribute)
}

/// `type 名字 = ..;`：别名后必须是 `=`，带 `<..>` 的泛型别名不能当类型用。
fn alias(tokens: &[Token], position: Position, attribute: &str) -> Result<String, Error> {
    let Some(Token::Ident(name)) = tokens.get(position + 1) else {
        return Err(Error::at(position, "`type` 后应当紧跟别名"));
    };
    match tokens.get(position + 2) {
        Some(Token::Punct('=')) => Ok(name.clone()),
        Some(Token::Punct('<')) => Err(Error::at(
            position + 1,
            format!(
                "`#[{attribute}]` 不支持泛型别名：`{name}` 请先固定成一个具体实例（例如 `Quantized<32>`）"
            ),
        )),
        _ => Err(Error::at(
            position + 1,
            format!("`{name}` 必须写成 `type {name} = <具体格式>;`：语义表要绑到那个类型上"),
        )),
    }
}

/// 与 [`model_target`] 同一判据，只是错误里写的注解名不同：两道注解标的是同一类东西——"一份配置
/// + 一个具体类型"，判据只有一份，注解名作为参数传进来只是为了让文案指向用户真正写下的那一个。
pub(crate) fn target_name(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    let Some((position, word)) = keyword(tokens) else {
        return Err(Error::at(
            0,
            format!("`#[{attribute}]` 只能标注 struct / enum：找不到类型定义"),
        ));
    };
    match word {
        "struct" | "enum" => named(tokens, position, word, attribute),
        _ => Err(Error::at(
            position,
            format!(
                "`#[{attribute}]` 只能标注 struct / enum，得到 `{word}`：注解标的是类型，不是行为"
            ),
        )),
    }
}

/// `struct` / `enum` 后的类型名；泛型参数在此拒绝（形状必须在编译期固定）。
fn named(
    tokens: &[Token],
    position: Position,
    word: &str,
    attribute: &str,
) -> Result<String, Error> {
    let Some(Token::Ident(name)) = tokens.get(position + 1) else {
        return Err(Error::at(position, format!("`{word}` 后应当紧跟类型名")));
    };
    if matches!(tokens.get(position + 2), Some(Token::Punct('<'))) {
        return Err(Error::at(
            position + 1,
            format!("`#[{attribute}]` 不支持泛型参数：`{name}` 请改成形状固定的具体类型"),
        ));
    }
    Ok(name.clone())
}

/// 数组元素：`元素 (, 元素)*`，允许尾随逗号；任何错误都指回方括号组。
fn elements<'a>(group: Position, tokens: &'a [Token], what: &str) -> Result<Vec<&'a Token>, Error> {
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

/// 名字（层名、算子名、语言与缓存标签）的公共约束：非空、不含空白与控制字符。错误指在 `at` 上：
/// 名字本身没有独立 span，能指的最具体位置就是写下它的那个键。
fn checked_name(name: &str, at: Position) -> Result<(), Error> {
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

/// 注解参数的最小游标：只认标识符、`=`、`,` 与括号组；`index` 同时是「错误指在这里」的位置。
struct Cursor<'a> {
    tokens: &'a [Token],
    index: usize,
}

impl<'a> Cursor<'a> {
    fn is_empty(&self) -> bool {
        self.index >= self.tokens.len()
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
#[path = "../tests/syntax_tests.rs"]
mod tests;
