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
pub(crate) const SKELETONS: [&str; 14] = [
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
    "dequant_i2",
    "attention",
    "scan",
];

/// `gemm` 骨架认识的注册名：各格式各有 `op::gemm_*` 模块，名字就是模块选择器。
pub(crate) const GEMM_SKELETON_NAMES: [&str; 4] = ["gemm_f32", "gemm_i8", "gemm_i4", "gemm_i2"];

/// `#[model(..)]` 的 `dtype` 闭集：对拍**有阈值**的那几种输出类型。
///
/// 注解只允许写出**输出**类型（对拍读回的宽度由它定），闭集之外的名字在这里失败，而不是生成一个不
/// 存在的 `DType` 变体。判据不是"手挑三个"：它与 `cutile::reference::tolerance` 给出非零阈值的
/// `DType` 集合**逐项相等**（`tests/contracts.rs`），于是新增一种可对拍的浮点输出时，失败的是"注解
/// 写不出它"，而不是它悄悄地无法被声明。
pub(crate) const DTYPES: [&str; 3] = ["f16", "bf16", "f32"];

/// `#[model(..)]` 解析后的五件事：注册名、生产形状、输出类型、骨架、生成常量名，都是编译期常量。
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
    /// 生成的 `Model` 常量名，缺省 `MODEL`（多算子同模块时用 `const = ".."` 分名）。
    pub(crate) const_name: String,
}

/// `#[model(..)]` 的键表：注册名、生产形状、输出类型、骨架、生成的常量名。
const MODEL: [KeySpec<'static>; 5] = [
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
    row("const", name_value, "生成的常量名只能声明一次", None),
];

/// 解析 `#[model(name = "..", shape = [..], dtype = "..", skeleton = "..", const = "..")]`。
/// `name` / `shape` / `dtype` 必填，`shape` 恰好两个轴，`skeleton` 缺省 `gemv`，`const` 缺省 `MODEL`。
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
        const_name: chosen_const(&values[4]),
    })
}

/// 骨架名的缺省值：没写 `skeleton` 时用 `gemv`——生成器实际走的那条分支。
fn chosen_skeleton(value: &Value) -> &str {
    match value.text() {
        "" => "gemv",
        name => name,
    }
}

/// `const` 键的缺省值：没写就生成 `MODEL`（沿用历史名字；多算子同模块时才需要分名）。
fn chosen_const(value: &Value) -> String {
    match value.text() {
        "" => "MODEL".to_string(),
        name => name.to_string(),
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
mod tests {
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
            let text =
                format!(r#"name = "{name}", shape = [4, 4], dtype = "f32", skeleton = "gemm""#);
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
            sweep_one(parse_model, &text, case);
        }
    }
}
