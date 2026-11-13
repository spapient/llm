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

/// `#[frontend(..)]` 的键闭集：语言、产物后缀、缓存标签。
pub(crate) const FRONTEND_KEYS: [&str; 3] = ["emit", "ext", "tag"];

/// `Emit` 闭集的语言名：一条注解能声明的**全部**语言。
///
/// 注解里的 `emit = ".."` 只接受这张表里的名字：它决定生成的 `Language::EMIT`，写错是
/// 文法错误而不是拼出一段永远匹配不上的代码。`ext` 随后还会在编译期与 `Emit::ext()` 比对，
/// 所以"注解说的语言"与"注解说的后缀"不可能各说各的。
///
/// 这张表不是第三份事实源，而是 `prof::Emit` 的**投影**：它必须与 `Emit::ALL` 逐项相等，
/// 且每个名字的变体名必须等于 [`crate::codegen`] 的机械结果（首字母大写，`opencl → Opencl`）。
/// 两条同时被核对：`tests/contracts.rs` 的 `annotation_names_come_from_the_vocabulary` 管
/// 全量与映射规则，`#[frontend]` 生成的测试管**每一个调用点**用的是不是同一个变体。
pub(crate) const LANGUAGES: [&str; 9] = [
    "cuda", "ptx", "hip", "opencl", "spirv", "metal", "ascendc", "c", "wgsl",
];

/// `#[frontend(..)]` 解析后的三件事：语言、产物后缀、缓存标签。
///
/// 三者都是编译期常量：语言决定协商依据与 `Scope`，后缀决定见证文件叫什么，标签决定产物
/// 缓存的命名空间。没有一件需要运行期再判断。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Frontend {
    emit: String,
    ext: String,
    tag: String,
}

impl Frontend {
    /// 语言名（[`LANGUAGES`] 里的一个）。
    pub(crate) fn emit(&self) -> &str {
        &self.emit
    }

    /// 产物后缀，必须等于该语言的 `Emit::ext()`（编译期断言）。
    pub(crate) fn ext(&self) -> &str {
        &self.ext
    }

    /// 产物缓存的标签（`witness` / `self` …）。
    pub(crate) fn tag(&self) -> &str {
        &self.tag
    }
}

/// 解析 `#[frontend(emit = "..", ext = "..", tag = "..")]`。
///
/// 三个键都是必填：缺一个就意味着某处只能靠默认值猜——语言猜错会把见证编成另一门语言，
/// 后缀猜错会让 `every_device_fragment_set_splices_the_p0_template` 找不到文件。
pub(crate) fn parse_frontend(tokens: &[Token]) -> Result<Frontend, Error> {
    let mut cursor = Cursor::new(tokens);
    let mut emit: Option<String> = None;
    let mut ext: Option<String> = None;
    let mut tag: Option<String> = None;

    while !cursor.is_empty() {
        let key_position = cursor.position();
        let key = cursor.ident("键名")?;
        cursor.punct('=', "键与值之间的 `=`")?;
        let what = format!("`{key}` 的值");
        let token = cursor.next(&what)?;
        let text = string(token).map_err(|error| error.rebased(key_position))?;
        checked_name(&text).map_err(|error| error.rebased(key_position))?;
        let slot = match key.as_str() {
            "emit" => &mut emit,
            "ext" => &mut ext,
            "tag" => &mut tag,
            other => {
                return Err(Error::at(
                    key_position,
                    format!(
                        "未知的注解键 `{other}`（可用：{}）",
                        FRONTEND_KEYS.join("、")
                    ),
                ));
            }
        };
        if slot.is_some() {
            return Err(Error::at(
                key_position,
                format!("注解键 `{key}` 重复：一个前端只有一份 `{key}`"),
            ));
        }
        if key == "emit" && !LANGUAGES.contains(&text.as_str()) {
            return Err(Error::at(
                key_position,
                format!(
                    "`emit = \"{text}\"` 不在语言闭集内（可用：{}）",
                    LANGUAGES.join("、")
                ),
            ));
        }
        *slot = Some(text);
        if cursor.is_empty() {
            break;
        }
        cursor.punct(',', "键之间的 `,`")?;
    }

    let emit = emit.ok_or_else(|| Error::at(0, "缺少必填键 `emit = \"..\"`"))?;
    let ext = ext.ok_or_else(|| Error::at(0, "缺少必填键 `ext = \"..\"`"))?;
    let tag = tag.ok_or_else(|| Error::at(0, "缺少必填键 `tag = \"..\"`"))?;
    Ok(Frontend { emit, ext, tag })
}

/// `#[model(..)]` 的键闭集：注册名、生产形状、输出类型。
pub(crate) const MODEL_KEYS: [&str; 4] = ["name", "shape", "dtype", "skeleton"];

/// `skeleton` 闭集：决定生成的 `MODEL` 走哪副算子骨架。
pub(crate) const SKELETONS: [&str; 11] = [
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

/// `#[model(..)]` 的 `dtype` 闭集：对拍**有阈值**的那几种输出类型。
///
/// 注解只允许写出**输出**类型（对拍读回的宽度由它定）。闭集之外的名字在这里失败，而不是
/// 生成一个不存在的 `DType` 变体——那样错误会指向生成的代码，而不是写错的那一个 token。
///
/// 判据不是"手挑三个"：它与 `cutile::reference::tolerance` 给出非零阈值的 `DType` 集合
/// **逐项相等**（`tests/contracts.rs` 的 `annotation_names_come_from_the_vocabulary`）。
/// 于是新增一种可对拍的浮点输出时，失败的是"注解写不出它"，而不是它悄悄地无法被声明。
pub(crate) const DTYPES: [&str; 3] = ["f16", "bf16", "f32"];

/// `#[model(..)]` 解析后的三件事：注册名、生产形状、输出类型。
///
/// 三者都是编译期常量：注册名是 `REGISTRY` 的键，生产形状决定审计用的方阵，输出类型决定
/// 对拍阈值。没有一件需要运行期再判断。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelDecl {
    name: String,
    shape: [usize; 2],
    dtype: String,
    skeleton: String,
}

impl ModelDecl {
    /// 注册表的键（与 `ops/<name>.toml` 同名）。
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// 生产形状 `(m, n)`。
    pub(crate) fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// 输出类型名（[`DTYPES`] 里的一个）。
    pub(crate) fn dtype(&self) -> &str {
        &self.dtype
    }

    /// 骨架名（[`SKELETONS`] 里的一个），缺省为 `gemm`。
    pub(crate) fn skeleton(&self) -> &str {
        &self.skeleton
    }
}

/// 解析 `#[model(name = "..", shape = [..], dtype = "..")]`。
///
/// 三个键都是必填：缺 `name` 就没有注册表的键，缺 `shape` 就没有生产形状，缺 `dtype` 就只能
/// 靠默认值猜输出宽度。`shape` 恰好两个轴——`Model::shape` 就是 `(m, n)`。
pub(crate) fn parse_model(tokens: &[Token]) -> Result<ModelDecl, Error> {
    let mut cursor = Cursor::new(tokens);
    let mut name: Option<String> = None;
    let mut shape: Option<[usize; 2]> = None;
    let mut dtype: Option<String> = None;
    let mut skeleton: Option<String> = None;

    while !cursor.is_empty() {
        let key_position = cursor.position();
        let key = cursor.ident("键名")?;
        cursor.punct('=', "键与值之间的 `=`")?;
        match key.as_str() {
            "name" => {
                if name.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `name` 重复：一条语义只有一个注册名",
                    ));
                }
                let token = cursor.next("`name` 的值")?;
                let text = string(token).map_err(|error| error.rebased(key_position))?;
                checked_name(&text).map_err(|error| error.rebased(key_position))?;
                name = Some(text);
            }
            "shape" => {
                if shape.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `shape` 重复：生产形状只能声明一次",
                    ));
                }
                let (group, array) = cursor.group(Bracket::Square, "`shape = [..]` 的方括号")?;
                let mut dims = Vec::with_capacity(array.len());
                for item in elements(group, array, "`shape` 的元素")? {
                    let dimension = unsigned(item).map_err(|error| error.rebased(group))?;
                    if dimension == 0 {
                        return Err(Error::at(
                            group,
                            "生产形状的每个轴都必须大于 0：零宽形状不是一个可计算的形状",
                        ));
                    }
                    dims.push(dimension);
                }
                let [m, n] = dims.as_slice() else {
                    return Err(Error::at(
                        group,
                        format!(
                            "生产形状必须写成两个轴 `[m, n]`，得到 {} 个：`Model::shape` 是 `(m, n)`",
                            dims.len()
                        ),
                    ));
                };
                shape = Some([*m, *n]);
            }
            "dtype" => {
                if dtype.is_some() {
                    return Err(Error::at(
                        key_position,
                        "注解键 `dtype` 重复：输出类型只能声明一次",
                    ));
                }
                let token = cursor.next("`dtype` 的值")?;
                let text = string(token).map_err(|error| error.rebased(key_position))?;
                if !DTYPES.contains(&text.as_str()) {
                    return Err(Error::at(
                        key_position,
                        format!(
                            "`dtype = \"{text}\"` 不在输出类型闭集内（可用：{}）",
                            DTYPES.join("、")
                        ),
                    ));
                }
                dtype = Some(text);
            }
            "skeleton" => {
                let token = cursor.next("`skeleton` 的值")?;
                let text = string(token).map_err(|error| error.rebased(key_position))?;
                if !SKELETONS.contains(&text.as_str()) {
                    return Err(Error::at(
                        key_position,
                        format!(
                            "`skeleton = \"{text}\"` 不在骨架闭集内（可用：{}）",
                            SKELETONS.join("、")
                        ),
                    ));
                }
                skeleton = Some(text);
            }
            other => {
                return Err(Error::at(
                    key_position,
                    format!("未知的注解键 `{other}`（可用：{}）", MODEL_KEYS.join("、")),
                ));
            }
        }
        if cursor.is_empty() {
            break;
        }
        cursor.punct(',', "键之间的 `,`")?;
    }

    let name = name.ok_or_else(|| Error::at(0, "缺少必填键 `name = \"..\"`"))?;
    let shape = shape.ok_or_else(|| Error::at(0, "缺少必填键 `shape = [m, n]`"))?;
    let dtype = dtype.ok_or_else(|| Error::at(0, "缺少必填键 `dtype = \"..\"`"))?;
    Ok(ModelDecl {
        name,
        shape,
        dtype,
        skeleton: skeleton.unwrap_or_else(|| "gemm".to_string()),
    })
}

/// `#[model]` 标注的是**一个具体格式的别名**：`type 名字 = <具体实例>;`。
///
/// 与 [`target_name`] 不同（那两道注解标 struct / enum），这里是别名：`MODEL` 的入口要写成
/// `<名字 as Format>`，名字必须能被当作类型用，而泛型别名不能。
pub(crate) fn model_target(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    for (position, token) in tokens.iter().enumerate() {
        let Token::Ident(word) = token else {
            continue;
        };
        match word.as_str() {
            "type" => {
                let Some(Token::Ident(name)) = tokens.get(position + 1) else {
                    return Err(Error::at(position, "`type` 后应当紧跟别名"));
                };
                return match tokens.get(position + 2) {
                    Some(Token::Punct('=')) => Ok(name.clone()),
                    Some(Token::Punct('<')) => Err(Error::at(
                        position + 1,
                        format!(
                            "`#[{attribute}]` 不支持泛型别名：`{name}` 请先固定成一个具体实例（例如 `Quantized<32>`）"
                        ),
                    )),
                    _ => Err(Error::at(
                        position + 1,
                        format!(
                            "`{name}` 必须写成 `type {name} = <具体格式>;`：语义表要绑到那个类型上"
                        ),
                    )),
                };
            }
            "struct" | "enum" | "fn" | "trait" | "impl" | "mod" | "union" | "static" | "const" => {
                return Err(Error::at(
                    position,
                    format!(
                        "`#[{attribute}]` 只能标注 `type` 别名，得到 `{word}`：语义表绑的是\"哪份格式契约\"，不是行为"
                    ),
                ));
            }
            _ => {}
        }
    }
    Err(Error::at(
        0,
        format!("`#[{attribute}]` 只能标注 `type` 别名：找不到类型定义"),
    ))
}

/// 与 [`type_name`] 同一判据，只是错误里写的注解名不同。
///
/// 两道注解标的是同一类东西——"一份配置 + 一个具体类型"。被判据只有一份，所以
/// `#[frontend]` 与 `#[layer]` 不会对"能不能标 fn / 能不能带泛型"给出两种答案。
/// 注解名作为参数传进来，是为了让错误文案指向用户真正写下的那一个。
pub(crate) fn target_name(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    for (position, token) in tokens.iter().enumerate() {
        let Token::Ident(word) = token else {
            continue;
        };
        match word.as_str() {
            "fn" | "trait" | "impl" | "mod" | "union" | "type" | "static" | "const" => {
                return Err(Error::at(
                    position,
                    format!(
                        "`#[{attribute}]` 只能标注 struct / enum，得到 `{word}`：注解标的是类型，不是行为"
                    ),
                ));
            }
            "struct" | "enum" => {
                let Some(Token::Ident(name)) = tokens.get(position + 1) else {
                    return Err(Error::at(position, format!("`{word}` 后应当紧跟类型名")));
                };
                if matches!(tokens.get(position + 2), Some(Token::Punct('<'))) {
                    return Err(Error::at(
                        position + 1,
                        format!(
                            "`#[{attribute}]` 不支持泛型参数：`{name}` 请改成形状固定的具体类型"
                        ),
                    ));
                }
                return Ok(name.clone());
            }
            _ => {}
        }
    }
    Err(Error::at(
        0,
        format!("`#[{attribute}]` 只能标注 struct / enum：找不到类型定义"),
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

    fn parse_frontend_text(text: &str) -> Result<Frontend, Error> {
        parse_frontend(&lex(text))
    }

    /// 三个键都是必填，且键顺序不影响结果：注解是配置，不该有顺序语义。
    #[test]
    fn a_frontend_annotation_carries_the_language_the_suffix_and_the_tag() {
        let frontend = parse_frontend_text(r#"tag = "witness", emit = "metal", ext = "metal""#)
            .expect("合法前端注解");
        assert_eq!(frontend.emit(), "metal");
        assert_eq!(frontend.ext(), "metal");
        assert_eq!(frontend.tag(), "witness");
    }

    /// 缺一个键就必须报错：缺 `ext` 会让见证按另一门语言的后缀去找文件。
    #[test]
    fn a_frontend_annotation_requires_all_three_keys() {
        for (text, missing) in [
            (r#"emit = "metal", ext = "metal""#, "tag"),
            (r#"emit = "metal", tag = "witness""#, "ext"),
            (r#"ext = "metal", tag = "witness""#, "emit"),
        ] {
            let error = parse_frontend_text(text).expect_err("缺必填键");
            assert!(
                error.to_string().contains(missing),
                "报错要点名缺的键 `{missing}`：{error}"
            );
        }
    }

    /// 闭集之外的写法一律拒收：未知键、未知语言、同一个键写两遍。
    #[test]
    fn a_frontend_annotation_rejects_anything_outside_the_closed_sets() {
        let error = parse_frontend_text(r#"emit = "metal", ext = "metal", tag = "w", nope = "x""#)
            .expect_err("未知键");
        assert!(error.to_string().contains("nope"), "{error}");

        let error = parse_frontend_text(r#"emit = "vulkan", ext = "spv", tag = "w""#)
            .expect_err("语言不在闭集内");
        assert!(error.to_string().contains("vulkan"), "{error}");

        let error = parse_frontend_text(r#"emit = "metal", emit = "metal", ext = "m", tag = "w""#)
            .expect_err("重复键");
        assert!(error.to_string().contains("重复"), "{error}");
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
            target_name(&lex("pub struct OutputHead;"), "layer").expect("struct"),
            "OutputHead"
        );
        assert_eq!(
            target_name(&lex("pub(crate) enum Head { A }"), "layer").expect("enum"),
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
            let error = target_name(&lex(text), "layer").expect_err(keyword);
            assert!(error.message().contains(keyword), "{text}: {error}");
        }
    }

    #[test]
    fn a_generic_target_is_rejected_with_the_reason() {
        let error = target_name(&lex("struct Head<T> { inner: T }"), "layer").expect_err("泛型");
        assert!(error.message().contains("泛型"), "{error}");
        assert_eq!(error.position(), 1, "错误指向类型名");
    }

    #[test]
    fn an_empty_item_is_rejected_rather_than_guessed() {
        let error = target_name(&[], "layer").expect_err("空项");
        assert!(error.message().contains("找不到类型定义"), "{error}");
    }
    #[test]
    fn a_model_declaration_carries_name_shape_and_dtype() {
        let model = parse_model(&lex(
            r#"name = "gemv_i4", shape = [4096, 4096], dtype = "f32""#,
        ))
        .expect("合法注解");
        assert_eq!(model.name(), "gemv_i4");
        assert_eq!(model.shape(), [4096, 4096]);
        assert_eq!(model.dtype(), "f32");
    }

    #[test]
    fn a_model_needs_all_three_keys() {
        for (text, key) in [
            (r#"shape = [4, 4], dtype = "f32""#, "name"),
            (r#"name = "a", dtype = "f32""#, "shape"),
            (r#"name = "a", shape = [4, 4]"#, "dtype"),
        ] {
            let error = parse_model(&lex(text)).expect_err(key);
            assert!(error.message().contains(key), "{text}: {error}");
        }
    }

    #[test]
    fn a_model_shape_is_exactly_two_axes() {
        let error = parse_model(&lex(r#"name = "a", shape = [4, 4, 4], dtype = "f32""#))
            .expect_err("三个轴");
        assert!(error.message().contains("两个轴"), "{error}");
        assert!(
            parse_model(&lex(r#"name = "a", shape = [4], dtype = "f32""#)).is_err(),
            "一条轴上没有可计价的投影"
        );
        assert!(
            parse_model(&lex(r#"name = "a", shape = [4, 0], dtype = "f32""#)).is_err(),
            "零宽轴不是一个可计算的形状"
        );
    }

    #[test]
    fn a_model_dtype_is_a_closed_set() {
        let error = parse_model(&lex(r#"name = "a", shape = [4, 4], dtype = "f64""#))
            .expect_err("闭集之外");
        assert!(error.message().contains("闭集"), "{error}");
        let error = parse_model(&lex(
            r#"name = "a", shape = [4, 4], dtype = "f32", extra = 1"#,
        ))
        .expect_err("未知键");
        assert!(error.message().contains("未知的注解键"), "{error}");
        assert!(
            parse_model(&lex(
                r#"name = "a", shape = [4, 4], dtype = "f32", name = "b""#
            ))
            .is_err(),
            "重复键"
        );
    }

    #[test]
    fn a_model_annotation_targets_a_type_alias() {
        assert_eq!(
            model_target(&lex("pub type I4 = Quantized<32>;"), "model").expect("别名"),
            "I4"
        );
        // 行为不是配置：标注 struct 是错的，`MODEL` 要绑的是"哪份格式"。
        let error = model_target(&lex("pub struct Head;"), "model").expect_err("struct");
        assert!(error.message().contains("type"), "{error}");
        // 泛型别名不能当类型用。
        let error = model_target(&lex("pub type Head<T> = T;"), "model").expect_err("泛型别名");
        assert!(error.message().contains("泛型"), "{error}");
    }
}
