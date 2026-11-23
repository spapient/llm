//! 三种声明的**键表**与解析产物：层（`#[spec]` 静态形态）、语义表（`#[model]`）、
//! 层（`#[spec]` 运行时形态）。

use crate::syntax::{
    Bracket, Cursor, Error, FIELDS, Hole, KeySpec, Position, Token, Value, checked_name, collect,
    declared_name, elements, generic, require, row, string, target_with, unsigned,
};

/// 注解允许的键；顺序即错误信息里的提示顺序。
pub(crate) const KEYS: [&str; 4] = ["name", "out", "in", "ops"];

/// 层解析后的四件事实：名字、输出形状、输入宽度、算子列表。
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

/// `#[spec(..)]` 的静态键表，顺序即「未知键」提示里的顺序。
const LAYER: [KeySpec<'static>; 4] = [
    row(KEYS[0], text_value, "名字只能声明一次", Some(" = \"..\"")),
    row(KEYS[1], axes_value, "输出形状只能声明一次", Some(" = [..]")),
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

/// 字符串字面量键值（`name` / `const` / `dtype` / `skeleton`）：再套一层名字约束。
fn text_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let token = cursor.next(&format!("`{}` 的值", spec.name))?;
    let text = string(token).map_err(|error| error.rebased(at))?;
    checked_name(&text, at)?;
    Ok(Value::Text(text))
}

/// 非零轴数组：括号与元素的措辞由键名推出。
fn axes_value(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let what = format!("`{} = [..]` 的方括号", spec.name);
    let (group, items) = cursor.group(Bracket::Square, &what)?;
    let mut axes: Vec<usize> = Vec::with_capacity(items.len());
    for item in elements(group, items, &format!("`{}` 的元素", spec.name))? {
        axes.push(positive(item, group, spec.name)?);
    }
    Ok(Value::Axes(axes, group))
}

/// 一个轴：十进制无符号整数，且大于 0——零宽轴不是一个可计算的形状。
fn positive(item: &Token, group: Position, key: &str) -> Result<usize, Error> {
    let axis = unsigned(item).map_err(|error| error.rebased(group))?;
    if axis == 0 {
        return Err(Error::at(
            group,
            format!("`{key}` 的每个轴都必须大于 0：零宽轴不是一个可计算的形状"),
        ));
    }
    Ok(axis)
}

/// `ops = [".."]`：名字数组，逐项校验、去重；空表由 [`parse`] 单独拦。
fn ops_value(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let what = format!("`{} = [\"..\"]` 的方括号", spec.name);
    let (group, items) = cursor.group(Bracket::Square, &what)?;
    let mut names: Vec<String> = Vec::with_capacity(items.len());
    for item in elements(group, items, &format!("`{}` 的元素", spec.name))? {
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
    let token = cursor.next(&format!("`{}` 的值", spec.name))?;
    let width = unsigned(token).map_err(|error| error.rebased(at))?;
    if width == 0 {
        return Err(Error::at(at, "输入宽度必须大于 0：零宽的规约轴没有下界"));
    }
    Ok(Value::Width(width, at))
}

/// `dtype = ".."`：对拍有阈值的输出类型闭集。
fn dtype_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    closed_value(cursor, at, spec, "输出类型", &DTYPES)
}

/// `skeleton = ".."`：骨架闭集。
fn skeleton_value(
    cursor: &mut Cursor<'_>,
    at: Position,
    spec: &KeySpec<'_>,
) -> Result<Value, Error> {
    closed_value(cursor, at, spec, "骨架", &SKELETONS)
}

/// 闭集校验：`what` 是「输出类型 / 骨架」，提示里直接列出可用名字。
fn closed_value(
    cursor: &mut Cursor<'_>,
    at: Position,
    spec: &KeySpec<'_>,
    what: &str,
    set: &[&str],
) -> Result<Value, Error> {
    let text = text_value(cursor, at, spec)?;
    if set.contains(&text.text()) {
        return Ok(text);
    }
    Err(Error::at(
        at,
        format!(
            "`{} = \"{}\"` 不在{what}闭集内（可用：{}）",
            spec.name,
            text.text(),
            set.join("、")
        ),
    ))
}

/// `skeleton` 闭集：决定生成的 `MODEL` 走哪副算子骨架。
pub(crate) const SKELETONS: [&str; 15] = [
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
    "matmul",
    "scan",
];

/// `gemm` 骨架认识的注册名：各格式各有 `nn::gemm_*` 模块，名字就是模块选择器。
pub(crate) const GEMM_SKELETON_NAMES: [&str; 4] = ["gemm_f32", "gemm_i8", "gemm_i4", "gemm_i2"];

/// `#[model(..)]` 的 `dtype` 闭集：对拍**有阈值**的那几种输出类型。
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
        text_value,
        "一条语义只有一个注册名",
        Some(" = \"..\""),
    ),
    row(
        "shape",
        axes_value,
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
    row("const", text_value, "生成的常量名只能声明一次", None),
];

/// 解析 `#[model(name = "..", shape = [..], dtype = "..", skeleton = "..", const = "..")]`。
/// `name` / `shape` / `dtype` 必填，`shape` 恰好两个轴，`skeleton` 缺省 `gemv`，`const` 缺省 `MODEL`。
pub(crate) fn parse_model(tokens: &[Token]) -> Result<ModelDecl, Error> {
    let values = collect(tokens, &MODEL)?;
    require(&values, &MODEL)?;
    let name = values[0].text();
    let skeleton = defaulted(&values[3], "gemv");
    check_gemm_name(skeleton, name)?;
    Ok(ModelDecl {
        name: name.to_string(),
        shape: two_axes(&values[1])?,
        dtype: values[2].text().to_string(),
        skeleton: skeleton.to_string(),
        const_name: defaulted(&values[4], "MODEL").to_string(),
    })
}

/// 可选键的缺省值：没写就用生成器实际走的那条分支（不是另写一个默认分支）。
fn defaulted<'a>(value: &'a Value, fallback: &'a str) -> &'a str {
    match value.text() {
        "" => fallback,
        text => text,
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

/// 运行时形态解析后的四件事实：名字、输出形状、算子与角色全部写成 `$构造参数`，
/// 由生成的构造宏在调用点取值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeDecl {
    /// 层名的参数名。
    pub(crate) name: String,
    /// 输出轴的参数名，顺序即轴序；同一个参数可以出现两次（方阵）。
    pub(crate) out: Vec<String>,
    /// 算子名的参数名，非空且不重复。
    pub(crate) ops: Vec<String>,
    /// 参数角色的参数名；`None` 表示走默认角色（`Trainable`）。
    pub(crate) role: Option<String>,
}

/// `#[spec(..)]` 的运行时键表：每个值只能是 `$构造参数`；`in` 不在表里——下界由层自己实现。
const RUNTIME: [KeySpec<'static>; 4] = [
    row("name", hole_value, "名字只能声明一次", Some(" = $参数")),
    row(
        "out",
        runtime_holes,
        "输出形状只能声明一次",
        Some(" = [$参数]"),
    ),
    row(
        "ops",
        runtime_ops,
        "算子列表只能声明一次",
        Some(" = [$参数]"),
    ),
    row("role", hole_value, "参数角色只能声明一次", None),
];

/// 注解里只要出现 `$` 就走运行时形态：静态形态的取值文法不接受它。
pub(crate) fn is_runtime(tokens: &[Token]) -> bool {
    tokens.iter().any(|token| match token {
        Token::Punct('$') => true,
        Token::Group(_, items) => is_runtime(items),
        _ => false,
    })
}

/// 解析运行时形态；跨键冲突在 [`distinct`] 里一次报清。
pub(crate) fn parse_runtime(tokens: &[Token]) -> Result<RuntimeDecl, Error> {
    let values = collect(tokens, &RUNTIME)?;
    require(&values, &RUNTIME)?;
    distinct(&values)?;
    let ops: Vec<String> = values[2].hole_names().map(str::to_string).collect();
    if ops.is_empty() {
        return Err(Error::at(
            tokens.len().saturating_sub(1),
            "`ops` 不能为空：一个层至少要引用一个已声明的算子",
        ));
    }
    Ok(RuntimeDecl {
        name: values[0].hole().to_string(),
        out: values[1].hole_names().map(str::to_string).collect(),
        ops,
        role: values[3].hole_names().next().map(str::to_string),
    })
}

/// 同一个参数名不能同时用作名字、轴宽、算子与角色：生成的宏只有一个参数，却要喂给不同类型。
fn distinct(values: &[Value]) -> Result<(), Error> {
    let groups = [
        (0, "名字"),
        (1, "输出形状"),
        (2, "算子列表"),
        (3, "参数角色"),
    ];
    for (left, left_name) in groups {
        for (right, right_name) in groups.iter().skip(left + 1) {
            let (holes, _) = values[left].holes();
            let (others, _) = values[*right].holes();
            for (name, at) in holes {
                let reused = others.iter().any(|(other, _)| other == name);
                if reused {
                    return Err(Error::at(
                        *at,
                        format!(
                            "运行时参数 `${name}` 同时用作{left_name}与{right_name}：同一个名字不能承担两种类型"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `$ident`：运行时形态的唯一取值写法；错误指在 `$` 或它后面的那个 token 上。
fn hole(cursor: &mut Cursor<'_>, key: &str) -> Result<Hole, Error> {
    let at = cursor.idx;
    cursor.punct('$', &format!("`{key}` 的 `$` 参数标记"))?;
    let name_at = cursor.idx;
    match cursor.next(&format!("`{key}` 的 `$` 参数名"))? {
        Token::Ident(name) => Ok((name.clone(), at)),
        other => Err(Error::at(
            name_at,
            format!("`$` 后应当是参数名（标识符），得到 {}", other.describe()),
        )),
    }
}

/// `name = $ident` / `role = $ident`：单个洞。
fn hole_value(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let (name, at) = hole(cursor, spec.name)?;
    Ok(Value::Holes(vec![(name, at)], at))
}

/// `out = [$..]` / `ops = [$..]`：方括号里每一项都是一个运行时洞，允许尾随逗号。
fn runtime_holes(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let what = format!("`{} = [$参数]` 的方括号", spec.name);
    let (group, items) = cursor.group(Bracket::Square, &what)?;
    let mut inner = Cursor {
        tokens: items,
        idx: 0,
    };
    let mut holes: Vec<Hole> = Vec::new();
    while let Some(item) = next_hole(&mut inner, group, spec.name)? {
        holes.push(item);
    }
    Ok(Value::Holes(holes, group))
}

/// 读下一个洞并吃掉分隔符；数组读完返回 `None`。
fn next_hole(inner: &mut Cursor<'_>, group: Position, key: &str) -> Result<Option<Hole>, Error> {
    if inner.is_empty() {
        return Ok(None);
    }
    let item = hole(inner, key)?;
    if !inner.is_empty() {
        inner
            .punct(',', "数组元素之间的 `,`")
            .map_err(|error| error.rebased(group))?;
    }
    Ok(Some(item))
}

/// `ops = [$a, ..]`：同一个算子参数写两次是静态可知的重复，直接拒绝。
fn runtime_ops(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let holes = runtime_holes(cursor, 0, spec)?;
    let (items, group) = holes.holes();
    for (idx, (name, _)) in items.iter().enumerate() {
        if items[..idx].iter().any(|(other, _)| other == name) {
            return Err(Error::at(
                group,
                format!("算子参数 `${name}` 在同一个层里出现了两次：多写的这一次不会生效"),
            ));
        }
    }
    Ok(holes)
}

/// 运行时形态只能标命名字段 `struct`：`spec` 字段要注入到 `{ .. }` 字段体里。
pub(crate) fn runtime_target(tokens: &[Token], attribute: &str) -> Result<String, Error> {
    let (position, word) = target_with(tokens, attribute, &FIELDS)?;
    generic(
        tokens,
        position,
        format!("`#[{attribute}]` 不支持泛型参数：请把类型改成形状固定的具体类型"),
    )?;
    let name = declared_name(tokens, position, word, "类型名")?;
    let has_body = tokens[position..]
        .iter()
        .any(|token| matches!(token, Token::Group(Bracket::Brace, _)));
    if has_body {
        return Ok(name);
    }
    Err(Error::at(
        position,
        format!(
            "运行时形态的 `#[{attribute}]` 只能标注带命名字段的 struct：`{name}` 没有 `{{ .. }}` 字段体"
        ),
    ))
}

/// `#[parallel(..)]` 的归约闭集短名；行序与 `shard::Reduce` 行表一一对应。
pub(crate) const REDUCES: [&str; 4] = ["sum", "max", "min", "prod"];

/// `#[parallel(..)]` 的优先级闭集短名；行序与 `Priority` 行表一一对应。
pub(crate) const PRIORITIES: [&str; 3] = ["high", "normal", "low"];

/// `#[parallel(..)]` 解析后的六件事实：三维度数、归约种类、平均开关、调度优先级。
///
/// 短名原样保留，渲染层再映射成 `shard` 侧的枚举变体；注解层不反向依赖 `shard` 的类型。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParallelDecl {
    /// 张量并行度。
    pub(crate) tp: usize,
    /// 流水并行度。
    pub(crate) pp: usize,
    /// 数据并行度。
    pub(crate) dp: usize,
    /// 归约种类短名（[`REDUCES`] 之一）。
    pub(crate) reduce: String,
    /// 归约后是否按数据并行组数取平均。
    pub(crate) average: bool,
    /// 调度优先级短名（[`PRIORITIES`] 之一）。
    pub(crate) priority: String,
}

/// `#[parallel(..)]` 的键表：三维度数可选（缺省 1），口径键可选（缺省求和 / 平均 / 普通）。
const PARALLEL: [KeySpec<'static>; 6] = [
    row("tp", degree_value, "张量并行度只能声明一次", None),
    row("pp", degree_value, "流水并行度只能声明一次", None),
    row("dp", degree_value, "数据并行度只能声明一次", None),
    row(
        "reduce",
        reduce_value,
        "归约种类只能声明一次",
        Some(" = \"sum\""),
    ),
    row("average", flag_value, "平均开关只能声明一次", None),
    row(
        "priority",
        priority_value,
        "调度优先级只能声明一次",
        Some(" = \"normal\""),
    ),
];

/// 解析 `#[parallel(tp = .., pp = .., dp = .., reduce = "..", average = .., priority = "..")]`。
/// 所有键可选；缺省口径与 `ParallelConfig::single()` 一致。
pub(crate) fn parse_parallel(tokens: &[Token]) -> Result<ParallelDecl, Error> {
    let values = collect(tokens, &PARALLEL)?;
    Ok(ParallelDecl {
        tp: degree_of(&values[0], 1),
        pp: degree_of(&values[1], 1),
        dp: degree_of(&values[2], 1),
        reduce: defaulted(&values[3], "sum").to_string(),
        average: !matches!(values[4].text(), "false"),
        priority: defaulted(&values[5], "normal").to_string(),
    })
}

/// 未写的度数回落到单进程 1；零值在 [`degree_value`] 里已被拒绝。
fn degree_of(value: &Value, fallback: usize) -> usize {
    match value {
        Value::Width(width, _) => *width,
        _ => fallback,
    }
}

/// 一个并行度：十进制正整数；零度并行切不出组，构造期就拒。
fn degree_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let token = cursor.next(&format!("`{}` 的值", spec.name))?;
    let width = unsigned(token).map_err(|error| error.rebased(at))?;
    if width == 0 {
        return Err(Error::at(
            at,
            format!("`{}` 并行度必须大于 0：零度并行没有可切的组", spec.name),
        ));
    }
    Ok(Value::Width(width, at))
}

/// `reduce = ".."`：归约闭集。
fn reduce_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    closed_value(cursor, at, spec, "归约种类", &REDUCES)
}

/// `priority = ".."`：优先级闭集。
fn priority_value(
    cursor: &mut Cursor<'_>,
    at: Position,
    spec: &KeySpec<'_>,
) -> Result<Value, Error> {
    closed_value(cursor, at, spec, "优先级", &PRIORITIES)
}

/// `average = true | false`：布尔字面量；只认两个标识符，落空走解析层的缺省 `true`。
fn flag_value(cursor: &mut Cursor<'_>, at: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let token = cursor.next(&format!("`{}` 的值", spec.name))?;
    match token {
        Token::Ident(word) if word == "true" || word == "false" => Ok(Value::Text(word.clone())),
        other => Err(Error::at(
            at,
            format!(
                "`{}` 只能是 `true` 或 `false`，得到 {}",
                spec.name,
                other.describe()
            ),
        )),
    }
}
