//! 运行时形态：每个值写成 `$构造参数`，由生成的构造宏在调用点取值。
//!
//! 与静态形态共用父模块的键机器（[`collect`] / [`KeySpec`] / [`Cursor`]），只是把取值文法换成
//! 洞：`name = $name, out = [$tokens, $dim], ops = [$op], role = $role`。`in` 不在运行时表里——
//! 下界由层自己实现，静态的 `in` 无法绑定运行期宽度。

use super::*;

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
        runtime_out,
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

impl Value {
    /// 单个运行时洞的参数名；其余变体回落到空串。
    pub(super) fn hole(&self) -> &str {
        match self {
            Self::Hole(name, _) => name,
            _ => "",
        }
    }

    /// 运行时洞列表与它的位置；单个洞也算一条，其余变体回落到空表。
    pub(super) fn holes(&self) -> (Vec<RuntimeHole>, Position) {
        match self {
            Self::Hole(name, at) => (vec![(name.clone(), *at)], *at),
            Self::Holes(holes, at) => (holes.clone(), *at),
            _ => (Vec::new(), 0),
        }
    }

    /// 运行时洞参数名，顺序即写法顺序。
    pub(super) fn hole_names(&self) -> Vec<String> {
        self.holes().0.into_iter().map(|(name, _)| name).collect()
    }
}

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
    let ops = values[2].hole_names();
    if ops.is_empty() {
        return Err(Error::at(
            tokens.len().saturating_sub(1),
            "`ops` 不能为空：一个层至少要引用一个已声明的算子",
        ));
    }
    Ok(RuntimeDecl {
        name: values[0].hole().to_string(),
        out: values[1].hole_names(),
        ops,
        role: chosen_role(&values[3]),
    })
}

/// `role` 键的缺省值：没写就是 `None`，构造宏走默认角色。
fn chosen_role(value: &Value) -> Option<String> {
    match value {
        Value::Empty => None,
        _ => Some(value.hole().to_string()),
    }
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
        for (right, right_name) in groups.into_iter().skip(left + 1) {
            for (name, at) in values[left].holes().0 {
                if values[right]
                    .holes()
                    .0
                    .iter()
                    .any(|(other, _)| other == &name)
                {
                    return Err(Error::at(
                        at,
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
fn hole(cursor: &mut Cursor<'_>, key: &str) -> Result<RuntimeHole, Error> {
    let at = cursor.index;
    cursor.punct('$', &format!("`{key}` 的 `$` 参数标记"))?;
    let name_at = cursor.index;
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
    Ok(Value::Hole(name, at))
}

/// `out = [$..]` / `ops = [$..]`：方括号里每一项都是一个运行时洞，允许尾随逗号。
fn runtime_holes(cursor: &mut Cursor<'_>, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let group_what = format!("`{} = [$参数]` 的方括号", spec.name);
    let (group, items) = cursor.group(Bracket::Square, &group_what)?;
    let mut inner = Cursor {
        tokens: items,
        index: 0,
    };
    let mut holes = Vec::new();
    while let Some(item) = next_hole(&mut inner, group, spec.name)? {
        holes.push(item);
    }
    Ok(Value::Holes(holes, group))
}

/// 读下一个洞并吃掉分隔符；数组读完返回 `None`。
fn next_hole(
    inner: &mut Cursor<'_>,
    group: Position,
    key: &str,
) -> Result<Option<RuntimeHole>, Error> {
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

/// `out = [$a, ..]`：轴序即参数顺序；空数组仍是标量输出。
fn runtime_out(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    runtime_holes(cursor, spec)
}

/// `ops = [$a, ..]`：同一个算子参数写两次是静态可知的重复，直接拒绝。
fn runtime_ops(cursor: &mut Cursor<'_>, _: Position, spec: &KeySpec<'_>) -> Result<Value, Error> {
    let holes = runtime_holes(cursor, spec)?;
    let (items, group) = holes.holes();
    for (index, (name, _)) in items.iter().enumerate() {
        if items[..index].iter().any(|(other, _)| other == name) {
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
    let Some((position, word)) = keyword(tokens) else {
        return Err(Error::at(
            0,
            format!("`#[{attribute}]` 只能标注命名字段 struct：找不到类型定义"),
        ));
    };
    if word != "struct" {
        return Err(Error::at(
            position,
            format!(
                "运行时形态的 `#[{attribute}]` 只能标注 struct，得到 `{word}`：字段注入需要命名字段体"
            ),
        ));
    }
    let name = named(tokens, position, word, attribute)?;
    let has_body = tokens[position..]
        .iter()
        .any(|token| matches!(token, Token::Group(Bracket::Brace, _)));
    if !has_body {
        return Err(Error::at(
            position,
            format!(
                "运行时形态的 `#[{attribute}]` 只能标注带命名字段的 struct：`{name}` 没有 `{{ .. }}` 字段体"
            ),
        ));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
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
}
