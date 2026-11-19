//! 单 kernel 级结构化 AST（ADR 0017 的 L1）。
//!
//! 这是模板渲染的**可见中间产物**：词法片段先提升成拥有所有权的节点序列（[`Node`]），再由 [`render`]
//! 打印回文本。节点只对应模板级结构（文本 / 槽位 / 原语调用），不解析 C 语句本身——原语片段仍以文本
//! 挂在 [`PrimCall::body`] 上。[`encode`] 给出确定性编码作为缓存指纹输入，不依赖 `Debug` 格式。
//! 单一路径：`substitute::render` 就是 [`build`] + [`render`] 的薄包装，没有第二套字符串实现。

use crate::decl::{OpDecl, TileCombo};
use crate::substitute::{self, Arg, FragmentSet, Segment, error, lex};
use core::fmt::{self, Write as _};
use prof::atom::Prim;

/// 模板级节点：拥有所有权，不借用模板正文。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// 原样透传的 C 文本。
    Text(String),
    /// 槽位解析后的最终值（算子名 / 轴值 / 布局构造子）。
    Slot(String),
    /// 结构化的原语调用。
    Prim(PrimCall),
}

/// 一次 `@原语(...)` 调用：原语、参数与当前后端片段的展开文本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimCall {
    /// 原语（闭集之一）。
    pub prim: Prim,
    /// 调用参数（owned）。
    pub args: Vec<CallArg>,
    /// 片段经占位符替换后的文本；分号裁剪与缩进留给 [`render`]。
    pub body: String,
}

/// 原语调用参数的 owned 版本，对应 [`op::tpl::Arg`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallArg {
    /// 位置参数。
    Expr(String),
    /// 命名参数（`key: value`）。
    Named {
        /// 键。
        key: String,
        /// 值。
        value: String,
    },
}

impl CallArg {
    /// 从借用的词法参数构造 owned 参数（不做槽位替换，仅测试核对用）。
    #[cfg(test)]
    fn from_ref(arg: &Arg<'_>) -> Self {
        match arg {
            Arg::Expr(text) => Self::Expr((*text).to_string()),
            Arg::Named { key, value } => Self::Named {
                key: (*key).to_string(),
                value: (*value).to_string(),
            },
        }
    }

    /// 借用视图：owned 参数转回词法参数，供 [`substitute::splice`] 直接消费，不需要第二套占位符规则。
    fn as_ref(&self) -> Arg<'_> {
        match self {
            Self::Expr(text) => Arg::Expr(text),
            Self::Named { key, value } => Arg::Named { key, value },
        }
    }
}

/// 把一份声明 + tile + 片段层提升成节点序列。
///
/// # Errors
///
/// 槽位无法解析、片段缺失或占位符替换失败时，返回与 [`substitute::render`] 同源的诊断。
pub fn build(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
) -> diagnostic::Result<Vec<Node>> {
    let segments = lex(&op.template);
    let mut nodes = Vec::with_capacity(segments.len());
    for segment in &segments {
        nodes.push(node(op, combo, fragments, segment)?);
    }
    Ok(nodes)
}

/// 一个词法片段 → 一个节点：三种片段各有唯一的一条转换。
fn node(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
    segment: &Segment<'_>,
) -> diagnostic::Result<Node> {
    match segment {
        Segment::Text(part) => Ok(Node::Text((*part).to_string())),
        Segment::Slot { name, line } => Ok(Node::Slot(slot_value(op, combo, name, *line)?)),
        Segment::Call { prim, args, line } => prim_node(op, combo, fragments, *prim, args, *line),
    }
}

/// 槽位取值：算子名 / 轴取值 / 布局构造子 / 模板常量，四者之外的槽位是声明错误。
fn slot_value(
    op: &OpDecl,
    combo: &TileCombo,
    name: &str,
    line: usize,
) -> diagnostic::Result<String> {
    if name == "op" {
        return Ok(op.name.clone());
    }
    combo
        .get(name)
        .map(|value| value.to_string())
        .or_else(|| op.layout(name).map(|spec| spec.as_str().to_string()))
        .or_else(|| op.const_value(name).map(str::to_string))
        .ok_or_else(|| {
            error(
                line,
                format!("槽位 `{{{{{name}}}}}` 既不是轴、布局槽位，也不是模板常量"),
            )
        })
}

/// 原语调用节点：片段必须存在，占位符必须能全部解析；实参里的 `{{槽位}}` 先解析再拼接。
fn prim_node(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
    prim: Prim,
    args: &[Arg<'_>],
    line: usize,
) -> diagnostic::Result<Node> {
    let Some(fragment) = fragments.get(prim) else {
        return Err(substitute::missing_fragment(line, prim, fragments));
    };
    let resolved: Vec<CallArg> = args
        .iter()
        .map(|arg| resolve_arg(op, combo, arg, line))
        .collect::<diagnostic::Result<_>>()?;
    let borrowed: Vec<Arg<'_>> = resolved.iter().map(CallArg::as_ref).collect();
    let body = substitute::splice(fragment, &borrowed, line)?;
    Ok(Node::Prim(PrimCall {
        prim,
        args: resolved,
        body,
    }))
}

/// 一个实参的 `{{槽位}}` 替换：位置参数与命名参数的值都过一遍 [`resolve_slots`]。
fn resolve_arg(
    op: &OpDecl,
    combo: &TileCombo,
    arg: &Arg<'_>,
    line: usize,
) -> diagnostic::Result<CallArg> {
    match arg {
        Arg::Expr(text) => Ok(CallArg::Expr(resolve_slots(op, combo, text, line)?)),
        Arg::Named { key, value } => Ok(CallArg::Named {
            key: (*key).to_string(),
            value: resolve_slots(op, combo, value, line)?,
        }),
    }
}

/// 把一段文本里的 `{{槽位}}` 逐个换成取值；多数实参不含槽位，无 `{{` 时原样返回免分配。
fn resolve_slots(
    op: &OpDecl,
    combo: &TileCombo,
    text: &str,
    line: usize,
) -> diagnostic::Result<String> {
    if !text.contains("{{") {
        return Ok(text.to_string());
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let inner = &rest[start + 2..];
        let Some(end) = inner.find("}}") else {
            out.push_str(&rest[start..]);
            break;
        };
        out.push_str(&slot_value(op, combo, inner[..end].trim(), line)?);
        rest = &inner[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// 把节点序列打印回源文本：[`substitute::render`] 就是 [`build`] + 本函数的薄包装。
///
/// 原语片段是语句块，模板已提供 `;`：这里去掉重复终结符，并按调用点当前缩进对齐多行片段。
#[must_use]
pub fn render(nodes: &[Node]) -> String {
    let mut text = String::new();
    for node in nodes {
        match node {
            Node::Text(part) => text.push_str(part),
            Node::Slot(value) => text.push_str(value),
            Node::Prim(call) => crate::substitute::place(&mut text, &call.body),
        }
    }
    text
}

/// 节点序列的确定性规范编码（缓存指纹输入）。
///
/// 每个节点一行，前缀区分种类，字段经 `escape` 转义，换行、反斜杠与分隔符都不会产生歧义。
#[must_use]
pub fn encode(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        // 写入 `String` 永远不会失败：只有格式化实现本身出错才会走到 `Err`。
        let _ = write_node(&mut out, node);
    }
    out
}

/// 一个节点一行：`T`/`S`/`P` 三种前缀，字段一律经 [`escape`] 转义。
fn write_node(out: &mut String, node: &Node) -> fmt::Result {
    match node {
        Node::Text(part) => writeln!(out, "T {}", escape(part)),
        Node::Slot(value) => writeln!(out, "S {}", escape(value)),
        Node::Prim(call) => writeln!(
            out,
            "P {} {} {}",
            call.prim.as_str(),
            encode_args(&call.args),
            escape(&call.body)
        ),
    }
}

/// 参数序列：位置参数 `e…|`、命名参数 `n…=…|`，两者都以 `|` 结尾。
fn encode_args(args: &[CallArg]) -> String {
    args.iter()
        .map(|arg| match arg {
            CallArg::Expr(value) => format!("e{}|", escape(value)),
            CallArg::Named { key, value } => format!("n{}={}|", escape(key), escape(value)),
        })
        .collect()
}

/// 转义：反斜杠、换行与字段分隔符各有唯一编码，保证编码可逆且无歧义。
fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match escape_char(ch) {
            Some(coded) => out.push_str(coded),
            None => out.push(ch),
        }
    }
    out
}

/// 单个字符的编码；`None` 表示原样输出。闭集只有四个，多一个就多一种解码可能。
fn escape_char(ch: char) -> Option<&'static str> {
    match ch {
        '\\' => Some("\\\\"),
        '\n' => Some("\\n"),
        '|' => Some("\\p"),
        '=' => Some("\\q"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
use super::*;
use crate::builtin::Catalogue;
use crate::substitute;

/// 双射门：全部内置算子 × 每个 tile × 每台设备，AST 路径与字符串路径逐字节相等。
#[test]
fn ast_renders_byte_identical_text() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for profile in catalogue.profiles() {
        let fragments =
            prim::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext()).expect("片段层");
        for op in catalogue.ops() {
            for combo in crate::decl::expand(op).expect("tile 候选") {
                let direct = substitute::render(op, &combo, &fragments)
                    .unwrap_or_else(|err| panic!("字符串路径失败：{err}"));
                let nodes = build(op, &combo, &fragments).expect("AST 构建");
                let via_ast = render(&nodes);
                assert_eq!(
                    direct, via_ast,
                    "`{}`/`{}`/`{combo}` 双射破裂",
                    profile.meta.vendor, op.name
                );
            }
        }
    }
}

/// 规范编码确定且对字段敏感：改任一节点都换编码。
#[test]
fn encode_is_stable_and_field_sensitive() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let op = catalogue.op("gemv_i4").expect("算子");
    let combo = crate::decl::expand(op).expect("tile").remove(0);
    let fragments = prim::builtin("cpu", "c").expect("片段");
    let nodes = build(op, &combo, &fragments).expect("AST");

    assert_eq!(encode(&nodes), encode(&nodes), "同一序列编码稳定");

    let mut changed = nodes.clone();
    if let Some(Node::Slot(value)) = changed.iter_mut().find(|n| matches!(n, Node::Slot(_))) {
        value.push('7');
    }
    assert_ne!(encode(&nodes), encode(&changed), "改槽位必须换编码");

    let mut changed = nodes.clone();
    if let Some(Node::Prim(call)) = changed.iter_mut().find(|n| matches!(n, Node::Prim(_))) {
        call.body.push(' ');
    }
    assert_ne!(encode(&nodes), encode(&changed), "改片段体必须换编码");
}

/// 节点往返：owned 参数与借用参数视图一致，splice 结果不依赖走哪条路径。
#[test]
fn owned_args_splice_like_borrowed_args() {
    let borrowed = [
        Arg::Expr("a"),
        Arg::Named {
            key: "along",
            value: "32",
        },
    ];
    let owned: Vec<CallArg> = borrowed.iter().map(CallArg::from_ref).collect();
    let view: Vec<Arg<'_>> = owned.iter().map(CallArg::as_ref).collect();
    assert_eq!(&borrowed[..], &view[..]);
}

/// 确定性 fuzz：对真实片段 body 随机插入空白 / 换行 / 分号，重建片段层后两条路径零差异。
///
/// 不依赖外部随机源：固定种子的 LCG 在任何机器上派生同一批扰动，缩进对齐与分号裁剪的边界
/// 因此被反复对拍。只插入空白与分号、不碰 `$` 占位符，保证 splice 语义不变。
#[test]
fn fuzzed_fragments_keep_the_two_paths_identical() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let op = catalogue.op("gemv_i4").expect("算子");
    let combos = crate::decl::expand(op).expect("tile 候选");
    let base = prim::builtin("cpu", "c").expect("真实片段");

    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..300 {
        let mut set = FragmentSet::new("cpu", "c");
        for &prim in Prim::ALL {
            let Some(fragment) = base.fragment(prim) else {
                continue;
            };
            let body = perturb(fragment.body(), &mut state);
            let source = format!("/* {} {prim}/z */\n{body}", prim::SIGN_MARKER);
            set.insert(prim, source).expect("合法签名片段");
        }
        for combo in &combos {
            let direct = substitute::render(op, combo, &set).expect("字符串路径");
            let nodes = build(op, combo, &set).expect("AST 构建");
            assert_eq!(direct, render(&nodes), "fuzz 双射破裂（{combo}）");
        }
    }
}

/// 确定性 LCG：无外部依赖的伪随机源。
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// 在随机位置插入空白 / 换行 / 分号，制造缩进与终结符的边界情况。
///
/// `$` 占位符的名字整体保留：不在 `$ax` 与 `is` 之间插入字符，否则会把 `$axis`
/// 拆成不存在的 `$ax`——那是生成器在破坏输入，不是被测代码的问题。
fn perturb(body: &str, state: &mut u64) -> String {
    const TOKENS: [&str; 5] = [" ", "  ", "\n", "\n    ", ";"];
    let mut out = String::with_capacity(body.len() + 16);
    let mut chars = body.chars().peekable();
    while let Some(ch) = chars.next() {
        out.push(ch);
        if ch == '$' {
            while let Some(&next) = chars.peek() {
                if next.is_ascii_alphanumeric() || next == '_' {
                    out.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            continue;
        }
        if lcg(state).is_multiple_of(7) {
            out.push_str(TOKENS[(lcg(state) as usize) % TOKENS.len()]);
        }
    }
    out
}
}

