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
    /// 从借用的词法参数构造 owned 参数。
    fn from_ref(arg: &Arg<'_>) -> Self {
        match arg {
            Arg::Expr(text) => Self::Expr((*text).to_string()),
            Arg::Named { key, value } => Self::Named {
                key: (*key).to_string(),
                value: (*value).to_string(),
            },
        }
    }

    /// 借用视图：测试里核对 owned 参数与词法参数一致，不需要第二套占位符规则。
    #[cfg(test)]
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
        Segment::Call { prim, args, line } => prim_node(fragments, *prim, args, *line),
    }
}

/// 槽位取值：算子名 / 轴取值 / 布局构造子，三者之外的槽位是声明错误。
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
        .ok_or_else(|| {
            error(
                line,
                format!("槽位 `{{{{{name}}}}}` 既不是轴也不是布局槽位"),
            )
        })
}

/// 原语调用节点：片段必须存在，占位符必须能全部解析。
fn prim_node(
    fragments: &FragmentSet,
    prim: Prim,
    args: &[Arg<'_>],
    line: usize,
) -> diagnostic::Result<Node> {
    let Some(fragment) = fragments.get(prim) else {
        return Err(substitute::missing_fragment(line, prim, fragments));
    };
    let body = substitute::splice(fragment, args, line)?;
    let args = args.iter().map(CallArg::from_ref).collect();
    Ok(Node::Prim(PrimCall { prim, args, body }))
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
#[path = "../tests/ast_tests.rs"]
mod tests;
