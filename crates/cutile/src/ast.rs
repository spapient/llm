//! 单 kernel 级结构化 AST（ADR 0017 的 L1）。
//!
//! 这是模板渲染的**可见中间产物**：`substitute::render` 把词法片段直接拼成字符串，本模块把同一份
//! 词法片段先提升成拥有所有权的节点序列（[`Node`]），再由 [`render`] 打印回文本。节点对应
//! **模板级结构**（文本 / 槽位 / 原语调用），不解析 C 语句本身——原语的后端片段仍以文本挂在
//! [`PrimCall::body`] 上，后端真正分叉的 lowering 不会被强行抽象掉。
//!
//! # 双射
//!
//! 两条路径对同一份输入必须逐字节相等：
//!
//! ```text
//! 旧：lex → substitute::render                 → String
//! 新：lex → ast::build → Vec<Node> → ast::render → String
//! ```
//!
//! 片段占位符替换共用 `substitute::splice`，缩进对齐与分号裁剪在 [`render`] 里逐行复刻，因此
//! 双射不是"差不多"，而是由 `ast_renders_byte_identical_text` 在全部内置算子 × tile × 设备上断言。
//!
//! # 规范编码
//!
//! [`encode`] 给出确定性、无歧义的节点序列编码，作为缓存指纹的输入：改任何一个节点的任何字段都
//! 换编码，且不依赖 Rust 的 `Debug` 格式（那不是稳定的跨版本契约）。

use crate::decl::{OpDecl, TileCombo};
use crate::substitute::{self, Arg, FragmentSet, error, lex};
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
    let mut nodes = Vec::new();
    for segment in lex(&op.template) {
        match segment {
            crate::substitute::Segment::Text(part) => nodes.push(Node::Text(part.to_string())),
            crate::substitute::Segment::Slot { name, line } => {
                let value = if name == "op" {
                    op.name.clone()
                } else if let Some(value) = combo.get(name) {
                    value.to_string()
                } else if let Some(spec) = op.layout(name) {
                    spec.as_str().to_string()
                } else {
                    return Err(error(
                        line,
                        format!("槽位 `{{{{{name}}}}}` 既不是轴也不是布局槽位"),
                    ));
                };
                nodes.push(Node::Slot(value));
            }
            crate::substitute::Segment::Call { prim, args, line } => {
                let Some(fragment) = fragments.get(prim) else {
                    return Err(crate::substitute::missing_fragment(line, prim, fragments));
                };
                let body = substitute::splice(fragment, &args, line)?;
                let call_args = args.iter().map(CallArg::from_ref).collect();
                nodes.push(Node::Prim(PrimCall {
                    prim,
                    args: call_args,
                    body,
                }));
            }
        }
    }
    Ok(nodes)
}

/// 把节点序列打印回源文本，逐字节复刻 [`substitute::render`] 的输出。
///
/// 原语片段是语句块，模板已提供 `;`：这里去掉重复终结符，并按调用点当前缩进对齐多行片段。
#[must_use]
pub fn render(nodes: &[Node]) -> String {
    let mut text = String::new();
    for node in nodes {
        match node {
            Node::Text(part) => text.push_str(part),
            Node::Slot(value) => text.push_str(value),
            Node::Prim(call) => {
                let trimmed = call.body.trim_end();
                let trimmed = trimmed.strip_suffix(';').unwrap_or(trimmed);
                let indent: String = text
                    .rsplit('\n')
                    .next()
                    .map(|prefix| " ".repeat(prefix.chars().count()))
                    .unwrap_or_default();
                text.push_str(&trimmed.replace('\n', &format!("\n{indent}")));
            }
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
        match node {
            Node::Text(part) => {
                out.push_str("T ");
                out.push_str(&escape(part));
                out.push('\n');
            }
            Node::Slot(value) => {
                out.push_str("S ");
                out.push_str(&escape(value));
                out.push('\n');
            }
            Node::Prim(call) => {
                out.push_str("P ");
                out.push_str(call.prim.as_str());
                out.push(' ');
                for arg in &call.args {
                    match arg {
                        CallArg::Expr(value) => {
                            out.push('e');
                            out.push_str(&escape(value));
                            out.push('|');
                        }
                        CallArg::Named { key, value } => {
                            out.push('n');
                            out.push_str(&escape(key));
                            out.push('=');
                            out.push_str(&escape(value));
                            out.push('|');
                        }
                    }
                }
                out.push(' ');
                out.push_str(&escape(&call.body));
                out.push('\n');
            }
        }
    }
    out
}

/// 转义：反斜杠、换行与字段分隔符各有唯一编码，保证编码可逆且无歧义。
fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '|' => out.push_str("\\p"),
            '=' => out.push_str("\\q"),
            other => out.push(other),
        }
    }
    out
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
            let fragments = prim::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext())
                .expect("片段层");
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
