//! 片段拼接（DESIGN §4.2、§10.1、§10.2 的 ④）。
//!
//! 词法（`{{槽位}}` / `@原语(...)` 怎么切）属于声明层，在 [`op::tpl`]；本模块只把切出来的片段
//! 换成取值与片段文本。

use crate::decl::{OpDecl, TileCombo};
use std::collections::BTreeMap;

/// 词法与行号唯一的错误构造器直接转发：词法只有一份，声明解析与拼接看同一套 `{{…}}` 规则。
pub use ::op::tpl::{Arg, Segment, error, lex};

// 片段层的事实源在 `prim` crate（`<vendor>/<prim>.<ext>`）：本模块只做拼接，不持有片段。
pub use prim::FragmentSet;

/// 片段层缺少 `prim` 时的统一错误：ast 展开与本模块拼接都走这条，免得两处各写一遍格式串。
pub(crate) fn missing_fragment(
    line: usize,
    prim: crate::atom::Prim,
    fragments: &FragmentSet,
) -> diagnostic::Error {
    error(
        line,
        format!(
            "片段层缺少 `{}/{}.{}`：该设备尚未实现 `{prim}`",
            fragments.vendor(),
            prim,
            fragments.ext()
        ),
    )
}

/// ④ substitute：模板槽位 ← 轴/布局取值，`@原语(...)` ← 片段文本。
pub fn render(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
) -> diagnostic::Result<String> {
    let mut text = String::new();

    for segment in lex(&op.template) {
        match segment {
            Segment::Text(part) => text.push_str(part),
            Segment::Slot { name, line } => {
                if name == "op" {
                    text.push_str(&op.name);
                } else if let Some(value) = combo.get(name) {
                    text.push_str(&value.to_string());
                } else if let Some(spec) = op.layout(name) {
                    text.push_str(&spec.as_str());
                } else {
                    return Err(error(
                        line,
                        format!("槽位 `{{{{{name}}}}}` 既不是轴也不是布局槽位"),
                    ));
                }
            }
            Segment::Call { prim, args, line } => {
                let Some(fragment) = fragments.get(prim) else {
                    return Err(missing_fragment(line, prim, fragments));
                };
                let expanded = splice(fragment, &args, line)?;
                place(&mut text, &expanded);
            }
        }
    }

    Ok(text)
}

/// 把一段片段放进输出：去掉模板已提供的重复 `;`，并按调用点当前缩进对齐多行片段。
///
/// 字符串路径（[`render`]）与结构化路径（`ast::render`）共用这一份：两条路径的产物逐字节相同
/// 由 `ast_renders_byte_identical_text` 守着，但规则本身只该写一次。
pub(crate) fn place(text: &mut String, expanded: &str) {
    let trimmed = expanded.trim_end();
    let trimmed = trimmed.strip_suffix(';').unwrap_or(trimmed);
    let indent: String = text
        .rsplit('\n')
        .next()
        .map(|prefix| " ".repeat(prefix.chars().count()))
        .unwrap_or_default();
    text.push_str(&trimmed.replace('\n', &format!("\n{indent}")));
}

/// 片段占位符替换：`$0`..`$9` 是位置参数，`$name` 是命名参数。
///
/// 词法占位符的替换规则只有这一份：字符串路径（[`render`]）与结构化路径（`ast::build`）
/// 都调用它，保证两条路径对同一片段得到完全相同的展开。
pub(crate) fn splice(fragment: &str, args: &[Arg<'_>], line: usize) -> diagnostic::Result<String> {
    let positional: Vec<&str> = args
        .iter()
        .filter_map(|arg| match arg {
            Arg::Expr(text) => Some(*text),
            Arg::Named { .. } => None,
        })
        .collect();
    let named: BTreeMap<&str, &str> = args
        .iter()
        .filter_map(|arg| match arg {
            Arg::Named { key, value } => Some((*key, *value)),
            Arg::Expr(_) => None,
        })
        .collect();

    let mut out = String::with_capacity(fragment.len());
    let mut chars = fragment.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            out.push(ch);
            continue;
        }
        let mut name = String::new();
        while let Some(next) = chars.peek() {
            if next.is_ascii_alphanumeric() || *next == '_' {
                name.push(*next);
                chars.next();
            } else {
                break;
            }
        }
        if name.is_empty() {
            out.push('$');
            continue;
        }
        if let Ok(index) = name.parse::<usize>() {
            let Some(value) = positional.get(index) else {
                return Err(error(
                    line,
                    format!(
                        "片段占用了 `${index}`，但模板只给了 {} 个位置参数",
                        positional.len()
                    ),
                ));
            };
            out.push_str(value);
            continue;
        }
        let Some(value) = named.get(name.as_str()) else {
            return Err(error(line, format!("片段引用了未提供的命名参数 `${name}`")));
        };
        out.push_str(value);
    }
    Ok(out)
}
