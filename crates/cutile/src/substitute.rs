//! 职责：片段拼接（README §4.2、§10.1、§10.2 的 ④）。

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
    Ok(crate::ast::render(&crate::ast::build(
        op, combo, fragments,
    )?))
}

/// 把一段片段放进输出：去掉模板已提供的重复 `;`，并按调用点当前缩进对齐多行片段。
pub(crate) fn place(text: &mut String, expanded: &str) {
    let trimmed = expanded.trim_end();
    let trimmed = trimmed.strip_suffix(';').unwrap_or(trimmed);
    // 缩进取自插入点的当前行；单行片段只追加一次，不 `repeat` 缩进、也不走 `replace`。
    let indent = text
        .rsplit('\n')
        .next()
        .map_or(0, |prefix| prefix.chars().count());
    let mut lines = trimmed.split('\n');
    text.push_str(lines.next().unwrap_or_default());
    for line in lines {
        text.push('\n');
        for _ in 0..indent {
            text.push(' ');
        }
        text.push_str(line);
    }
}

/// 片段占位符替换：`$0`..`$9` 是位置参数，`$name` 是命名参数。
pub(crate) fn splice(fragment: &str, args: &[Arg<'_>], line: usize) -> diagnostic::Result<String> {
    let table = Args::new(args);
    let mut out = String::with_capacity(fragment.len());
    let mut rest = fragment;
    while let Some((head, name, tail)) = next_hole(rest) {
        out.push_str(head);
        out.push_str(table.resolve(name, line)?);
        rest = tail;
    }
    out.push_str(rest);
    Ok(out)
}

/// 一段片段里的占位符切成“字面量 + 名字”：位置参数是 `$0`，命名参数是 `$name`。
fn next_hole(text: &str) -> Option<(&str, &str, &str)> {
    let mut from = 0;
    loop {
        let dollar = from + text[from..].find('$')?;
        let start = dollar + 1;
        let len = text[start..]
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if len > 0 {
            return Some((
                &text[..dollar],
                &text[start..start + len],
                &text[start + len..],
            ));
        }
        // 孤立的 `$` 是字面量，从它后面继续找。
        from = start;
    }
}

/// 片段的实参表：位置参数按序号、命名参数按名字，拼接时各查一次。
struct Args<'a> {
    positional: Vec<&'a str>,
    named: BTreeMap<&'a str, &'a str>,
}

impl<'a> Args<'a> {
    fn new(args: &'a [Arg<'a>]) -> Self {
        let mut positional = Vec::new();
        let mut named = BTreeMap::new();
        for arg in args {
            match arg {
                Arg::Expr(text) => positional.push(*text),
                Arg::Named { key, value } => {
                    named.insert(*key, *value);
                }
            }
        }
        Self { positional, named }
    }

    /// 一个占位符的取值；缺参是模板错误，而不是空串。
    fn resolve(&self, name: &str, line: usize) -> diagnostic::Result<&'a str> {
        if let Ok(index) = name.parse::<usize>() {
            return self.positional.get(index).copied().ok_or_else(|| {
                error(
                    line,
                    format!(
                        "片段占用了 `${index}`，但模板只给了 {} 个位置参数",
                        self.positional.len()
                    ),
                )
            });
        }
        self.named
            .get(name)
            .copied()
            .ok_or_else(|| error(line, format!("片段引用了未提供的命名参数 `${name}`")))
    }
}

#[cfg(test)]
#[path = "../tests/substitute_tests.rs"]
mod tests;
