//! 扫描前预处理：剔除 `#[cfg(test)]` 条目并保留行号。
//!
//! 基于 [`crate::lex`] token 树定位条目，而不是逐行匹配大括号：字符串/原始字符串/字符/注释里的
//! `{`、`}` 不会被当成真实括号。`#[cfg(test)]` 与 `#[cfg(all(.. test ..))]` 标注的整个条目
//! 被空行替换（行号不变），`#[cfg(not(test))]` 不算。

use crate::lex::{Delim, Token, parse};
use diagnostic::Result;

/// 剔除 cfg(test) 条目；源码无法解析返回 invalid 错误。
pub fn strip_tests(src: &str) -> Result<String> {
    let tokens = parse(src)?;
    let mut blank = vec![false; src.lines().count()];
    collect(&tokens, &mut blank);

    let out: Vec<&str> = src
        .lines()
        .enumerate()
        .map(|(at, line)| if blank[at] { "" } else { line })
        .collect();
    Ok(out.join("\n"))
}

/// 一层里的属性归属：首个属性的起始行与是否 `cfg(test)`。
#[derive(Default)]
struct Attr {
    start: Option<usize>,
    is_test: bool,
}

impl Attr {
    /// 吸收一个属性：首个属性的起始行与 `cfg(test)` 口径。
    fn absorb(&mut self, hash: &Token, bracket: &Token) {
        self.start = self.start.or(Some(hash.start_line()));
        if let Token::Group { items, .. } = bracket {
            self.is_test |= is_cfg_test(items);
        }
    }

    /// 口径为 test 时，把条目覆盖的行置空。
    fn blank_if_test(&self, last: &Token, blank: &mut [bool]) {
        if !self.is_test {
            return;
        }
        let from = self.start.unwrap_or(last.start_line());
        for line in from..=last.end_line() {
            if let Some(flag) = blank.get_mut(line - 1) {
                *flag = true;
            }
        }
    }
}

/// 递归遍历一层 token：标记本层 cfg(test) 条目，并进入分组处理嵌套的 cfg(test)。
fn collect(tokens: &[Token], blank: &mut [bool]) {
    let mut attr = Attr::default();
    let mut idx = 0;
    while idx < tokens.len() {
        if is_attr(tokens, idx) {
            attr.absorb(&tokens[idx], &tokens[idx + 1]);
            idx += 2; // 跳过 `#` 与 `[..]`
            continue;
        }

        let end = item_end(tokens, idx);
        attr.blank_if_test(&tokens[end], blank);
        for token in &tokens[idx..=end] {
            if let Token::Group { items, .. } = token {
                collect(items, blank);
            }
        }
        attr = Attr::default();
        idx = end + 1;
    }
}

/// 顶层位置 `idx` 是否是属性：`#` 后跟一个方括号分组。
fn is_attr(tokens: &[Token], idx: usize) -> bool {
    matches!(tokens.get(idx), Some(Token::Punct { ch: '#', .. }))
        && matches!(
            tokens.get(idx + 1),
            Some(Token::Group {
                delim: Delim::Bracket,
                ..
            })
        )
}

/// 从条目首个 token 起，定位其结束 token：带体的条目结束于花括号分组，否则结束于分号。
fn item_end(tokens: &[Token], first: usize) -> usize {
    let mut idx = first;
    while idx < tokens.len() {
        match &tokens[idx] {
            Token::Group {
                delim: Delim::Brace,
                ..
            }
            | Token::Punct { ch: ';', .. } => return idx,
            _ => idx += 1,
        }
    }
    first
}

/// 方括号分组内容是否是 `cfg(..)` 且口径为 test。
fn is_cfg_test(items: &[Token]) -> bool {
    let mut cursor = 0;
    while cursor + 1 < items.len() {
        if let Token::Ident { text, .. } = &items[cursor]
            && text == "cfg"
            && let Token::Group {
                delim: Delim::Paren,
                items: body,
                ..
            } = &items[cursor + 1]
        {
            return is_test_body(&render(body));
        }
        cursor += 1;
    }
    false
}

/// 口径与旧实现一致：只认纯 `test` 与 `all(.. test ..)`；`not(test)` 与 `any(..)` 不剔除。
fn is_test_body(body: &str) -> bool {
    let compact: String = body.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact.contains("not(test)") {
        return false;
    }
    compact == "test" || (compact.starts_with("all(") && contains_token(&compact, "test"))
}

/// `test` 是否作为独立 token 出现（两侧边界都不是标识符字符）。
fn contains_token(body: &str, token: &str) -> bool {
    let boundary = |ch: Option<char>| ch.is_none_or(|c| !c.is_alphanumeric() && c != '_');
    body.match_indices(token).any(|(at, _)| {
        let before = body[..at].chars().next_back();
        let after = body[at + token.len()..].chars().next();
        boundary(before) && boundary(after)
    })
}

/// 把 token 渲染回紧凑文本（字面量保留原文，分组补括号）。
fn render(tokens: &[Token]) -> String {
    tokens.iter().map(piece).collect()
}

/// 单个 token 的紧凑文本。
fn piece(token: &Token) -> String {
    match token {
        Token::Ident { text, .. } | Token::Literal { text, .. } => text.clone(),
        Token::Punct { ch, .. } => ch.to_string(),
        Token::Group { delim, items, .. } => {
            format!("{}{}{}", delim.open(), render(items), delim.close())
        }
    }
}

#[cfg(test)]
#[path = "strip_tests.rs"]
mod tests;
