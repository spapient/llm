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

    let out: Vec<String> = src
        .lines()
        .enumerate()
        .map(|(i, line)| {
            if blank[i] {
                String::new()
            } else {
                line.to_string()
            }
        })
        .collect();
    Ok(out.join("\n"))
}

/// 递归遍历一层 token：标记本层 cfg(test) 条目，并进入分组处理嵌套的 cfg(test)。
fn collect(tokens: &[Token], blank: &mut [bool]) {
    let mut attr_start = None;
    let mut attr_is_test = false;
    let mut idx = 0;
    while idx < tokens.len() {
        if is_attr(tokens, idx) {
            if attr_start.is_none() {
                attr_start = Some(tokens[idx].start_line());
            }
            if let Token::Group { items, .. } = &tokens[idx + 1]
                && is_cfg_test(items)
            {
                attr_is_test = true;
            }
            idx += 2; // 跳过 `#` 与 `[..]`
            continue;
        }

        let end = item_end(tokens, idx);
        let last = &tokens[end];
        if attr_is_test {
            let start = attr_start.unwrap_or_else(|| last.start_line());
            for line in start..=last.end_line() {
                if let Some(flag) = blank.get_mut(line - 1) {
                    *flag = true;
                }
            }
        }
        for token in &tokens[idx..=end] {
            if let Token::Group { items, .. } = token {
                collect(items, blank);
            }
        }
        attr_start = None;
        attr_is_test = false;
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

/// `test` 是否作为独立 token 出现（边界不是标识符字符）。
fn contains_token(body: &str, token: &str) -> bool {
    let mut from = 0;
    while let Some(at) = body[from..].find(token) {
        let start = from + at;
        let end = start + token.len();
        let before = body[..start].chars().next_back();
        let after = body[end..].chars().next();
        let boundary = |ch: Option<char>| ch.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if boundary(before) && boundary(after) {
            return true;
        }
        from = end;
    }
    false
}

/// 把 token 渲染回紧凑文本（字面量保留原文，分组补括号）。
fn render(tokens: &[Token]) -> String {
    let mut out = String::new();
    for token in tokens {
        match token {
            Token::Ident { text, .. } | Token::Literal { text, .. } => out.push_str(text),
            Token::Punct { ch, .. } => out.push(*ch),
            Token::Group { delim, items, .. } => {
                let (open, close) = match delim {
                    Delim::Paren => ('(', ')'),
                    Delim::Brace => ('{', '}'),
                    Delim::Bracket => ('[', ']'),
                };
                out.push(open);
                out.push_str(&render(items));
                out.push(close);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_test_items_with_attached_attributes() {
        let src = "#[cfg(target_os = \"macos\")]\n#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn keep() {}\n";
        let stripped = strip_tests(src).expect("strip");
        assert_eq!(stripped.lines().count(), src.lines().count());
        assert!(!stripped.contains("mod tests"));
        assert!(!stripped.contains("target_os"));
        assert!(stripped.contains("fn keep"));
    }

    #[test]
    fn keeps_not_test_items() {
        let src = "#[cfg(not(test))]\nfn real() {}\n";
        assert!(strip_tests(src).expect("strip").contains("fn real"));
    }

    #[test]
    fn detects_compound_test_cfg() {
        assert!(is_test_body("all(test, feature = \"x\")"));
        assert!(!is_test_body("not(test)"));
        assert!(!is_test_body("feature = \"testing\""));
    }

    #[test]
    fn braces_inside_strings_do_not_break_strip() {
        let src =
            "#[cfg(test)]\nmod tests {\n    let t = \"void f() { {{x}} }\";\n}\nfn keep() {}\n";
        let stripped = strip_tests(src).expect("strip");
        assert!(!stripped.contains("void f"));
        assert!(stripped.contains("fn keep"));
    }
}
