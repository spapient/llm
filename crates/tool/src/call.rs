//! 禁止调用形态：生产代码里的裸 `unwrap`/`expect` 与 panic 宏。
//!
//! 纯函数：基于 [`crate::lex`] token 精确匹配，字符串/注释里的 `.unwrap()` 不会误报。
//! 命名 snake_case 与 `todo!`/`unimplemented!` 已由编译器与 workspace lints 强制，这里不重复。

use crate::lex::{Delim, Pos, Token, parse};

/// 禁止形态类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 裸 `.unwrap()`。
    Unwrap,
    /// 裸 `.expect(..)`。
    Expect,
    /// `panic!(..)`。
    Panic,
}

/// 一处禁止形态：类别与位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Call {
    /// 类别。
    pub kind: Kind,
    /// 起始位置。
    pub pos: Pos,
}

/// 扫描全部禁止形态。
pub fn scan(text: &str) -> diagnostic::Result<Vec<Call>> {
    let tokens = parse(text)?;
    let mut out = Vec::new();
    walk(&tokens, &mut out);
    Ok(out)
}

fn walk(tokens: &[Token], out: &mut Vec<Call>) {
    let mut idx = 0;
    while idx < tokens.len() {
        if let Token::Punct { ch: '.', pos } = tokens[idx]
            && let Some(Token::Ident { text, .. }) = tokens.get(idx + 1)
            && let Some(Token::Group {
                delim: Delim::Paren,
                ..
            }) = tokens.get(idx + 2)
        {
            let kind = match text.as_str() {
                "unwrap" => Some(Kind::Unwrap),
                "expect" => Some(Kind::Expect),
                _ => None,
            };
            if let Some(kind) = kind {
                out.push(Call { kind, pos });
            }
        }
        if let Token::Ident { text, pos } = &tokens[idx]
            && text == "panic"
            && matches!(tokens.get(idx + 1), Some(Token::Punct { ch: '!', .. }))
        {
            out.push(Call {
                kind: Kind::Panic,
                pos: *pos,
            });
        }
        if let Token::Group { items, .. } = &tokens[idx] {
            walk(items, out);
        }
        idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_unwrap_expect_and_panic_are_caught() {
        let src = "fn f() { let a = x.unwrap(); let b = y.expect(\"m\"); panic!(\"p\"); }";
        let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect, Kind::Panic]);
    }

    #[test]
    fn unwrap_inside_a_string_is_not_flagged() {
        let src = "let s = \"x.unwrap()\";";
        assert!(scan(src).unwrap().is_empty());
    }

    #[test]
    fn result_returning_code_is_clean() {
        let src = "fn f() -> Result<u32, E> { let v = ok()?; Ok(v) }";
        assert!(scan(src).unwrap().is_empty());
    }
}
