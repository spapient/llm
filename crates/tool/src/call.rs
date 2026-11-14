//! 禁止调用形态：生产代码里的裸 `unwrap`/`expect` 与 panic 宏。
//!
//! 纯函数：基于 [`crate::lex`] token 精确匹配，字符串/注释里的 `.unwrap()` 不会误报。
//! 命名 snake_case 与 `todo!`/`unimplemented!` 已由编译器与 workspace lints 强制，这里不重复。
//!
//! `unreachable!` 必须由本模块兜住：它是 panic 路径，却没有任何 lint 覆盖——`todo`/
//! `unimplemented` 被 `clippy` 拒了，`unreachable` 没有。「零 panic」这句话少这一条就只是
//! 「零一部分 panic」。

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
    /// `unreachable!(..)`。
    Unreachable,
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

/// 递归扫描 token 树，收集禁止形态。
fn walk(tokens: &[Token], out: &mut Vec<Call>) {
    for (idx, token) in tokens.iter().enumerate() {
        out.extend(method_call(tokens, idx));
        out.extend(macro_call(tokens, idx));
        if let Token::Group { items, .. } = token {
            walk(items, out);
        }
    }
}

/// `.unwrap()` / `.expect(..)` 形态。
fn method_call(tokens: &[Token], idx: usize) -> Option<Call> {
    let Token::Punct { ch: '.', pos } = &tokens[idx] else {
        return None;
    };
    let Some(Token::Ident { text, .. }) = tokens.get(idx + 1) else {
        return None;
    };
    let kind = match text.as_str() {
        "unwrap" => Kind::Unwrap,
        "expect" => Kind::Expect,
        _ => return None,
    };
    let paren = matches!(
        tokens.get(idx + 2),
        Some(Token::Group {
            delim: Delim::Paren,
            ..
        })
    );
    paren.then_some(Call { kind, pos: *pos })
}

/// panic 类宏形态；其它宏（如 `write!`/`println!`）不算。
fn macro_call(tokens: &[Token], idx: usize) -> Option<Call> {
    let Token::Ident { text, pos } = &tokens[idx] else {
        return None;
    };
    let kind = match text.as_str() {
        "panic" => Kind::Panic,
        "unreachable" => Kind::Unreachable,
        _ => return None,
    };
    let bang = matches!(tokens.get(idx + 1), Some(Token::Punct { ch: '!', .. }));
    bang.then_some(Call { kind, pos: *pos })
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

    /// `unreachable!` 是 panic 路径但没有 lint 覆盖：漏掉它，「零 panic」就只覆盖一部分。
    #[test]
    fn unreachable_is_caught_but_its_plain_identifier_is_not() {
        let src = "fn f() { let unreachable = 1; g(unreachable); unreachable!(\"x\"); }";
        let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![Kind::Unreachable], "只有带 `!` 的那次是宏调用");
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
