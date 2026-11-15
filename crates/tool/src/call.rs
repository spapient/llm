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
mod tests;
