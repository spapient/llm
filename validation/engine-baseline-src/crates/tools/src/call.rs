//! 禁止调用形态：生产代码里的裸 `unwrap`/`expect` 与 panic 宏。

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
        out.extend(ufcs_call(tokens, idx));
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
    let kind = unwrap_expect(text)?;
    let paren = has_paren(tokens, idx + 2);
    paren.then_some(Call { kind, pos: *pos })
}

/// UFCS 关联函数形态：`Type::unwrap(..)` / `Type::expect(..)`，与点号方法同属裸解包。
fn ufcs_call(tokens: &[Token], idx: usize) -> Option<Call> {
    let (text, pos) = ufcs_target(tokens, idx)?;
    let kind = unwrap_expect(text)?;
    let paren = has_paren(tokens, idx + 4);
    paren.then_some(Call { kind, pos })
}

/// 该下标处是否为圆括号分组；方法与 UFCS 两条形态共用同一种判定。
fn has_paren(tokens: &[Token], idx: usize) -> bool {
    matches!(
        tokens.get(idx),
        Some(Token::Group {
            delim: Delim::Paren,
            ..
        })
    )
}

/// UFCS 前缀 `Type::method`：返回方法名与位置；前缀不成立则 `None`。
fn ufcs_target(tokens: &[Token], idx: usize) -> Option<(&str, Pos)> {
    let [
        Some(Token::Ident { .. }),
        Some(Token::Punct { ch: ':', .. }),
        Some(Token::Punct { ch: ':', .. }),
        Some(Token::Ident { text, pos }),
    ] = [
        tokens.get(idx),
        tokens.get(idx + 1),
        tokens.get(idx + 2),
        tokens.get(idx + 3),
    ]
    else {
        return None;
    };
    Some((text.as_str(), *pos))
}

/// 解包方法名 → 对应禁止形态；其余名字不是裸解包。
fn unwrap_expect(text: &str) -> Option<Kind> {
    match text {
        "unwrap" => Some(Kind::Unwrap),
        "expect" => Some(Kind::Expect),
        _ => None,
    }
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
