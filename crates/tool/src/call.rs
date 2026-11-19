//! 禁止调用形态：生产代码里的裸 `unwrap`/`expect` 与 panic 宏。
//!
//! 纯函数：基于 [`crate::lex`] token 精确匹配，字符串/注释里的 `.unwrap()` 不会误报。
//! 点号方法形态与 UFCS 关联函数形态（`Type::unwrap(x)`）都认。
//! 命名 snake_case 与 `todo!`/`unimplemented!` 已由编译器与 workspace lints 强制，这里不重复。
//!
//! `unreachable!` 必须由本模块兜住：它是 panic 路径，却没有任何 lint 覆盖——`todo`/
//! `unimplemented` 被 `clippy` 拒了，`unreachable` 没有。「零 panic」这句话少这一条就只是
//! 「零一部分 panic」。
//!
//! `assert!`/`assert_eq!` 不在本门形态内：它们表达逻辑不变量，测试代码先由 `strip_tests`
//! 剔除；生产路径用断言兜底不可信输入属评审纪律，不由本扫描器判定。

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
fn has_paren(tokens: &[Token], index: usize) -> bool {
    matches!(
        tokens.get(index),
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
fn ufcs_unwrap_expect_are_caught() {
    let src = "fn f() { let a = Option::unwrap(x); let b = Result::expect(y, \"m\"); }";
    let kinds: Vec<Kind> = scan(src).unwrap().iter().map(|c| c.kind).collect();
    assert_eq!(kinds, vec![Kind::Unwrap, Kind::Expect]);
}

#[test]
fn ordinary_associated_function_is_not_flagged() {
    let src = "fn f() { let v = Type::new(x); }";
    assert!(scan(src).unwrap().is_empty());
}

#[test]
fn result_returning_code_is_clean() {
    let src = "fn f() -> Result<u32, E> { let v = ok()?; Ok(v) }";
    assert!(scan(src).unwrap().is_empty());
}
}

