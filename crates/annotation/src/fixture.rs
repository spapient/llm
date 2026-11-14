//! 测试夹具：把"用户敲下的注解文本"变成 token 模型。
//!
//! 生产路径的词法由 `proc_macro` 提供，因此这里不是第二套实现，而是**只有测试才会编译**的最小
//! 词法器：它让用例写成 `name = "lm_head", out = [4096], ops = ["gemv_i4"]`，而不是几十行
//! `Token::new(..)`。
//!
//! 真实的 token 流由 `crates/nn` 与 `crates/cli` 上的注解间接覆盖（那些是真的走 `proc_macro`）。
//! 少数与真实词法相关的行为（例如 `0x10` 是一个字面量 token）无法用这里模拟，由手工构造 token
//! 的用例单独覆盖。

use std::iter::Peekable;
use std::str::Chars;

use crate::syntax::{Bracket, Token};

/// 只覆盖注解里可能出现的 token：标识符、`=`、`,`、`[..]`、字符串字面量、十进制整数。
///
/// 括号配平由末尾的断言兜底：夹具文本不平衡是用例写错了，宁可让测试炸掉，也不要把半个 token 流
/// 喂给文法层。
pub(crate) fn lex(text: &str) -> Vec<Token> {
    let mut groups: Vec<Vec<Token>> = vec![Vec::new()];
    let mut chars = text.chars().peekable();
    while let Some(&current) = chars.peek() {
        scan(&mut groups, &mut chars, current);
    }
    assert_eq!(groups.len(), 1, "夹具文本的方括号必须配平");
    let Some(tokens) = groups.pop() else {
        return Vec::new();
    };
    tokens
}

/// 一个 token 的开始：空白跳过，方括号改层次，其余交给 [`item`]。
fn scan(groups: &mut Vec<Vec<Token>>, chars: &mut Peekable<Chars<'_>>, current: char) {
    if current.is_whitespace() {
        chars.next();
        return;
    }
    if current == '[' {
        chars.next();
        groups.push(Vec::new());
        return;
    }
    if current == ']' {
        chars.next();
        return close(groups);
    }
    item(groups, chars, current);
}

/// `[` 之后的那些 token 种类：字符串、单字符标点、数字、标识符。
fn item(groups: &mut [Vec<Token>], chars: &mut Peekable<Chars<'_>>, current: char) {
    if current == '"' {
        return quoted(groups, chars);
    }
    if matches!(current, '=' | ',') {
        chars.next();
        return push(groups, Token::Punct(current));
    }
    if current.is_ascii_digit() {
        return push(
            groups,
            Token::Literal(take(chars, |digit| digit.is_ascii_digit())),
        );
    }
    if is_word(current) {
        return push(groups, Token::Ident(take(chars, is_word)));
    }
    chars.next();
    push(groups, Token::Punct(current));
}

/// `]`：把刚收完的那一层折成一个方括号组交给外层；缺外层说明夹具文本不配平，留给断言。
fn close(groups: &mut Vec<Vec<Token>>) {
    let Some(inner) = groups.pop() else {
        return;
    };
    push(groups, Token::Group(Bracket::Square, inner));
}

/// `".."`：夹具里的字符串一律是"普通字面量"，转义留给手工构造的 token 用例。
fn quoted(groups: &mut [Vec<Token>], chars: &mut Peekable<Chars<'_>>) {
    chars.next();
    let body = take(chars, |character| character != '"');
    chars.next();
    push(groups, Token::Literal(format!("\"{body}\"")));
}

/// 取出满足 `keep` 的前缀，不消耗第一个不满足的字符。
fn take(chars: &mut Peekable<Chars<'_>>, keep: fn(char) -> bool) -> String {
    let mut text = String::new();
    while let Some(&current) = chars.peek() {
        if !keep(current) {
            return text;
        }
        text.push(current);
        chars.next();
    }
    text
}

/// 标识符字符：注解里的键与名字都是"字母数字加下划线"。
fn is_word(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// 把 token 放进最内层分组；配平由 [`lex`] 末尾的断言兜底。
fn push(groups: &mut [Vec<Token>], token: Token) {
    if let Some(group) = groups.last_mut() {
        group.push(token);
    }
}
