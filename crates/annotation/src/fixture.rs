//! 测试夹具：把"用户敲下的注解文本"变成 token 模型。
//!
//! 生产路径的词法由 `proc_macro` 提供，因此这里不是第二套实现，而是**只有测试才会编译**
//! 的最小词法器：它让用例写成 `name = "lm_head", out = [4096], ops = ["gemv_i4"]`，
//! 而不是几十行 `Token::new(..)`。
//!
//! 真实的 token 流由 `crates/nn` 与 `crates/cli` 上的注解间接覆盖（那些是真的走 `proc_macro`）。
//! 少数与真实词法相关的行为（例如 `0x10` 是一个字面量 token）无法用这里模拟，由手工构造
//! token 的用例单独覆盖。

use crate::syntax::{Bracket, Token};

/// 只覆盖注解里可能出现的 token：标识符、`=`、`,`、`[..]`、字符串字面量、十进制整数。
pub(crate) fn lex(text: &str) -> Vec<Token> {
    let mut stack: Vec<Vec<Token>> = vec![Vec::new()];
    let mut chars = text.chars().peekable();
    while let Some(&current) = chars.peek() {
        match current {
            current if current.is_whitespace() => {
                chars.next();
            }
            '[' => {
                chars.next();
                stack.push(Vec::new());
            }
            ']' => {
                chars.next();
                let inner = stack.pop().expect("夹具文本的方括号必须配平");
                stack
                    .last_mut()
                    .expect("至少有一个外层")
                    .push(Token::Group(Bracket::Square, inner));
            }
            '=' | ',' => {
                chars.next();
                stack.last_mut().expect("外层").push(Token::Punct(current));
            }
            '"' => {
                chars.next();
                let mut body = String::new();
                for next in chars.by_ref() {
                    if next == '"' {
                        break;
                    }
                    body.push(next);
                }
                stack
                    .last_mut()
                    .expect("外层")
                    .push(Token::Literal(format!("\"{body}\"")));
            }
            current if current.is_ascii_digit() => {
                let mut number = String::new();
                while let Some(&digit) = chars.peek() {
                    if digit.is_ascii_digit() {
                        number.push(digit);
                        chars.next();
                    } else {
                        break;
                    }
                }
                stack.last_mut().expect("外层").push(Token::Literal(number));
            }
            current if current.is_alphanumeric() || current == '_' => {
                let mut word = String::new();
                while let Some(&character) = chars.peek() {
                    if character.is_alphanumeric() || character == '_' {
                        word.push(character);
                        chars.next();
                    } else {
                        break;
                    }
                }
                stack.last_mut().expect("外层").push(Token::Ident(word));
            }
            other => {
                chars.next();
                stack.last_mut().expect("外层").push(Token::Punct(other));
            }
        }
    }
    assert_eq!(stack.len(), 1, "夹具文本的方括号必须配平");
    stack.pop().expect("外层")
}
