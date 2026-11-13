//! 圈复杂度与函数长度分析：lizard 1.25 的 Rust 计数口径。
//!
//! 条件 token：关键字 `if/for/while/catch/where`、`&&`/`||`、`?`、let-else 的 `else`，
//! 以及 match 臂（`=>`，首臂不计）。`if` 的 `else` 不重复计数。
//! 复杂度 `ccn = 1 + 条件计数`；函数长度 `nloc` 为函数体范围内 token 覆盖到的行数。

use std::collections::{BTreeMap, BTreeSet};

use diagnostic::{Code, Error};

use crate::lex::{Delim, Pos, Token, parse};
use crate::strip::strip_tests;

/// 单函数分析结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// 函数名。
    pub name: String,
    /// 同名函数序号（按源码顺序，0-based）。
    pub nth: usize,
    /// 函数起始行（`fn` 所在行）。
    pub start: usize,
    /// 函数体闭括号所在行。
    pub end: usize,
    /// 圈复杂度。
    pub ccn: u32,
    /// 函数体有效行数。
    pub nloc: u32,
}

/// 函数范围：身份与函数体引用。
#[derive(Debug)]
struct Range {
    name: String,
    fn_pos: Pos,
    open: Pos,
    close: Pos,
    body: Vec<Token>,
}

/// token 扫描状态：`if` 体待匹配、`else` 归属。
#[derive(Default)]
struct State {
    body: bool,
    pending: bool,
}

/// 分析单份源码，返回按源码位置排序的全部函数 finding。
pub fn analyze(src: &str) -> diagnostic::Result<Vec<Finding>> {
    let prepared = strip_tests(src)?;
    let tokens = parse(&prepared)?;
    let mut ranges = Vec::new();
    collect(&tokens, &mut ranges);
    order(&mut ranges);
    let mut hits = vec![0u32; ranges.len()];
    let mut state = State::default();
    walk(&tokens, &ranges, &mut hits, &mut state)?;
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut findings = Vec::new();
    for (idx, range) in ranges.iter().enumerate() {
        let nth = *seen
            .entry(range.name.clone())
            .and_modify(|v| *v += 1)
            .or_insert(0);
        findings.push(Finding {
            name: range.name.clone(),
            nth,
            start: range.fn_pos.line,
            end: range.close.line,
            ccn: hits[idx] + 1,
            nloc: nloc(range),
        });
    }
    Ok(findings)
}

/// 递归收集函数定义；函数指针类型（`fn` 后直接是圆括号组）不算定义。
fn collect(tokens: &[Token], out: &mut Vec<Range>) {
    let mut idx = 0;
    while idx < tokens.len() {
        // 宏定义体内的 fn 只有展开后才存在（syn 不展开宏），跳过整个宏体不收集。
        if let Token::Ident { text, .. } = &tokens[idx]
            && (text == "macro_rules" || text == "macro")
            && let Some(abs) = brace_after(tokens, idx + 1)
        {
            idx = abs + 1;
            continue;
        }
        if is_fn_keyword(&tokens[idx])
            && matches!(tokens.get(idx + 1), Some(Token::Ident { .. }))
            && let Some((range, body_idx)) = definition(tokens, idx)
        {
            collect(&range.body, out);
            out.push(range);
            idx = body_idx + 1;
            continue;
        }
        if let Token::Group { items, .. } = &tokens[idx] {
            collect(items, out);
        }
        idx += 1;
    }
}

fn is_fn_keyword(token: &Token) -> bool {
    matches!(token, Token::Ident { text, .. } if text == "fn")
}

/// 从 `fn` 位置构造函数范围；返回范围与函数体在同级序列中的下标。
fn definition(tokens: &[Token], at: usize) -> Option<(Range, usize)> {
    let fn_pos = match &tokens[at] {
        Token::Ident { pos, .. } => *pos,
        _ => return None,
    };
    let name = match tokens.get(at + 1) {
        Some(Token::Ident { text, .. }) => text.clone(),
        _ => return None,
    };
    for (idx, token) in tokens.iter().enumerate().skip(at + 2) {
        match token {
            Token::Group {
                delim: Delim::Brace,
                items,
                open,
                close,
                ..
            } => {
                let range = Range {
                    name,
                    fn_pos,
                    open: *open,
                    close: *close,
                    body: items.clone(),
                };
                return Some((range, idx));
            }
            Token::Punct { ch: ';', .. } => return None,
            Token::Group { items, .. } => {
                // 签名组内只可能有函数指针类型，递归不会产生定义。
                let mut nested = Vec::new();
                collect(items, &mut nested);
            }
            Token::Ident { .. } | Token::Punct { .. } | Token::Literal { .. } => {}
        }
    }
    None
}

fn order(ranges: &mut [Range]) {
    ranges.sort_by_key(|r| (r.fn_pos.line, r.fn_pos.col));
}

/// 递归 token 树，按 lizard 口径累计条件；位置归属最内层函数。
fn walk(
    tokens: &[Token],
    ranges: &[Range],
    out: &mut [u32],
    state: &mut State,
) -> diagnostic::Result<()> {
    let mut idx = 0;
    let mut match_brace = false;
    while idx < tokens.len() {
        match &tokens[idx] {
            Token::Ident { text, pos } => {
                if text == "else" {
                    if !state.pending {
                        bump(ranges, out, *pos);
                    }
                    state.pending = false;
                } else {
                    state.pending = false;
                    if is_word(text) {
                        bump(ranges, out, *pos);
                    }
                    if text == "if" {
                        state.body = true;
                    }
                    if text == "match" {
                        match_brace = true;
                    }
                }
            }
            Token::Punct { ch, pos } => {
                state.pending = false;
                if doubled(tokens, idx, *ch) || *ch == '?' {
                    bump(ranges, out, *pos);
                }
            }
            Token::Group {
                delim, items, open, ..
            } => {
                let is_brace = *delim == Delim::Brace;
                if is_brace && match_brace {
                    let arrows = arrows(items);
                    if arrows > 1 {
                        add(ranges, out, *open, arrows - 1);
                    }
                    match_brace = false;
                }
                let closes = is_brace && state.body;
                if is_brace {
                    state.body = false;
                }
                walk(items, ranges, out, state)?;
                state.pending = closes;
            }
            Token::Literal { .. } => {
                state.pending = false;
            }
        }
        idx += 1;
    }
    Ok(())
}

fn is_word(word: &str) -> bool {
    matches!(word, "if" | "for" | "while" | "catch" | "where")
}

/// 从 `from` 开始第一个花括号组的绝对下标。
fn brace_after(tokens: &[Token], from: usize) -> Option<usize> {
    tokens
        .iter()
        .enumerate()
        .skip(from)
        .find_map(|(idx, token)| {
            matches!(
                token,
                Token::Group {
                    delim: Delim::Brace,
                    ..
                }
            )
            .then_some(idx)
        })
}

fn doubled(tokens: &[Token], idx: usize, ch: char) -> bool {
    if !matches!(ch, '&' | '|') {
        return false;
    }
    matches!(tokens.get(idx + 1), Some(Token::Punct { ch: next, .. }) if *next == ch)
}

/// 组内直接 `=>` 箭头数（不递归嵌套组：嵌套组是单个 token，内部箭头由它自己的 match 计数）。
fn arrows(tokens: &[Token]) -> u32 {
    let mut count = 0;
    let mut idx = 0;
    while idx + 1 < tokens.len() {
        if matches!(tokens[idx], Token::Punct { ch: '=', .. })
            && matches!(tokens[idx + 1], Token::Punct { ch: '>', .. })
        {
            count += 1;
            idx += 2;
        } else {
            idx += 1;
        }
    }
    count
}

fn add(ranges: &[Range], out: &mut [u32], pos: Pos, extra: u32) {
    if let Some(idx) = holder(ranges, pos) {
        out[idx] += extra;
    }
}

fn bump(ranges: &[Range], out: &mut [u32], pos: Pos) {
    add(ranges, out, pos, 1)
}

/// 位置所属的最内层函数；无函数归属（如顶层 const）的条件不计。
fn holder(ranges: &[Range], pos: Pos) -> Option<usize> {
    ranges
        .iter()
        .enumerate()
        .filter(|(_, r)| holds(r, pos))
        .map(|(idx, _)| idx)
        .max_by_key(|&idx| (ranges[idx].open.line, ranges[idx].open.col))
}

fn holds(range: &Range, pos: Pos) -> bool {
    // 范围从 `fn` 起始：签名里的条件（`?Sized` 的 `?`、`where`）同样归属本函数。
    ge(pos, range.fn_pos) && ge(range.close, pos)
}

fn ge(a: Pos, b: Pos) -> bool {
    a.line > b.line || (a.line == b.line && a.col >= b.col)
}

/// 函数体范围内 token 覆盖到的行数（空行与纯注释行无 token，自然不计）。
fn nloc(range: &Range) -> u32 {
    let mut lines = BTreeSet::new();
    mark(&range.body, &mut lines);
    lines.range(range.open.line..=range.close.line).count() as u32
}

fn mark(tokens: &[Token], lines: &mut BTreeSet<usize>) {
    for token in tokens {
        for line in token.start_line()..=token.end_line() {
            lines.insert(line);
        }
        if let Token::Group { items, .. } = token {
            mark(items, lines);
        }
    }
}

/// 内部不变量失败时的统一构造。
pub fn error(message: impl Into<String>) -> Error {
    Error::new(Code::Internal, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(src: &str) -> Vec<(String, u32)> {
        analyze(src)
            .expect("analyze")
            .into_iter()
            .map(|f| (f.name, f.ccn))
            .collect()
    }

    #[test]
    fn counts_conditions_like_lizard() {
        let src = r#"
fn branch(a: bool, b: bool) -> u32 {
    if a && b { 1 } else { 2 }
}
fn choose(x: Option<u32>) -> u32 {
    match x {
        Some(0) => 0,
        Some(v) => v,
        None => 1,
    }
}
fn fallible(v: Option<u32>) -> Result<u32, ()> {
    let Some(x) = v else { return Err(()) };
    Ok(x?)
}
"#;
        assert_eq!(
            cc(src),
            vec![
                ("branch".into(), 3),
                ("choose".into(), 3),
                ("fallible".into(), 3)
            ]
        );
    }

    #[test]
    fn function_pointer_types_are_not_definitions() {
        let src = "fn real(f: fn(u8) -> u8) -> u8 { f(1) }";
        let findings = analyze(src).expect("analyze");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].name, "real");
    }

    #[test]
    fn nested_functions_keep_their_own_complexity() {
        let src = "fn outer() { fn inner(a: bool) { if a { } } }";
        let by_name: BTreeMap<String, u32> = analyze(src)
            .expect("analyze")
            .into_iter()
            .map(|f| (f.name, f.ccn))
            .collect();
        assert_eq!(by_name["inner"], 2);
        assert_eq!(by_name["outer"], 1);
    }

    #[test]
    fn two_top_level_functions_are_both_collected() {
        let src = "fn first() {}\nfn second() {}\n";
        assert_eq!(cc(src).len(), 2);
    }
}
