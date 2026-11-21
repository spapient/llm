//! 圈复杂度与函数长度分析：lizard 1.25 的 Rust 计数口径。
//!
//! 条件 token：关键字 `if/for/while/catch/where`、`&&`/`||`、`?`、let-else 的 `else`，
//! 以及 match 臂（`=>`，首臂不计）。`if` 的 `else` 不重复计数。
//! 复杂度 `ccn = 1 + 条件计数`；函数长度 `nloc` 为函数体范围内 token 覆盖到的行数。
//!
//! 已知限制：`macro_rules!` / `macro` 定义体是生成模板，不是函数，整体跳过——宏臂内的
//! 条件不计入任何函数的复杂度；模板分支的密度目前不在本门口径内。

use std::collections::BTreeMap;

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
struct Range<'a> {
    name: String,
    fn_pos: Pos,
    open: Pos,
    close: Pos,
    body: &'a [Token],
}

/// token 扫描时共享的状态：`if` 体待匹配、`else` 归属。
#[derive(Default)]
struct State {
    body: bool,
    pending: bool,
}

/// 遍历上下文：函数范围与逐函数条件数。
struct Walker<'a> {
    ranges: &'a [Range<'a>],
    out: Vec<u32>,
    state: State,
}

impl Walker<'_> {
    /// 递归 token 树，按 lizard 口径累计条件；位置归属最内层函数。
    fn walk(&mut self, tokens: &[Token]) {
        let mut match_brace = false;
        for (idx, token) in tokens.iter().enumerate() {
            match token {
                Token::Ident { text, pos } => match_brace |= self.ident(text, *pos),
                Token::Punct { ch, pos } => self.punct(tokens, idx, *ch, *pos),
                Token::Group {
                    delim, items, open, ..
                } => match_brace = self.group(delim, items, *open, match_brace),
                Token::Literal { .. } => self.state.pending = false,
            }
        }
    }

    /// 标识符：条件词 +1（`else` 只在 let-else 时计）；返回是否还要吃掉花括号（`match`）。
    fn ident(&mut self, text: &str, pos: Pos) -> bool {
        let pending = std::mem::take(&mut self.state.pending);
        if text == "else" {
            if !pending {
                self.add(pos, 1);
            }
            return false;
        }
        if text == "if" {
            self.state.body = true;
        }
        if matches!(text, "if" | "for" | "while" | "catch" | "where") {
            self.add(pos, 1);
        }
        text == "match"
    }

    /// 标点：`&&`/`||` 与 `?` 各算一个条件。
    fn punct(&mut self, tokens: &[Token], idx: usize, ch: char, pos: Pos) {
        self.state.pending = false;
        if doubled(tokens, idx, ch) || ch == '?' {
            self.add(pos, 1);
        }
    }

    /// 分组：`match` 花括号计臂数（首臂不计），`if` 体决定后续 `else` 归属。
    fn group(&mut self, delim: &Delim, items: &[Token], open: Pos, match_brace: bool) -> bool {
        if *delim != Delim::Brace {
            self.walk(items);
            self.state.pending = false;
            return match_brace;
        }
        let extra = arrows(items).saturating_sub(1);
        if match_brace && extra > 0 {
            self.add(open, extra);
        }
        let closes = self.state.body;
        self.state.body = false;
        self.walk(items);
        self.state.pending = closes;
        false
    }

    /// 把一个条件记到覆盖该位置的最内层函数。
    fn add(&mut self, pos: Pos, extra: u32) {
        if let Some(idx) = holder(self.ranges, pos) {
            self.out[idx] += extra;
        }
    }
}

/// 分析单份源码，返回按源码位置排序的全部函数 finding（同名函数按源码顺序编号）。
pub fn analyze(src: &str) -> diagnostic::Result<Vec<Finding>> {
    let prepared = strip_tests(src)?;
    let tokens = parse(&prepared)?;
    let mut ranges = Vec::new();
    collect(&tokens, &mut ranges);
    ranges.sort_by_key(|r| (r.fn_pos.line, r.fn_pos.col));
    let hits = {
        let mut walker = Walker {
            ranges: &ranges,
            out: vec![0; ranges.len()],
            state: State::default(),
        };
        walker.walk(&tokens);
        walker.out
    };
    let mut seen = BTreeMap::new();
    let findings = ranges
        .iter()
        .enumerate()
        .map(|(idx, range)| Finding {
            name: range.name.clone(),
            nth: *seen
                .entry(range.name.clone())
                .and_modify(|v| *v += 1)
                .or_insert(0),
            start: range.fn_pos.line,
            end: range.close.line,
            ccn: hits[idx] + 1,
            nloc: nloc(range),
        })
        .collect();
    Ok(findings)
}

/// 递归收集函数定义；宏定义体与函数指针类型都不算定义。
fn collect<'a>(tokens: &'a [Token], out: &mut Vec<Range<'a>>) {
    let mut idx = 0;
    while idx < tokens.len() {
        if let Some(abs) = macro_body_end(tokens, idx) {
            idx = abs + 1;
            continue;
        }
        if let Some((range, body_idx)) = function_at(tokens, idx) {
            collect(range.body, out);
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

/// 宏定义体：`macro_rules! name { .. }` 或 `macro name { .. }` 的花括号下标。
fn macro_body_end(tokens: &[Token], idx: usize) -> Option<usize> {
    let Token::Ident { text, .. } = &tokens[idx] else {
        return None;
    };
    if text != "macro_rules" && text != "macro" {
        return None;
    }
    brace_after(tokens, idx + 1)
}

/// `idx` 处的函数定义：范围与函数体在同级序列中的下标。
fn function_at<'a>(tokens: &'a [Token], idx: usize) -> Option<(Range<'a>, usize)> {
    let Token::Ident { text, pos } = &tokens[idx] else {
        return None;
    };
    let Some(Token::Ident { text: name, .. }) = tokens.get(idx + 1) else {
        return None;
    };
    if text != "fn" {
        return None;
    }
    scan_body(tokens, idx + 2, name, *pos)
}

/// 从签名之后找函数体：先遇到分号说明只是声明，先遇到花括号就是函数体。
fn scan_body<'a>(
    tokens: &'a [Token],
    from: usize,
    name: &str,
    fn_pos: Pos,
) -> Option<(Range<'a>, usize)> {
    for (idx, token) in tokens.iter().enumerate().skip(from) {
        match token {
            Token::Group {
                delim: Delim::Brace,
                items,
                open,
                close,
            } => {
                let range = Range {
                    name: name.to_string(),
                    fn_pos,
                    open: *open,
                    close: *close,
                    body: items,
                };
                return Some((range, idx));
            }
            Token::Punct { ch: ';', .. } => return None,
            _ => {}
        }
    }
    None
}

/// 从 `from` 开始第一个花括号组的绝对下标。
fn brace_after(tokens: &[Token], from: usize) -> Option<usize> {
    tokens
        .iter()
        .skip(from)
        .position(is_brace)
        .map(|at| at + from)
}

fn is_brace(token: &Token) -> bool {
    matches!(
        token,
        Token::Group {
            delim: Delim::Brace,
            ..
        }
    )
}

fn doubled(tokens: &[Token], idx: usize, ch: char) -> bool {
    matches!(ch, '&' | '|')
        && matches!(tokens.get(idx + 1), Some(Token::Punct { ch: next, .. }) if *next == ch)
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

/// 位置所属的最内层函数；无函数归属（如顶层 const）的条件不计。
fn holder(ranges: &[Range<'_>], pos: Pos) -> Option<usize> {
    (0..ranges.len())
        .filter(|&at| holds(&ranges[at], pos))
        .max_by_key(|&at| ranges[at].open)
}

fn holds(range: &Range<'_>, pos: Pos) -> bool {
    // 范围从 `fn` 起始：签名里的条件（`?Sized` 的 `?`、`where`）同样归属本函数。
    pos >= range.fn_pos && range.close >= pos
}

/// 函数体范围内 token 覆盖到的行数（空行与纯注释行无 token，自然不计）。
fn nloc(range: &Range<'_>) -> u32 {
    let mut marked = vec![false; range.close.line];
    mark(range.body, &mut marked);
    marked[range.open.line - 1..]
        .iter()
        .filter(|&&flag| flag)
        .count() as u32
}

fn mark(tokens: &[Token], marked: &mut [bool]) {
    for token in tokens {
        for line in token.start_line()..=token.end_line() {
            if let Some(slot) = marked.get_mut(line - 1) {
                *slot = true;
            }
        }
        if let Token::Group { items, .. } = token {
            mark(items, marked);
        }
    }
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
