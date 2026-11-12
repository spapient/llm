//! 模板词法：把 `<name>.k` 切成文本、`{{槽位}}` 与 `@原语(...)`（DESIGN §4.2）。
//!
//! 词法属于声明层：它解释的是 `ops/` 里的文本格式，声明解析（`OpDecl::check_slots`）要用它提前
//! 拦住 `{{GMAXX}}` 这类笔误。片段拼接（`@原语` → 片段文本）才是机制，留在 `cutile`。
//!
//! 位置信息只记行号——这是"诊断精度换解析器规模"的刻意取舍。

use diagnostic::{Code, Error};
use prof::atom::Prim;

/// 模板片段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// 原样保留的文本。
    Text(String),
    /// `{{槽位}}`。
    Slot {
        /// 槽位名。
        name: String,
        /// 行号。
        line: usize,
    },
    /// `@原语(参数)`。
    Call {
        /// 原语。
        prim: Prim,
        /// 参数。
        args: Vec<Arg>,
        /// 行号。
        line: usize,
    },
}

/// `@原语` 的一个参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arg {
    /// 位置参数，原样拼入片段。
    Expr(String),
    /// 命名参数（`along: BN`、`op: sum`），作为 `$key` 占位符的值。
    Named {
        /// 键。
        key: String,
        /// 值。
        value: String,
    },
}

/// 模板拼接错误（只给行号，不给列）。
///
/// 模板是输入，所以类别固定为 [`Code::Invalid`]；行号作为一层上下文，文件由调用方补。
pub fn error(line: usize, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("第 {line} 行"))
}

/// 列出模板里出现的全部槽位（声明层提前校验用）。
pub fn slots(template: &str) -> Vec<(usize, String)> {
    lex(template)
        .into_iter()
        .filter_map(|segment| match segment {
            Segment::Slot { name, line } => Some((line, name)),
            _ => None,
        })
        .collect()
}

/// 词法：把模板切成 [`Segment`]。
pub fn lex(template: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut text = String::new();
    let mut line = 1usize;
    let mut chars = template.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        if ch == '\n' {
            line += 1;
        }
        if ch == '{'
            && template[index..].starts_with("{{")
            && let Some(end) = template[index + 2..].find("}}")
        {
            let name = template[index + 2..index + 2 + end].trim().to_string();
            if !text.is_empty() {
                segments.push(Segment::Text(std::mem::take(&mut text)));
            }
            segments.push(Segment::Slot { name, line });
            let consumed = 2 + end + 2;
            for _ in 0..consumed - 1 {
                chars.next();
            }
            continue;
        }
        if ch == '@' {
            let rest = &template[index + 1..];
            let name_len = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .count();
            if name_len > 0 && rest[name_len..].starts_with('(') {
                let prim_name = &rest[..name_len];
                if let Some(end) = matching_paren(&rest[name_len..])
                    && let Some(prim) = Prim::parse(prim_name)
                {
                    if !text.is_empty() {
                        segments.push(Segment::Text(std::mem::take(&mut text)));
                    }
                    let inner = &rest[name_len + 1..name_len + end];
                    segments.push(Segment::Call {
                        prim,
                        args: parse_args(inner),
                        line,
                    });
                    let consumed = 1 + name_len + end + 1;
                    for _ in 0..consumed - 1 {
                        chars.next();
                    }
                    continue;
                }
            }
        }
        text.push(ch);
    }
    if !text.is_empty() {
        segments.push(Segment::Text(text));
    }
    segments
}

fn matching_paren(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_args(inner: &str) -> Vec<Arg> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut pieces = Vec::new();
    for ch in inner.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if depth == 0 => pieces.push(std::mem::take(&mut current)),
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        pieces.push(current);
    }
    for piece in pieces {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim();
            if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                args.push(Arg::Named {
                    key: key.to_string(),
                    value: value.trim().to_string(),
                });
                continue;
            }
        }
        args.push(Arg::Expr(trimmed.to_string()));
    }
    args
}
