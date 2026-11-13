//! 零依赖 Rust 词法器：把源码切成带分组与位置的 token 树。
//!
//! 不依赖 syn/proc-macro2：注释（含嵌套块注释）、字符串、原始字符串、字符字面量在词法层
//! 正确剔除，因此分析器不会把字符串里的 `?` 或注释里的 `if` 误计为条件。
//! 位置从 1 起算（行、列），对齐 proc-macro2 的 `LineColumn` 口径，便于与旧实现对拍。

use diagnostic::{Code, Error};

/// 源码位置：行与列均从 1 开始。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pos {
    /// 行号（1-based）。
    pub line: usize,
    /// 列号（1-based）。
    pub col: usize,
}

impl Pos {
    const fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

/// 分组括号闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delim {
    /// 圆括号 `( )`。
    Paren,
    /// 花括号 `{ }`。
    Brace,
    /// 方括号 `[ ]`。
    Bracket,
}

/// 一个 token：标识符、标点、字面量或分组。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// 标识符或关键字（词法层不区分，分析器按文本判定）。
    Ident {
        /// 标识符文本。
        text: String,
        /// 起点。
        pos: Pos,
    },
    /// 单字符标点。
    Punct {
        /// 标点字符。
        ch: char,
        /// 起点。
        pos: Pos,
    },
    /// 字面量（数字、字符串、字符、原始字符串），整体一个 token，内部不参与分析。
    Literal {
        /// 字面量原文。
        text: String,
        /// 起点。
        start: Pos,
        /// 终点（最后一个字符的位置）。
        end: Pos,
    },
    /// 括号分组，内部保持 token 顺序与嵌套。
    Group {
        /// 括号种类。
        delim: Delim,
        /// 组内 token。
        items: Vec<Token>,
        /// 开括号位置。
        open: Pos,
        /// 闭括号位置。
        close: Pos,
    },
}

impl Token {
    /// token 覆盖的起始行。
    pub fn start_line(&self) -> usize {
        match self {
            Token::Ident { pos, .. } | Token::Punct { pos, .. } => pos.line,
            Token::Literal { start, .. } => start.line,
            Token::Group { open, .. } => open.line,
        }
    }

    /// token 覆盖的结束行（跨行字面量/分组可能大于起始行）。
    pub fn end_line(&self) -> usize {
        match self {
            Token::Ident { pos, .. } | Token::Punct { pos, .. } => pos.line,
            Token::Literal { end, .. } => end.line,
            Token::Group { close, .. } => close.line,
        }
    }
}

/// 词法器：持有字符游标与当前位置。
struct Lexer {
    chars: Vec<char>,
    at: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            at: 0,
            line: 1,
            col: 1,
        }
    }

    fn pos(&self) -> Pos {
        Pos::new(self.line, self.col)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn peek_at(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.at + ahead).copied()
    }

    /// 消费一个字符并推进位置；换行复位列号。
    fn bump(&mut self) -> Option<char> {
        let ch = self.chars.get(self.at).copied()?;
        self.at += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    /// 解析到对应闭括号（或 EOF）；`close` 为 None 表示顶层。
    fn parse_until(&mut self, close: Option<char>) -> diagnostic::Result<Vec<Token>> {
        let mut items = Vec::new();
        loop {
            self.skip_trivia()?;
            let Some(ch) = self.peek() else {
                if close.is_some() {
                    return Err(Error::new(Code::Invalid, "括号未闭合").with("tool::lex"));
                }
                break;
            };
            if close == Some(ch) {
                break;
            }
            items.push(self.next_token()?);
        }
        Ok(items)
    }

    /// 跳过空白与注释（注释内换行同样推进行号）。
    fn skip_trivia(&mut self) -> diagnostic::Result<()> {
        loop {
            match self.peek() {
                Some(ch) if ch.is_whitespace() => {
                    self.bump();
                }
                Some('/') if self.peek_at(1) == Some('/') => {
                    while let Some(ch) = self.peek() {
                        if ch == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some('/') if self.peek_at(1) == Some('*') => self.block_comment()?,
                _ => break,
            }
        }
        Ok(())
    }

    /// 块注释：Rust 允许嵌套，用深度计数。
    fn block_comment(&mut self) -> diagnostic::Result<()> {
        self.bump();
        self.bump();
        let mut depth = 1usize;
        while depth > 0 {
            match self.peek() {
                Some('/') if self.peek_at(1) == Some('*') => {
                    self.bump();
                    self.bump();
                    depth += 1;
                }
                Some('*') if self.peek_at(1) == Some('/') => {
                    self.bump();
                    self.bump();
                    depth -= 1;
                }
                Some(_) => {
                    self.bump();
                }
                None => {
                    return Err(Error::new(Code::Invalid, "块注释未闭合").with("tool::lex"));
                }
            }
        }
        Ok(())
    }

    /// 取下一个非空白 token（调用前已跳过 trivia）。
    fn next_token(&mut self) -> diagnostic::Result<Token> {
        let ch = self.peek().expect("token 存在");
        if is_ident_start(ch) {
            return self.ident_or_string();
        }
        if ch.is_ascii_digit() {
            return self.number();
        }
        if ch == '"' {
            return self.quoted('"');
        }
        if ch == '\'' {
            return self.quote_or_lifetime();
        }
        if let Some((delim, close)) = open_delim(ch) {
            let open = self.pos();
            self.bump();
            let items = self.parse_until(Some(close))?;
            let close_pos = self.pos();
            self.bump();
            return Ok(Token::Group {
                delim,
                items,
                open,
                close: close_pos,
            });
        }
        let pos = self.pos();
        self.bump();
        Ok(Token::Punct { ch, pos })
    }

    /// 标识符；同时处理 `r"..."`/`r#"..."#` 与 `b"..."` 等带前缀字符串。
    fn ident_or_string(&mut self) -> diagnostic::Result<Token> {
        let start = self.pos();
        let first = self.peek().expect("ident 起始");
        // 原始字符串 / 字节原始字符串前缀。
        let raw = matches!(first, 'r') && matches!(self.peek_at(1), Some('"') | Some('#'));
        let byte_raw = matches!(first, 'b')
            && self.peek_at(1) == Some('r')
            && matches!(self.peek_at(2), Some('"') | Some('#'));
        if raw || byte_raw {
            return self.raw_string(start);
        }
        if first == 'b' && self.peek_at(1) == Some('"') {
            self.bump();
            return self.quoted('"');
        }
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if is_ident_cont(ch) {
                text.push(ch);
                self.bump();
            } else {
                break;
            }
        }
        Ok(Token::Ident { text, pos: start })
    }

    /// 原始字符串：`r"..."`、`r#"..."#`、`br"..."`（含任意个 `#`）。
    fn raw_string(&mut self, start: Pos) -> diagnostic::Result<Token> {
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if ch == '#' || ch == 'r' || ch == 'b' {
                text.push(ch);
                self.bump();
            } else {
                break;
            }
        }
        let hashes = text.matches('#').count();
        let Some('"') = self.peek() else {
            return Err(Error::new(Code::Invalid, "原始字符串缺少开引号").with("tool::lex"));
        };
        text.push('"');
        self.bump();
        loop {
            let Some(ch) = self.peek() else {
                return Err(Error::new(Code::Invalid, "原始字符串未闭合").with("tool::lex"));
            };
            if ch == '"' && self.closes_raw(hashes) {
                text.push('"');
                self.bump();
                for _ in 0..hashes {
                    text.push('#');
                    self.bump();
                }
                break;
            }
            text.push(ch);
            self.bump();
        }
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 当前位置的 `"` 后是否恰好跟 `hashes` 个 `#`（再下一个不能是 `#`，否则属于更长闭合）。
    fn closes_raw(&self, hashes: usize) -> bool {
        for ahead in 1..=hashes {
            if self.peek_at(ahead) != Some('#') {
                return false;
            }
        }
        self.peek_at(hashes + 1) != Some('#')
    }

    /// 普通（字节）字符串：处理反斜杠转义，转义字符整体跳过。
    fn quoted(&mut self, quote: char) -> diagnostic::Result<Token> {
        let start = self.pos();
        let mut text = String::new();
        text.push(quote);
        self.bump();
        loop {
            let Some(ch) = self.peek() else {
                return Err(Error::new(Code::Invalid, "字符串未闭合").with("tool::lex"));
            };
            text.push(ch);
            self.bump();
            if ch == '\\' {
                if let Some(escaped) = self.peek() {
                    text.push(escaped);
                    self.bump();
                }
            } else if ch == quote {
                break;
            }
        }
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 字符字面量 vs 生命周期：严格局部判定，不在整行内搜索闭合。
    ///
    /// 字符字面量只有两种形态：`'x'`（一个内容字符后紧跟 `'`）与 `'\..'`（反斜杠转义）；
    /// 生命周期（`'a`、`'static`）后面紧跟的不是 `'`。起始 `'` 单独当标点即可。
    fn quote_or_lifetime(&mut self) -> diagnostic::Result<Token> {
        let start = self.pos();
        let inner = self.peek_at(1);
        let after = self.peek_at(2);
        let escaped = inner == Some('\\');
        let plain = matches!(inner, Some(c) if c != '\'' && c != '\n') && after == Some('\'');
        if !escaped && !plain {
            self.bump();
            return Ok(Token::Punct {
                ch: '\'',
                pos: start,
            });
        }
        self.bump(); // 起始 '
        let mut text = String::from('\'');
        if escaped {
            self.bump(); // 反斜杠
            text.push('\\');
            loop {
                let Some(ch) = self.peek() else {
                    return Err(Error::new(Code::Invalid, "字符字面量未闭合").with("tool::lex"));
                };
                text.push(ch);
                self.bump();
                if ch == '\'' {
                    break;
                }
            }
        } else {
            let c = inner.expect("内容字符");
            text.push(c);
            self.bump();
            text.push('\'');
            self.bump();
        }
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 数字字面量：连续读到分隔符为止（含浮点、指数与类型后缀）。
    fn number(&mut self) -> diagnostic::Result<Token> {
        let start = self.pos();
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() || is_open_or_close(ch) {
                break;
            }
            // 范围符号 `..` 不属于字面量。
            if ch == '.' && self.peek_at(1) == Some('.') {
                break;
            }
            text.push(ch);
            self.bump();
        }
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 按原文长度推算 token 终点位置（字面量可能跨行）。
    fn after_last(&self, text: &str, start: Pos) -> Pos {
        let mut line = start.line;
        let mut col = start.col;
        for (idx, ch) in text.chars().enumerate() {
            if idx + 1 == text.chars().count() {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        Pos::new(line, col)
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn is_ident_cont(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn is_open_or_close(ch: char) -> bool {
    open_delim(ch).is_some() || matches!(ch, ')' | '}' | ']')
}

fn open_delim(ch: char) -> Option<(Delim, char)> {
    match ch {
        '(' => Some((Delim::Paren, ')')),
        '{' => Some((Delim::Brace, '}')),
        '[' => Some((Delim::Bracket, ']')),
        _ => None,
    }
}

/// 词法解析入口：把整份源码解析为顶层 token 序列。
pub fn parse(src: &str) -> diagnostic::Result<Vec<Token>> {
    let mut lexer = Lexer::new(src);
    lexer.parse_until(None)
}

/// 前 N 个顶层 token 的调试文本（测试与诊断用）。
#[must_use]
pub fn outline(tokens: &[Token], max: usize) -> String {
    tokens
        .iter()
        .take(max)
        .map(|token| match token {
            Token::Ident { text, .. } => text.clone(),
            Token::Punct { ch, .. } => ch.to_string(),
            Token::Literal { text, .. } => text.clone(),
            Token::Group { delim, .. } => match delim {
                Delim::Paren => "(..)".to_string(),
                Delim::Brace => "{..}".to_string(),
                Delim::Bracket => "[..]".to_string(),
            },
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_comments_and_strings_before_analysis() {
        let src = "// if ? &&\nfn s() -> u8 { let x = \"if ? &&\"; 1 }\n/* if ? */\n";
        let tokens = parse(src).expect("parse");
        let rendered = outline(&tokens, 20);
        assert!(rendered.contains("fn"));
        assert!(!rendered.contains("if ?"));
        assert!(!rendered.contains("/*"));
    }

    #[test]
    fn nested_groups_keep_order_and_positions() {
        let src = "fn f() { if a { b } else { c } }";
        let tokens = parse(src).expect("parse");
        assert_eq!(tokens.first().map(Token::start_line), Some(1));
        let group = tokens
            .iter()
            .find(|t| matches!(t, Token::Group { .. }))
            .expect("body group");
        assert_eq!(group.start_line(), 1);
        assert_eq!(group.end_line(), 1);
    }

    #[test]
    fn raw_and_char_literals_are_single_tokens() {
        let src = "r#\"a ? b\"#; 'c'; 'static;";
        let tokens = parse(src).expect("parse");
        assert!(matches!(&tokens[0], Token::Literal { .. }));
        assert!(matches!(&tokens[2], Token::Literal { .. }));
        // 生命周期：' 是标点，static 是标识符。
        assert!(matches!(&tokens[4], Token::Punct { ch: '\'', .. }));
        assert!(matches!(&tokens[5], Token::Ident { text, .. } if text == "static"));
    }

    #[test]
    fn unclosed_group_is_an_invalid_error() {
        let err = parse("fn f() { ").expect_err("must fail");
        assert_eq!(err.code(), Code::Invalid);
    }
}
