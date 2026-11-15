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

impl Delim {
    /// 开括号字符。
    pub const fn open(self) -> char {
        match self {
            Self::Paren => '(',
            Self::Brace => '{',
            Self::Bracket => '[',
        }
    }

    /// 闭括号字符。
    pub const fn close(self) -> char {
        match self {
            Self::Paren => ')',
            Self::Brace => '}',
            Self::Bracket => ']',
        }
    }
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
///
/// 源码先收集成 `Vec<char>`：前缀判定需要向前看一两个字符（[`Lexer::peek_at`]），
/// 按字符而非字节计数列号，中文注释与字符串也能正确定位；这是静态分析器，不追求零拷贝。
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

    /// 当前位置是否以 `first` + `second` 两个字符开头。
    fn at_pair(&self, first: char, second: char) -> bool {
        self.peek() == Some(first) && self.peek_at(1) == Some(second)
    }

    /// 解析到对应闭括号（或 EOF）；`close` 为 None 表示顶层。
    fn parse_until(&mut self, close: Option<char>) -> diagnostic::Result<Vec<Token>> {
        let mut items = Vec::new();
        loop {
            self.skip_trivia()?;
            let Some(ch) = self.peek() else {
                return eof(close, items);
            };
            if close == Some(ch) {
                return Ok(items);
            }
            items.push(self.next_token(ch)?);
        }
    }

    /// 跳过空白与注释（注释内换行同样推进行号）。
    fn skip_trivia(&mut self) -> diagnostic::Result<()> {
        loop {
            self.skip_space_and_line();
            if !self.at_pair('/', '*') {
                return Ok(());
            }
            self.block_comment()?;
        }
    }

    /// 跳过空白与 `//` 行注释（读到换行前停下）；块注释交给调用方。
    fn skip_space_and_line(&mut self) {
        loop {
            if self.peek().is_some_and(char::is_whitespace) {
                self.bump();
                continue;
            }
            if !self.at_pair('/', '/') {
                return;
            }
            while self.peek().is_some_and(|ch| ch != '\n') {
                self.bump();
            }
        }
    }

    /// 块注释：Rust 允许嵌套，深度归零即结束。
    fn block_comment(&mut self) -> diagnostic::Result<()> {
        self.bump();
        self.bump();
        let mut depth = 1usize;
        while depth > 0 {
            if self.at_pair('/', '*') {
                depth += 1;
                self.bump();
            } else if self.at_pair('*', '/') {
                depth -= 1;
                self.bump();
            }
            if self.bump().is_none() {
                return Err(Error::new(Code::Invalid, "块注释未闭合").with("tool::lex"));
            }
        }
        Ok(())
    }

    /// 取下一个非空白 token（调用前已跳过 trivia）：括号分组或叶子 token。
    fn next_token(&mut self, ch: char) -> diagnostic::Result<Token> {
        let Some((delim, closer)) = open_delim(ch) else {
            return match ch {
                '"' => self.quoted('"', "字符串"),
                '\'' => self.quote_or_lifetime(),
                _ => self.word_or_punct(ch),
            };
        };
        let open = self.pos();
        self.bump();
        let items = self.parse_until(Some(closer))?;
        let close = self.pos();
        self.bump();
        Ok(Token::Group {
            delim,
            items,
            open,
            close,
        })
    }

    /// 标识符、数字与单字符标点。
    fn word_or_punct(&mut self, ch: char) -> diagnostic::Result<Token> {
        if is_ident_start(ch) {
            return self.ident_or_string(ch);
        }
        if ch.is_ascii_digit() {
            return Ok(self.number());
        }
        let pos = self.pos();
        self.bump();
        Ok(Token::Punct { ch, pos })
    }

    /// 标识符；带字符串前缀（`r#"`、`b"`）时改走字符串分支。
    fn ident_or_string(&mut self, first: char) -> diagnostic::Result<Token> {
        let start = self.pos();
        match self.string_prefix(first) {
            Some(true) => return self.raw_string(start),
            Some(false) => {
                self.bump();
                return self.quoted('"', "字符串");
            }
            None => {}
        }
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if !is_ident_cont(ch) {
                break;
            }
            text.push(ch);
            self.bump();
        }
        Ok(Token::Ident { text, pos: start })
    }

    /// 前缀形态：`Some(true)` 原始字符串，`Some(false)` 普通字符串，`None` 普通标识符。
    fn string_prefix(&self, first: char) -> Option<bool> {
        match (first, self.peek_at(1)) {
            ('r', Some('"') | Some('#')) => Some(true),
            ('b', Some('r')) if matches!(self.peek_at(2), Some('"') | Some('#')) => Some(true),
            ('b', Some('"')) => Some(false),
            _ => None,
        }
    }

    /// 原始字符串：`r"..."`、`r#"..."#`、`br"..."`（含任意个 `#`）。
    fn raw_string(&mut self, start: Pos) -> diagnostic::Result<Token> {
        let mut text = String::new();
        while self.peek().is_some_and(|ch| matches!(ch, '#' | 'r' | 'b')) {
            text.extend(self.bump());
        }
        let hashes = text.matches('#').count();
        if self.peek() != Some('"') {
            return Err(Error::new(Code::Invalid, "原始字符串缺少开引号").with("tool::lex"));
        }
        text.push('"');
        self.bump();
        self.raw_body(&mut text, hashes)?;
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 读到与 `hashes` 个 `#` 匹配的闭引号；未闭合是错误。
    fn raw_body(&mut self, text: &mut String, hashes: usize) -> diagnostic::Result<()> {
        while let Some(ch) = self.peek() {
            if ch != '"' || !self.closes_raw(hashes) {
                text.push(ch);
                self.bump();
                continue;
            }
            for _ in 0..=hashes {
                text.extend(self.bump());
            }
            return Ok(());
        }
        Err(Error::new(Code::Invalid, "原始字符串未闭合").with("tool::lex"))
    }

    /// 当前位置的 `"` 后是否恰好跟 `hashes` 个 `#`（再多一个 `#` 属更长闭合）。
    fn closes_raw(&self, hashes: usize) -> bool {
        (1..=hashes).all(|ahead| self.peek_at(ahead) == Some('#'))
            && self.peek_at(hashes + 1) != Some('#')
    }

    /// 读到 `quote` 为止（`\` 后的一个字符整体跳过）；`what` 用于未闭合的错误文案。
    fn quoted(&mut self, quote: char, what: &str) -> diagnostic::Result<Token> {
        let start = self.pos();
        let mut text = String::from(quote);
        self.bump();
        loop {
            let Some(ch) = self.peek() else {
                return Err(Error::new(Code::Invalid, format!("{what}未闭合")).with("tool::lex"));
            };
            text.push(ch);
            self.bump();
            if ch == '\\' {
                text.extend(self.bump());
            } else if ch == quote {
                break;
            }
        }
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 字符字面量 vs 生命周期：只按局部三个字符判定，不在整行内搜索闭合。
    fn quote_or_lifetime(&mut self) -> diagnostic::Result<Token> {
        let start = self.pos();
        let inner = self.peek_at(1);
        if inner == Some('\\') {
            return self.quoted('\'', "字符字面量");
        }
        let closed =
            self.peek_at(2) == Some('\'') && inner.is_some_and(|ch| ch != '\'' && ch != '\n');
        if !closed {
            self.bump();
            return Ok(Token::Punct {
                ch: '\'',
                pos: start,
            });
        }
        self.bump();
        let mut text = String::from('\'');
        text.extend(inner);
        self.bump();
        text.push('\'');
        self.bump();
        let end = self.after_last(&text, start);
        Ok(Token::Literal { text, start, end })
    }

    /// 数字字面量：连续读到分隔符为止（含浮点、指数与类型后缀）。
    fn number(&mut self) -> Token {
        let start = self.pos();
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() || is_open_or_close(ch) || self.at_pair('.', '.') {
                break;
            }
            text.push(ch);
            self.bump();
        }
        let end = self.after_last(&text, start);
        Token::Literal { text, start, end }
    }

    /// 按原文长度推算 token 终点位置（字面量可能跨行）。
    fn after_last(&self, text: &str, start: Pos) -> Pos {
        let keep = text.chars().count().saturating_sub(1);
        text.chars().take(keep).fold(start, |pos, ch| match ch {
            '\n' => Pos::new(pos.line + 1, 1),
            _ => Pos::new(pos.line, pos.col + 1),
        })
    }
}

/// 括号内到达 EOF 是错误，顶层是正常结束。
fn eof(close: Option<char>, items: Vec<Token>) -> diagnostic::Result<Vec<Token>> {
    if close.is_none() {
        return Ok(items);
    }
    Err(Error::new(Code::Invalid, "括号未闭合").with("tool::lex"))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 前 N 个顶层 token 的调试文本。
    fn outline(tokens: &[Token], max: usize) -> String {
        tokens
            .iter()
            .take(max)
            .map(outline_piece)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 单个 token 的调试文本。
    fn outline_piece(token: &Token) -> String {
        match token {
            Token::Ident { text, .. } | Token::Literal { text, .. } => text.clone(),
            Token::Punct { ch, .. } => ch.to_string(),
            Token::Group { delim, .. } => format!("{}..{}", delim.open(), delim.close()),
        }
    }

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
