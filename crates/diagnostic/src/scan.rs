//! 带字节位置的扫描游标：逐字节文法（头部 JSON 之类）共用的最小骨架。
//!
//! 位置既是前进下标、也是报错坐标，所以这里不缓存、不跳读：只回答「下一个字节是什么」
//! 「跳过空白」「吃一个必需字节」。文法不在这里（各层自己写），错误消息也由调用方给，
//! 本类型只补上位置与站点——同一段输入在两处被解析时，诊断格式必须一致。
//!
//! # 例子
//!
//! ```
//! use llm_diagnostic::scan::Cursor;
//!
//! let mut cursor = Cursor::new("  {}", "demo::scan");
//! cursor.skip_space();
//! cursor.eat(b'{').expect("有左花括号");
//! assert_eq!(cursor.pos(), 3);
//! assert!(cursor.eat(b'}').is_err(), "已到末尾");
//! ```

use crate::{Code, Error, Result};

/// 一块文本上的逐字节游标：`pos` 同时是报错位置。
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    site: &'static str,
}

impl<'a> Cursor<'a> {
    /// 在一段文本上开一个游标；`site` 会写进这里产生的每条错误的上下文。
    pub fn new(text: &'a str, site: &'static str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
            site,
        }
    }

    /// 全文的字节视图。
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// 当前下标。
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 把下标移到 `pos`（调用方已经校验过的跳过）。
    pub fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// 前进 `count` 个字节。
    pub fn advance(&mut self, count: usize) {
        self.pos += count;
    }

    /// 下一个字节；已到末尾为 `None`。
    pub fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// 消费一个字节；已到末尾为 `None`。
    pub fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    /// 跳过 ASCII 空白（空格 / 制表 / 换行 / 回车）。
    pub fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// 吃一个必需的字节；不一致与已到末尾两种失败带不同文案。
    pub fn eat(&mut self, wanted: u8) -> Result<()> {
        let at = self.pos;
        match self.bump() {
            Some(byte) if byte == wanted => Ok(()),
            Some(_) => Err(self.fail_at(at, format!("期望 `{}`", wanted as char))),
            None => Err(self.fail_at(at, format!("期望 `{}`，但已到末尾", wanted as char))),
        }
    }

    /// 在当前下标处报错。
    pub fn fail(&self, message: impl Into<String>) -> Error {
        self.fail_at(self.pos, message)
    }

    /// 在指定下标处报错：消息前补位置，上下文补站点。
    pub fn fail_at(&self, position: usize, message: impl Into<String>) -> Error {
        let message = format!("位置 {position}：{}", message.into());
        Error::new(Code::Invalid, message).with(self.site)
    }
}
