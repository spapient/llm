//! `json` 的数字层：整数/小数/指数的文法与原文保留。
//!
//! 单独成文件是因为数字的接受集是 [`crate::json::parse`] 与 [`crate::json::parse_ledger`] 的
//! 唯一差别，独立之后「哪个入口允许什么」只需读这一个文件。

use diagnostic::Result;

use crate::json::Value;
use crate::json_impl::{Reader, integer_value};

impl<'a> Reader<'a> {
    /// 一个数字：整数走 `i64`，小数与指数（只有账本模式接受）按原文保留。
    pub(crate) fn number(&mut self) -> Result<Value> {
        let start = self.cursor.pos();
        let digits = self.integer(start)?;
        if !matches!(self.cursor.peek(), Some(b'.' | b'e' | b'E')) {
            let text = &self.cursor.bytes()[start..self.cursor.pos()];
            return integer_value(&self.cursor, start, text).map(Value::Integer);
        }
        if !self.fractional {
            return Err(self
                .cursor
                .fail_at(start, "只接受整数：头部里的形状与偏移不能是小数或指数"));
        }
        let whole = self.cursor.pos();
        self.fraction(start)?;
        self.decimal(start, digits, whole).map(Value::Decimal)
    }

    /// 整数部分：可选负号 + 至少一位数字，返回数字起点。
    fn integer(&mut self, start: usize) -> Result<usize> {
        self.cursor
            .advance(usize::from(self.cursor.peek() == Some(b'-')));
        let digits = self.cursor.pos();
        if self.mantissa() {
            return Ok(digits);
        }
        Err(self.cursor.fail_at(start, "数字缺少整数部分"))
    }

    /// 小数与指数部分：`.` 后至少一位数字，`e`/`E` 后可带符号、同样至少一位数字。
    fn fraction(&mut self, start: usize) -> Result<()> {
        self.dot(start)?;
        self.exponent(start)
    }

    /// 小数部分（可缺）：`.` 之后必须至少一位数字。
    fn dot(&mut self, start: usize) -> Result<()> {
        if self.cursor.peek() != Some(b'.') {
            return Ok(());
        }
        self.cursor.advance(1);
        if self.mantissa() {
            return Ok(());
        }
        Err(self.cursor.fail_at(start, "小数点后缺少数字"))
    }

    /// 指数部分（可缺）：`e`/`E` 后可带符号，同样必须至少一位数字。
    fn exponent(&mut self, start: usize) -> Result<()> {
        if !matches!(self.cursor.peek(), Some(b'e' | b'E')) {
            return Ok(());
        }
        self.cursor.advance(1);
        self.cursor
            .advance(usize::from(matches!(self.cursor.peek(), Some(b'+' | b'-'))));
        if self.mantissa() {
            return Ok(());
        }
        Err(self.cursor.fail_at(start, "指数缺少数字"))
    }

    /// 数字原文，顺带挡掉前导零（`0.5` 合法，`01.5` 不合法）。
    fn decimal(&self, start: usize, digits: usize, whole: usize) -> Result<String> {
        let text = &self.cursor.bytes()[start..self.cursor.pos()];
        let head = &text[digits - start..whole - start];
        if head.len() > 1 && head.starts_with(b"0") {
            let shown = String::from_utf8_lossy(text);
            return Err(self
                .cursor
                .fail_at(start, format!("数字 `{shown}` 不能有前导零")));
        }
        Ok(String::from_utf8_lossy(text).into_owned())
    }

    /// 连续数字段；至少吃到一个才返回 `true`。
    fn mantissa(&mut self) -> bool {
        let start = self.cursor.pos();
        while matches!(self.cursor.peek(), Some(b'0'..=b'9')) {
            self.cursor.advance(1);
        }
        self.cursor.pos() > start
    }
}
