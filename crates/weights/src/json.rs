//! 受限 JSON 读取器：只服务权重文件的头部（safetensors）。
//!
//! 全仓零外部依赖，所以这里手写一个只认结构、不追求通用的读取器：它接受完整 JSON 文法，但只
//! 产出头部需要的几种值（`null` / 布尔 / 整数 / 字符串 / 数组 / 对象），并把小数与指数直接判错
//! ——头部里的形状与偏移都是整数，容忍 `1.0` 只会把一次输入错误推迟到下游。
//!
//! 报错一律带字节位置：头部是一个巨大的单行 JSON，没有位置就等于没有诊断。重复键也在这里被
//! 拒绝（与 [`crate::Header`] 拒绝重名条目同一原则）：否则"取第一个还是最后一个"会变成隐式约定。
//!
//! # 例子
//!
//! ```
//! use llm_weights::json::{self, Value};
//!
//! let value = json::parse(r#"{"a": [1, 2], "b": "x"}"#).expect("合法");
//! let fields = value.object().expect("顶层是对象");
//! assert_eq!(fields.len(), 2);
//! assert_eq!(fields[0].0, "a");
//! assert!(json::parse(r#"{"a": 1.5}"#).is_err(), "头部里的数字必须是整数");
//! ```

use diagnostic::{Code, Error, Result};

/// 嵌套上限：头部只有"对象 → 条目对象 → 数组"三层，超过它的一定是输入有问题。
const MAX_DEPTH: usize = 16;

/// 头部里可能出现的值。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`。
    Null,
    /// `true` / `false`。
    Bool(bool),
    /// 整数；小数与指数在解析期即被拒绝。
    Integer(i64),
    /// 字符串；`\uXXXX` 与代理对都已解码。
    String(String),
    /// 数组。
    Array(Vec<Value>),
    /// 对象：保持书写顺序，重复键在解析期即被拒绝。
    Object(Vec<(String, Value)>),
}

impl Value {
    /// 值的种类名，用于错误信息。
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "布尔",
            Self::Integer(_) => "整数",
            Self::String(_) => "字符串",
            Self::Array(_) => "数组",
            Self::Object(_) => "对象",
        }
    }

    /// 按对象取出字段（保持书写顺序）。
    ///
    /// # Errors
    ///
    /// 值不是对象时返回 [`Code::Invalid`]。
    pub fn object(&self) -> Result<&[(String, Value)]> {
        match self {
            Self::Object(fields) => Ok(fields),
            other => Err(self.mismatch("对象", other)),
        }
    }

    /// 按数组取出元素。
    ///
    /// # Errors
    ///
    /// 值不是数组时返回 [`Code::Invalid`]。
    pub fn array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(items) => Ok(items),
            other => Err(self.mismatch("数组", other)),
        }
    }

    /// 按字符串取出文本。
    ///
    /// # Errors
    ///
    /// 值不是字符串时返回 [`Code::Invalid`]。
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::String(text) => Ok(text),
            other => Err(self.mismatch("字符串", other)),
        }
    }

    /// 按整数取出数值。
    ///
    /// # Errors
    ///
    /// 值不是整数时返回 [`Code::Invalid`]。
    pub fn integer(&self) -> Result<i64> {
        match self {
            Self::Integer(number) => Ok(*number),
            other => Err(self.mismatch("整数", other)),
        }
    }

    /// 对象里按名字取字段；不是对象或没有该键时返回 `None`。
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(fields) => fields
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    fn mismatch(&self, expected: &str, other: &Value) -> Error {
        Error::new(
            Code::Invalid,
            format!("期望{expected}，得到{}", other.kind()),
        )
        .with("weights::json")
    }
}

/// 解析一段完整的 JSON 文本。
///
/// # Errors
///
/// 词法/文法错误、非整数数字、重复键、嵌套过深、值之后还有多余内容，一律 [`Code::Invalid`]，
/// 消息里带出错位置的字节下标。
pub fn parse(text: &str) -> Result<Value> {
    let mut reader = Reader::new(text);
    let value = reader.value(0)?;
    reader.skip_space();
    if reader.pos != reader.bytes.len() {
        return Err(reader.fail("值之后还有多余内容"));
    }
    Ok(value)
}

/// 逐字节推进的读取器：`pos` 同时是报错位置，所以不缓存、不跳读。
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn fail(&self, message: impl AsRef<str>) -> Error {
        self.fail_at(self.pos, message)
    }

    fn fail_at(&self, position: usize, message: impl AsRef<str>) -> Error {
        Error::new(
            Code::Invalid,
            format!("位置 {position}：{}", message.as_ref()),
        )
        .with("weights::json::parse")
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, wanted: u8) -> Result<()> {
        let at = self.pos;
        match self.bump() {
            Some(byte) if byte == wanted => Ok(()),
            Some(_) => Err(self.fail_at(at, format!("期望 `{}`", wanted as char))),
            None => Err(self.fail_at(at, format!("期望 `{}`，但已到末尾", wanted as char))),
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value> {
        let at = self.pos;
        let end = at + word.len();
        let slice = self
            .bytes
            .get(at..end)
            .ok_or_else(|| self.fail_at(at, format!("期望 `{word}`，但内容不足")))?;
        if slice != word.as_bytes() {
            return Err(self.fail_at(at, format!("期望 `{word}`")));
        }
        self.pos = end;
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(self.fail(format!("嵌套超过 {MAX_DEPTH} 层")));
        }
        self.skip_space();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.integer().map(Value::Integer),
            Some(_) => Err(self.fail("无法识别值的开头")),
            None => Err(self.fail("期望一个值，但已到末尾")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        let open = self.pos;
        self.expect(b'{')?;
        let mut fields: Vec<(String, Value)> = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(fields));
        }
        loop {
            self.skip_space();
            let at = self.pos;
            let key = self.string()?;
            if fields.iter().any(|(existing, _)| *existing == key) {
                return Err(self.fail_at(at, format!("对象里出现重复键 `{key}`")));
            }
            self.skip_space();
            self.expect(b':')?;
            let value = self.value(depth + 1)?;
            fields.push((key, value));
            self.skip_space();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(Value::Object(fields)),
                Some(_) => return Err(self.fail_at(self.pos - 1, "对象里期望 `,` 或 `}`")),
                None => return Err(self.fail_at(open, "对象没有闭合")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        let open = self.pos;
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => return Ok(Value::Array(items)),
                Some(_) => return Err(self.fail_at(self.pos - 1, "数组里期望 `,` 或 `]`")),
                None => return Err(self.fail_at(open, "数组没有闭合")),
            }
        }
    }

    fn string(&mut self) -> Result<String> {
        let open = self.pos;
        self.expect(b'"')?;
        let mut text = String::new();
        loop {
            let at = self.pos;
            let byte = self
                .bump()
                .ok_or_else(|| self.fail_at(open, "字符串没有闭合"))?;
            match byte {
                b'"' => return Ok(text),
                b'\\' => {
                    let escape = self
                        .bump()
                        .ok_or_else(|| self.fail_at(at, "转义序列不完整"))?;
                    match escape {
                        b'"' => text.push('"'),
                        b'\\' => text.push('\\'),
                        b'/' => text.push('/'),
                        b'b' => text.push('\u{8}'),
                        b'f' => text.push('\u{c}'),
                        b'n' => text.push('\n'),
                        b'r' => text.push('\r'),
                        b't' => text.push('\t'),
                        b'u' => text.push(self.unicode_escape(open)?),
                        other => {
                            return Err(self.fail_at(at, format!("未知转义 `\\{}`", other as char)));
                        }
                    }
                }
                0x00..=0x1f => {
                    return Err(self.fail_at(at, "控制字符必须转义"));
                }
                _ => {
                    let width = match byte {
                        0x00..=0x7f => 1,
                        0xc0..=0xdf => 2,
                        0xe0..=0xef => 3,
                        0xf0..=0xf7 => 4,
                        _ => return Err(self.fail_at(at, "非法的 UTF-8 起始字节")),
                    };
                    let end = at + width;
                    let slice = self
                        .bytes
                        .get(at..end)
                        .ok_or_else(|| self.fail_at(at, "多字节字符被截断"))?;
                    let text_slice = std::str::from_utf8(slice)
                        .map_err(|_| self.fail_at(at, "非法的 UTF-8 序列"))?;
                    text.push_str(text_slice);
                    self.pos = end;
                }
            }
        }
    }

    fn unicode_escape(&mut self, open: usize) -> Result<char> {
        let high = self.hex4()?;
        let code = if (0xd800..=0xdbff).contains(&high) {
            let at = self.pos;
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                return Err(self.fail_at(at, "高位代理后必须紧跟 `\\u` 低位代理"));
            }
            let low = self.hex4()?;
            if !(0xdc00..=0xdfff).contains(&low) {
                return Err(self.fail_at(at, "低位代理必须落在 U+DC00..U+DFFF"));
            }
            0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00)
        } else if (0xdc00..=0xdfff).contains(&high) {
            return Err(self.fail_at(open, "孤立的低位代理"));
        } else {
            high
        };
        char::from_u32(code).ok_or_else(|| self.fail_at(open, format!("U+{code:04X} 不是合法字符")))
    }

    fn hex4(&mut self) -> Result<u32> {
        let at = self.pos;
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self
                .bump()
                .ok_or_else(|| self.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
            let digit = (byte as char)
                .to_digit(16)
                .ok_or_else(|| self.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn integer(&mut self) -> Result<i64> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digits = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.pos == digits {
            return Err(self.fail_at(start, "整数缺少数字"));
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(self.fail_at(start, "只接受整数：头部里的形状与偏移不能是小数或指数"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).expect("数字是 ASCII");
        if text.len() > 1 && text.trim_start_matches('-').starts_with('0') {
            return Err(self.fail_at(start, format!("整数 `{text}` 不能有前导零")));
        }
        text.parse::<i64>()
            .map_err(|_| self.fail_at(start, format!("整数 `{text}` 超出 i64 范围")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_object_keeps_its_written_order() {
        let value = parse(r#"{"b": 1, "a": 2}"#).expect("合法");
        let fields = value.object().expect("对象");
        let names: Vec<&str> = fields.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["b", "a"]);
        assert_eq!(value.get("a").and_then(|it| it.integer().ok()), Some(2));
    }

    #[test]
    fn every_shape_the_header_uses_round_trips() {
        let text = r#"{"__metadata__": {"format": "pt"}, "w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}"#;
        let value = parse(text).expect("合法");
        assert_eq!(value.object().expect("对象").len(), 2);
        let entry = value.get("w").expect("有条目");
        assert_eq!(
            entry.get("dtype").and_then(|it| it.string().ok()),
            Some("F32")
        );
        let shape = entry.get("shape").expect("有形状").array().expect("数组");
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[1].integer().expect("整数"), 3);
    }

    #[test]
    fn escapes_including_surrogate_pairs_are_decoded() {
        let value = parse(r#""a\"b\\c\/d\ne\u0041\ud83d\ude00""#).expect("合法");
        assert_eq!(value.string().expect("字符串"), "a\"b\\c/d\neA😀");
    }

    #[test]
    fn non_integers_are_rejected_with_their_position() {
        for text in [r#"{"a": 1.5}"#, r#"{"a": 1e3}"#] {
            let error = parse(text).expect_err("小数或指数");
            assert_eq!(error.code(), Code::Invalid);
            assert!(error.to_string().contains("只接受整数"), "{error}");
        }
        let leading_zero = parse(r#"{"a": 01}"#).expect_err("前导零");
        assert!(
            leading_zero.to_string().contains("前导零"),
            "{leading_zero}"
        );
    }

    #[test]
    fn duplicate_keys_are_rejected_rather_than_silently_merged() {
        let error = parse(r#"{"a": 1, "a": 2}"#).expect_err("重复键");
        assert!(error.to_string().contains("重复键"), "{error}");
        assert!(error.to_string().contains("位置 "), "{error}");
    }

    #[test]
    fn truncated_and_trailing_input_is_rejected() {
        for text in [r#"{"a": 1"#, r#"{"a": "#, r#"{"a": 1} x"#, ""] {
            assert!(parse(text).is_err(), "{text:?}");
        }
        // 结尾空白是合法的：头部由长度前缀定界，最后一个字节常带 `\n`。
        assert!(parse("{} \n").is_ok());
    }

    #[test]
    fn deep_nesting_is_refused_instead_of_overflowing_the_stack() {
        let text = format!("{}{}", "[".repeat(MAX_DEPTH + 2), "]".repeat(MAX_DEPTH + 2));
        let error = parse(&text).expect_err("过深");
        assert!(error.to_string().contains("嵌套"), "{error}");
    }

    #[test]
    fn accessors_report_the_actual_kind() {
        let value = parse(r#""x""#).expect("合法");
        let error = value.object().expect_err("不是对象");
        assert!(error.to_string().contains("字符串"), "{error}");
        assert_eq!(value.kind(), "字符串");
        assert!(value.get("missing").is_none());
    }
}
