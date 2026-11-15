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
    /// 值的种类名，用于错误信息。容器与标量分两段查：这段只回答「叫什么」，
    /// 不回答「能不能当某个类型用」（那是 `object`/`array`/… 各自的判据）。
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Array(_) => "数组",
            Self::Object(_) => "对象",
            scalar => scalar.scalar_kind(),
        }
    }

    /// 标量（`null`/布尔/整数/字符串）的种类名；容器走到 `_` 说明调用点用错了本函数。
    fn scalar_kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "布尔",
            Self::Integer(_) => "整数",
            _ => "字符串",
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
        located(position, message.as_ref())
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

    /// 吃一个必需的字节；不一致与已到末尾两种失败带不同文案。
    fn eat(&mut self, wanted: u8) -> Result<()> {
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
        let opened = self
            .peek()
            .ok_or_else(|| self.fail("期望一个值，但已到末尾"))?;
        self.dispatch(opened, depth)
    }

    /// 按开头字节分派：两个标量就地转换，剩下的交给 [`Reader::structural`]。
    fn dispatch(&mut self, opened: u8, depth: usize) -> Result<Value> {
        match opened {
            b'"' => self.string().map(Value::String),
            b'-' | b'0'..=b'9' => self.integer().map(Value::Integer),
            opened => self.structural(opened, depth),
        }
    }

    /// 结构化值（对象/数组）与三个字面量；标量已在 [`Reader::value`] 就地分派过了。
    fn structural(&mut self, opened: u8, depth: usize) -> Result<Value> {
        match opened {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            opened => self.word(opened),
        }
    }

    /// 三个字面量：`true` / `false` / `null`；其余开头字节视为无法识别。
    fn word(&mut self, opened: u8) -> Result<Value> {
        match opened {
            b't' => self.literal("true", Value::Bool(true)),
            b'f' => self.literal("false", Value::Bool(false)),
            b'n' => self.literal("null", Value::Null),
            _ => Err(self.fail("无法识别值的开头")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        let open = self.pos;
        self.eat(b'{')?;
        let mut fields: Vec<(String, Value)> = Vec::new();
        if self.close(b'}') {
            return Ok(Value::Object(fields));
        }
        loop {
            let (field, more) = self.member(open, depth + 1, &fields)?;
            fields.push(field);
            if !more {
                return Ok(Value::Object(fields));
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        let open = self.pos;
        self.eat(b'[')?;
        let mut items = Vec::new();
        if self.close(b']') {
            return Ok(Value::Array(items));
        }
        loop {
            let (item, more) = self.element(open, depth)?;
            items.push(item);
            if !more {
                return Ok(Value::Array(items));
            }
        }
    }

    /// 一个键值对，以及它后面是逗号（还有下一个）还是闭合符（到此为止）。
    fn member(
        &mut self,
        open: usize,
        depth: usize,
        seen: &[(String, Value)],
    ) -> Result<((String, Value), bool)> {
        let field = self.field(depth, seen)?;
        Ok((field, self.separator(open, b'}', "对象")?))
    }

    /// 一个元素，以及它后面是逗号（还有下一个）还是闭合符（到此为止）。
    fn element(&mut self, open: usize, depth: usize) -> Result<(Value, bool)> {
        let value = self.value(depth + 1)?;
        Ok((value, self.separator(open, b']', "数组")?))
    }

    /// 空容器：`skip_space` 后紧跟闭合符就吃掉它。
    fn close(&mut self, closing: u8) -> bool {
        self.skip_space();
        if self.peek() != Some(closing) {
            return false;
        }
        self.pos += 1;
        true
    }

    /// 一个键值对：键、`:`、值；重复键在这里拒（与 [`crate::Header`] 拒绝重名条目同一原则）。
    fn field(&mut self, depth: usize, seen: &[(String, Value)]) -> Result<(String, Value)> {
        self.skip_space();
        let at = self.pos;
        let key = self.string()?;
        if seen.iter().any(|(existing, _)| *existing == key) {
            return Err(self.fail_at(at, format!("对象里出现重复键 `{key}`")));
        }
        self.skip_space();
        self.eat(b':')?;
        let value = self.value(depth)?;
        Ok((key, value))
    }

    /// 元素之间的分隔：`,` 继续，闭合符收尾，其余（含输入结束）按 `what` 报错。
    fn separator(&mut self, open: usize, close: u8, what: &str) -> Result<bool> {
        self.skip_space();
        match self.bump() {
            Some(b',') => Ok(true),
            Some(byte) if byte == close => Ok(false),
            Some(_) => Err(self.fail_at(
                self.pos - 1,
                format!("{what}里期望 `,` 或 `{}`", close as char),
            )),
            None => Err(self.fail_at(open, format!("{what}没有闭合"))),
        }
    }

    fn string(&mut self) -> Result<String> {
        let open = self.pos;
        self.eat(b'"')?;
        let mut text = String::new();
        while let Some(chunk) = self.chunk(open)? {
            text.push_str(&chunk);
        }
        Ok(text)
    }

    /// 字符串的一段：闭合引号返回 `None`，否则返回这一段贡献的文本。
    fn chunk(&mut self, open: usize) -> Result<Option<String>> {
        let at = self.pos;
        let byte = self
            .bump()
            .ok_or_else(|| self.fail_at(open, "字符串没有闭合"))?;
        if byte == b'"' {
            return Ok(None);
        }
        Ok(Some(self.fragment(open, at, byte)?))
    }

    /// 字符串里的一个片段：转义、控制字符（必须转义）与多字节字符各一条判据。
    /// `open` 是字符串起点、`at` 是本片段起点，两者在报错时含义不同。
    fn fragment(&mut self, open: usize, at: usize, byte: u8) -> Result<String> {
        match byte {
            b'\\' => self.escape(open, at).map(String::from),
            0x00..=0x1f => Err(self.fail_at(at, "控制字符必须转义")),
            _ => self.utf8(at),
        }
    }

    /// 一个转义序列：`\` 之后的一个字节；`u` 走四位十六进制与代理对。
    fn escape(&mut self, open: usize, at: usize) -> Result<char> {
        let escaped = self
            .bump()
            .ok_or_else(|| self.fail_at(at, "转义序列不完整"))?;
        if escaped == b'u' {
            return self.unicode_escape(open);
        }
        simple_escape(escaped)
            .ok_or_else(|| self.fail_at(at, format!("未知转义 `\\{}`", escaped as char)))
    }

    /// 一个多字节字符：按起始字节定宽度，切片必须是合法 UTF-8。
    fn utf8(&mut self, at: usize) -> Result<String> {
        let Some(width) = sequence_width(self.bytes[at]) else {
            return Err(self.fail_at(at, "非法的 UTF-8 起始字节"));
        };
        let end = at + width;
        let Some(slice) = self.bytes.get(at..end) else {
            return Err(self.fail_at(at, "多字节字符被截断"));
        };
        let text = std::str::from_utf8(slice).map_err(|_| self.fail_at(at, "非法的 UTF-8 序列"))?;
        self.pos = end;
        Ok(text.to_string())
    }

    fn unicode_escape(&mut self, open: usize) -> Result<char> {
        let high = self.hex4()?;
        if is_high_surrogate(high) {
            return code_point(open, self.pair_surrogate(high)?);
        }
        if is_low_surrogate(high) {
            return Err(self.fail_at(open, "孤立的低位代理"));
        }
        code_point(open, high)
    }

    /// 高位代理之后必须紧跟 `\u` 低位代理：两者合成一个补充平面码位。
    fn pair_surrogate(&mut self, high: u32) -> Result<u32> {
        let at = self.pos;
        if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
            return Err(self.fail_at(at, "高位代理后必须紧跟 `\\u` 低位代理"));
        }
        let low = self.hex4()?;
        if !is_low_surrogate(low) {
            return Err(self.fail_at(at, "低位代理必须落在 U+DC00..U+DFFF"));
        }
        Ok(0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00))
    }

    fn hex4(&mut self) -> Result<u32> {
        let at = self.pos;
        let mut value = 0u32;
        for _ in 0..4 {
            value = value * 16 + self.hex_digit(at)?;
        }
        Ok(value)
    }

    /// 一位十六进制：不足 4 位与非十六进制字符用同一句报错（`at` 是 `\u` 之后的起点）。
    fn hex_digit(&mut self, at: usize) -> Result<u32> {
        let byte = self
            .bump()
            .ok_or_else(|| self.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
        (byte as char)
            .to_digit(16)
            .ok_or_else(|| self.fail_at(at, "`\\u` 需要 4 位十六进制"))
    }

    fn integer(&mut self) -> Result<i64> {
        let start = self.pos;
        self.pos += usize::from(self.peek() == Some(b'-'));
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
        integer_value(start, &self.bytes[start..self.pos])
    }
}

/// 带位置前缀的解析错误：读取器的方法与不持有读取器的辅助函数共用同一句前缀。
fn located(at: usize, message: impl std::fmt::Display) -> Error {
    Error::new(Code::Invalid, format!("位置 {at}：{message}")).with("weights::json::parse")
}

/// 简单转义表：`\X` 的第二个字节 → 字符，`'\0'` 表示不是简单转义（`u` 另行处理）。
const SIMPLE: [char; 128] = {
    let mut table = ['\0'; 128];
    table[b'"' as usize] = '"';
    table[b'\\' as usize] = '\\';
    table[b'/' as usize] = '/';
    table[b'b' as usize] = '\u{8}';
    table[b'f' as usize] = '\u{c}';
    table[b'n' as usize] = '\n';
    table[b'r' as usize] = '\r';
    table[b't' as usize] = '\t';
    table
};

/// `\X` 的字符：表里没有（或超出 ASCII）就是未知转义。
fn simple_escape(byte: u8) -> Option<char> {
    SIMPLE.get(byte as usize).copied().filter(|&ch| ch != '\0')
}

/// UTF-8 起始字节 → 序列宽度；`None` 表示非法起始字节。
fn sequence_width(byte: u8) -> Option<usize> {
    match byte {
        0x00..=0x7f => Some(1),
        0xc0..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf7 => Some(4),
        _ => None,
    }
}

/// 码位 → 字符：代理区之外并不都是合法标量值，所以这一步也可能失败。
fn code_point(at: usize, code: u32) -> Result<char> {
    char::from_u32(code).ok_or_else(|| located(at, format!("U+{code:04X} 不是合法字符")))
}

/// 高位代理：`U+D800..=U+DBFF`，后面必须跟一个低位代理。
fn is_high_surrogate(code: u32) -> bool {
    (0xd800..=0xdbff).contains(&code)
}

/// 低位代理：`U+DC00..=U+DFFF`，只有紧跟高位代理时才合法。
fn is_low_surrogate(code: u32) -> bool {
    (0xdc00..=0xdfff).contains(&code)
}

/// 整数字节 → `i64`：前导零在本地拒，范围与解析口径交给 `str::parse`。
fn integer_value(at: usize, bytes: &[u8]) -> Result<i64> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Err(located(at, "整数不是 ASCII"));
    };
    if text.len() > 1 && text.trim_start_matches('-').starts_with('0') {
        return Err(located(at, format!("整数 `{text}` 不能有前导零")));
    }
    text.parse::<i64>()
        .map_err(|_| located(at, format!("整数 `{text}` 超出 i64 范围")))
}

#[cfg(test)]
mod tests;
