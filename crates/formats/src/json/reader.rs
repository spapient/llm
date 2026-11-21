//! `json` 的文法实现：逐字节读取器与转义/数字/UTF-8 细节。
//!
//! 从 [`crate::json`] 拆出：公开面只留 [`crate::json::Value`] 与两个入口，这里管「一个字符
//! 序列怎么变成一个值」的全部词法。

use diagnostic::Result;
use diagnostic::scan::Cursor;

use crate::json::{MAX_DEPTH, Value};

/// 简单转义表：`\X` 的第二个字节 → 字符，`'\0'` 表示不是简单转义（`u` 另行处理）。
pub(crate) const SIMPLE: [char; 128] = {
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
pub(crate) fn simple_escape(byte: u8) -> Option<char> {
    SIMPLE.get(byte as usize).copied().filter(|&ch| ch != '\0')
}

/// UTF-8 起始字节 → 序列宽度；`None` 表示非法起始字节。
pub(crate) fn sequence_width(byte: u8) -> Option<usize> {
    match byte {
        0x00..=0x7f => Some(1),
        0xc0..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf7 => Some(4),
        _ => None,
    }
}

/// 码位 → 字符：代理区之外并不都是合法标量值，所以这一步也可能失败。
pub(crate) fn code_point(cursor: &Cursor<'_>, at: usize, code: u32) -> Result<char> {
    char::from_u32(code).ok_or_else(|| cursor.fail_at(at, format!("U+{code:04X} 不是合法字符")))
}

/// 高位代理：`U+D800..=U+DBFF`，后面必须跟一个低位代理。
pub(crate) fn is_high_surrogate(code: u32) -> bool {
    (0xd800..=0xdbff).contains(&code)
}

/// 低位代理：`U+DC00..=U+DFFF`，只有紧跟高位代理时才合法。
pub(crate) fn is_low_surrogate(code: u32) -> bool {
    (0xdc00..=0xdfff).contains(&code)
}

/// 整数字节 → `i64`：前导零在本地拒，范围与解析口径交给 `str::parse`。
pub(crate) fn integer_value(cursor: &Cursor<'_>, at: usize, bytes: &[u8]) -> Result<i64> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Err(cursor.fail_at(at, "整数不是 ASCII"));
    };
    if text.len() > 1 && text.trim_start_matches('-').starts_with('0') {
        return Err(cursor.fail_at(at, format!("整数 `{text}` 不能有前导零")));
    }
    text.parse::<i64>()
        .map_err(|_| cursor.fail_at(at, format!("整数 `{text}` 超出 i64 范围")))
}

/// 两个入口的唯一差别：[`Reader`] 是否接受小数与指数。
pub(crate) fn parse_with(text: &str, fractional: bool) -> Result<Value> {
    let mut reader = Reader::new(text, fractional);
    let value = reader.value(0)?;
    reader.cursor.skip_space();
    if reader.cursor.pos() != reader.cursor.bytes().len() {
        return Err(reader.cursor.fail("值之后还有多余内容"));
    }
    Ok(value)
}

/// 逐字节推进的读取器：[`Cursor`] 负责字节与位置（位置同时是报错坐标），这里只有文法。
pub(crate) struct Reader<'a> {
    pub(crate) cursor: Cursor<'a>,
    /// 数字里是否允许小数与指数：头部不允许（形状与偏移都是整数），账本允许。
    pub(crate) fractional: bool,
}

impl<'a> Reader<'a> {
    fn new(text: &'a str, fractional: bool) -> Self {
        Self {
            cursor: Cursor::new(text, "weights::json::parse"),
            fractional,
        }
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value> {
        let at = self.cursor.pos();
        let end = at + word.len();
        let slice = self.cursor.bytes().get(at..end).ok_or_else(|| {
            self.cursor
                .fail_at(at, format!("期望 `{word}`，但内容不足"))
        })?;
        if slice != word.as_bytes() {
            return Err(self.cursor.fail_at(at, format!("期望 `{word}`")));
        }
        self.cursor.seek(end);
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(self.cursor.fail(format!("嵌套超过 {MAX_DEPTH} 层")));
        }
        self.cursor.skip_space();
        let opened = self
            .cursor
            .peek()
            .ok_or_else(|| self.cursor.fail("期望一个值，但已到末尾"))?;
        self.dispatch(opened, depth)
    }

    /// 按开头字节分派：两个标量就地转换，剩下的交给 [`Reader::structural`]。
    fn dispatch(&mut self, opened: u8, depth: usize) -> Result<Value> {
        match opened {
            b'"' => self.string().map(Value::String),
            b'-' | b'0'..=b'9' => self.number(),
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
            _ => Err(self.cursor.fail("无法识别值的开头")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        let open = self.cursor.pos();
        self.cursor.eat(b'{')?;
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
        let open = self.cursor.pos();
        self.cursor.eat(b'[')?;
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
        self.cursor.skip_space();
        if self.cursor.peek() != Some(closing) {
            return false;
        }
        self.cursor.advance(1);
        true
    }

    /// 一个键值对：键、`:`、值；重复键在这里拒（与 `weights` 的 `Header` 拒绝重名条目同一原则）。
    fn field(&mut self, depth: usize, seen: &[(String, Value)]) -> Result<(String, Value)> {
        self.cursor.skip_space();
        let at = self.cursor.pos();
        let key = self.string()?;
        if seen.iter().any(|(existing, _)| *existing == key) {
            return Err(self.cursor.fail_at(at, format!("对象里出现重复键 `{key}`")));
        }
        self.cursor.skip_space();
        self.cursor.eat(b':')?;
        let value = self.value(depth)?;
        Ok((key, value))
    }

    /// 元素之间的分隔：`,` 继续，闭合符收尾，其余（含输入结束）按 `what` 报错。
    fn separator(&mut self, open: usize, close: u8, what: &str) -> Result<bool> {
        self.cursor.skip_space();
        match self.cursor.bump() {
            Some(b',') => Ok(true),
            Some(byte) if byte == close => Ok(false),
            Some(_) => Err(self.cursor.fail_at(
                self.cursor.pos() - 1,
                format!("{what}里期望 `,` 或 `{}`", close as char),
            )),
            None => Err(self.cursor.fail_at(open, format!("{what}没有闭合"))),
        }
    }

    fn string(&mut self) -> Result<String> {
        let open = self.cursor.pos();
        self.cursor.eat(b'"')?;
        let mut text = String::new();
        while let Some(chunk) = self.chunk(open)? {
            text.push_str(&chunk);
        }
        Ok(text)
    }

    /// 字符串的一段：闭合引号返回 `None`，否则返回这一段贡献的文本。
    fn chunk(&mut self, open: usize) -> Result<Option<String>> {
        let at = self.cursor.pos();
        let byte = self
            .cursor
            .bump()
            .ok_or_else(|| self.cursor.fail_at(open, "字符串没有闭合"))?;
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
            0x00..=0x1f => Err(self.cursor.fail_at(at, "控制字符必须转义")),
            _ => self.utf8(at),
        }
    }

    /// 一个转义序列：`\` 之后的一个字节；`u` 走四位十六进制与代理对。
    fn escape(&mut self, open: usize, at: usize) -> Result<char> {
        let escaped = self
            .cursor
            .bump()
            .ok_or_else(|| self.cursor.fail_at(at, "转义序列不完整"))?;
        if escaped == b'u' {
            return self.unicode_escape(open);
        }
        simple_escape(escaped).ok_or_else(|| {
            self.cursor
                .fail_at(at, format!("未知转义 `\\{}`", escaped as char))
        })
    }

    /// 一个多字节字符：按起始字节定宽度，切片必须是合法 UTF-8。
    fn utf8(&mut self, at: usize) -> Result<String> {
        let Some(width) = sequence_width(self.cursor.bytes()[at]) else {
            return Err(self.cursor.fail_at(at, "非法的 UTF-8 起始字节"));
        };
        let end = at + width;
        let Some(slice) = self.cursor.bytes().get(at..end) else {
            return Err(self.cursor.fail_at(at, "多字节字符被截断"));
        };
        let text =
            std::str::from_utf8(slice).map_err(|_| self.cursor.fail_at(at, "非法的 UTF-8 序列"))?;
        self.cursor.seek(end);
        Ok(text.to_string())
    }

    fn unicode_escape(&mut self, open: usize) -> Result<char> {
        let high = self.hex4()?;
        if is_high_surrogate(high) {
            let code = self.pair_surrogate(high)?;
            return code_point(&self.cursor, open, code);
        }
        if is_low_surrogate(high) {
            return Err(self.cursor.fail_at(open, "孤立的低位代理"));
        }
        code_point(&self.cursor, open, high)
    }

    /// 高位代理之后必须紧跟 `\u` 低位代理：两者合成一个补充平面码位。
    fn pair_surrogate(&mut self, high: u32) -> Result<u32> {
        let at = self.cursor.pos();
        if self.cursor.bump() != Some(b'\\') || self.cursor.bump() != Some(b'u') {
            return Err(self.cursor.fail_at(at, "高位代理后必须紧跟 `\\u` 低位代理"));
        }
        let low = self.hex4()?;
        if !is_low_surrogate(low) {
            return Err(self.cursor.fail_at(at, "低位代理必须落在 U+DC00..U+DFFF"));
        }
        Ok(0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00))
    }

    fn hex4(&mut self) -> Result<u32> {
        let at = self.cursor.pos();
        let mut value = 0u32;
        for _ in 0..4 {
            value = value * 16 + self.hex_digit(at)?;
        }
        Ok(value)
    }

    /// 一位十六进制：不足 4 位与非十六进制字符用同一句报错（`at` 是 `\u` 之后的起点）。
    fn hex_digit(&mut self, at: usize) -> Result<u32> {
        let byte = self
            .cursor
            .bump()
            .ok_or_else(|| self.cursor.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
        (byte as char)
            .to_digit(16)
            .ok_or_else(|| self.cursor.fail_at(at, "`\\u` 需要 4 位十六进制"))
    }
}
