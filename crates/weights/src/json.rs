//! 职责：受限 JSON 读取器——服务权重文件的头部（safetensors）与沉降账本（`llm.cert.v1`）。
//!
//! 全仓零外部依赖，所以这里手写一个只认结构、不追求通用的读取器：它接受完整 JSON 文法，但只
//! 产出这几处需要的几种值（`null` / 布尔 / 数字 / 字符串 / 数组 / 对象）。两个入口的唯一差别是
//! **数字的接受集**：[`parse`] 只认整数——头部里的形状与偏移都是整数，容忍 `1.0` 只会把一次输入
//! 错误推迟到下游；[`parse_ledger`] 另认小数与指数，且把它们按**原文**保留（不折算成 `f64`），
//! 于是账本的规范文本可以逐字比对。
//!
//! 报错一律带字节位置：头部是一个巨大的单行 JSON，没有位置就等于没有诊断。重复键也在这里被
//! 拒绝（与 `weights` 的 `Header` 拒绝重名条目同一原则）：否则"取第一个还是最后一个"会变成隐式约定。
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

use diagnostic::scan::Cursor;
use diagnostic::{Code, Error, Result};

use std::collections::HashSet;

/// 嵌套上限：头部只有"对象 → 条目对象 → 数组"三层，超过它的一定是输入有问题。
const MAX_DEPTH: usize = 16;

/// `\` 之后的单字符转义表：码位 → 字符（`u` 不在此列，走代理对路径）。
const SIMPLE_ESCAPES: [(u8, char); 8] = [
    (b'"', '"'),
    (b'\\', '\\'),
    (b'/', '/'),
    (b'b', '\u{8}'),
    (b'f', '\u{c}'),
    (b'n', '\n'),
    (b'r', '\r'),
    (b't', '\t'),
];

/// 头部里可能出现的值。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`。
    Null,
    /// `true` / `false`。
    Bool(bool),
    /// 整数（[`parse`] 与 [`parse_ledger`] 都可能产出）。
    Integer(i64),
    /// 小数或指数（只有 [`parse_ledger`] 产出）：保留**原始写法**，不折算成 `f64`——
    /// 折算会把 `0.500000` 与 `0.5` 变成同一个值，而账本的规范文本正是要比对这两者。
    Decimal(String),
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
        if matches!(self, Self::Array(_)) {
            return "数组";
        }
        if matches!(self, Self::Object(_)) {
            return "对象";
        }
        self.scalar_kind()
    }

    /// 标量的种类名：`kind` 已把两个容器挡在前面，这里的 `_` 只会落在字符串上。
    fn scalar_kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "布尔",
            Self::Integer(_) => "整数",
            Self::Decimal(_) => "小数",
            _ => "字符串",
        }
    }

    /// 按对象取出字段（保持书写顺序）；不是对象即 [`Code::Invalid`]。
    pub fn object(&self) -> Result<&[(String, Value)]> {
        match self {
            Self::Object(fields) => Ok(fields),
            _ => Err(self.mismatch("对象")),
        }
    }

    /// 按数组取出元素；不是数组即 [`Code::Invalid`]。
    pub fn array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(items) => Ok(items),
            _ => Err(self.mismatch("数组")),
        }
    }

    /// 按字符串取出文本；不是字符串即 [`Code::Invalid`]。
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::String(text) => Ok(text),
            _ => Err(self.mismatch("字符串")),
        }
    }

    /// 按整数取出数值；不是整数即 [`Code::Invalid`]。
    pub fn integer(&self) -> Result<i64> {
        match self {
            Self::Integer(number) => Ok(*number),
            _ => Err(self.mismatch("整数")),
        }
    }

    /// 数字的原文写法（整数转十进制文本，小数原样）；不是数字即 [`Code::Invalid`]。
    pub fn number_text(&self) -> Result<String> {
        match self {
            Self::Integer(number) => Ok(number.to_string()),
            Self::Decimal(text) => Ok(text.clone()),
            _ => Err(self.mismatch("数字")),
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

    /// 类型不符：期望 `expected`，实际是 `self` 的种类（种类名由 [`Value::kind`] 唯一给出）。
    fn mismatch(&self, expected: &str) -> Error {
        Error::new(
            Code::Invalid,
            format!("期望{expected}，得到{}", self.kind()),
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
    parse_with(text, false)
}

/// 解析一段账本文本：与 [`parse`] 同一套文法，唯一差别是数字允许小数与指数。
///
/// 小数的**原文**保留在 [`Value::Decimal`] 里而不折算成 `f64`：账本的规范文本把比值与熵
/// 写成六位小数，逐字比对才能区分「同一份账原样读回」与「重算出一份近似的账」。
///
/// # Errors
///
/// 与 [`parse`] 同一套判据（含重复键与嵌套上限），只多接受「数字带小数/指数」这一种输入。
pub fn parse_ledger(text: &str) -> Result<Value> {
    parse_with(text, true)
}

/// 两个入口的唯一差别：数字是否接受小数与指数。
fn parse_with(text: &str, fractional: bool) -> Result<Value> {
    let mut reader = Reader {
        cursor: Cursor::new(text, "weights::json::parse"),
        fractional,
    };
    let value = reader.value(0)?;
    reader.cursor.skip_space();
    if reader.cursor.pos() != text.len() {
        return Err(reader.cursor.fail("值之后还有多余内容"));
    }
    Ok(value)
}

/// 逐字节推进的读取器：[`Cursor`] 负责字节与位置（位置同时是报错坐标），这里只有文法。
struct Reader<'a> {
    cursor: Cursor<'a>,
    /// 数字里是否允许小数与指数：头部不允许（形状与偏移都是整数），账本允许。
    fractional: bool,
}

impl Reader<'_> {
    /// 一个值：深度的上限在这里统一设，分派交给 [`Reader::dispatch`]。
    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(self.cursor.fail(format!("嵌套超过 {MAX_DEPTH} 层")));
        }
        self.cursor.skip_space();
        let at = self.cursor.pos();
        let Some(byte) = self.cursor.peek() else {
            return Err(self.cursor.fail("期望一个值，但已到末尾"));
        };
        self.dispatch(byte, at, depth)
    }

    /// 按开头字节分派：两种容器与两种标量留在本层，三个字面量下沉一层。
    fn dispatch(&mut self, byte: u8, at: usize, depth: usize) -> Result<Value> {
        match byte {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => self.string().map(Value::String),
            b'-' | b'0'..=b'9' => self.number(),
            _ => self.literal_word(byte, at),
        }
    }

    /// 三个字面量：首字节定词与值，逐字节比对交给 [`Reader::literal`]。
    fn literal_word(&mut self, byte: u8, at: usize) -> Result<Value> {
        let (word, value) = match byte {
            b't' => ("true", Value::Bool(true)),
            b'f' => ("false", Value::Bool(false)),
            b'n' => ("null", Value::Null),
            _ => return Err(self.cursor.fail_at(at, "无法识别值的开头")),
        };
        self.literal(at, word, value)
    }

    /// 三个字面量：从 `at` 起必须逐字节等于 `word`，'内容不足'与'对不上'分开报。
    fn literal(&mut self, at: usize, word: &str, value: Value) -> Result<Value> {
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

    fn object(&mut self, depth: usize) -> Result<Value> {
        let open = self.cursor.pos();
        self.cursor.eat(b'{')?;
        Ok(Value::Object(self.members(open, depth)?))
    }

    /// 对象的成员序列：空对象直接收尾，成员之间必须用 `,` 分开。
    fn members(&mut self, open: usize, depth: usize) -> Result<Vec<(String, Value)>> {
        let mut fields: Vec<(String, Value)> = Vec::new();
        let mut seen = HashSet::new();
        while !self.close(b'}') {
            fields.push(self.field(depth + 1, &mut seen)?);
            if !self.separator(open, b'}', "对象")? {
                break;
            }
        }
        Ok(fields)
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        let open = self.cursor.pos();
        self.cursor.eat(b'[')?;
        Ok(Value::Array(self.items(open, depth)?))
    }

    /// 数组的元素序列：空数组直接收尾，元素之间必须用 `,` 分开。
    fn items(&mut self, open: usize, depth: usize) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        while !self.close(b']') {
            items.push(self.value(depth + 1)?);
            if !self.separator(open, b']', "数组")? {
                break;
            }
        }
        Ok(items)
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
    fn field(&mut self, depth: usize, seen: &mut HashSet<String>) -> Result<(String, Value)> {
        self.cursor.skip_space();
        let at = self.cursor.pos();
        let key = self.string()?;
        if !seen.insert(key.clone()) {
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

    /// 字符串：闭合引号收尾，转义/控制字符/多字节各一条判据。
    fn string(&mut self) -> Result<String> {
        let open = self.cursor.pos();
        self.cursor.eat(b'"')?;
        self.chars(open)
    }

    /// 字符串体：`None` 表示闭合引号，循环到此结束。
    fn chars(&mut self, open: usize) -> Result<String> {
        let mut text = String::new();
        while let Some(ch) = self.unit(open)? {
            text.push(ch);
        }
        Ok(text)
    }

    /// 开引号之后的一个字符：先把字节吃进来，再按它分类。
    fn unit(&mut self, open: usize) -> Result<Option<char>> {
        let at = self.cursor.pos();
        let Some(byte) = self.cursor.bump() else {
            return Err(self.cursor.fail_at(open, "字符串没有闭合"));
        };
        self.classify(open, at, byte)
    }

    /// 三类：闭合引号收尾、反斜杠转义、其余交给 [`Reader::plain`]（控制字符在那里拒）。
    fn classify(&mut self, open: usize, at: usize, byte: u8) -> Result<Option<char>> {
        match byte {
            b'"' => Ok(None),
            b'\\' => Ok(Some(self.escape(open, at)?)),
            _ => Ok(Some(self.plain(at)?)),
        }
    }

    /// 一个转义序列：`\` 之后的一个字节；`u` 走四位十六进制与代理对，其余查固定表。
    fn escape(&mut self, open: usize, at: usize) -> Result<char> {
        let escaped = self
            .cursor
            .bump()
            .ok_or_else(|| self.cursor.fail_at(at, "转义序列不完整"))?;
        if escaped == b'u' {
            return self.unicode_escape(open);
        }
        SIMPLE_ESCAPES
            .iter()
            .find(|(code, _)| *code == escaped)
            .map(|(_, ch)| *ch)
            .ok_or_else(|| {
                self.cursor
                    .fail_at(at, format!("未知转义 `\\{}`", escaped as char))
            })
    }

    /// 一个非转义字符：先定宽度再取整字符；控制字符必须写成转义。
    fn plain(&mut self, at: usize) -> Result<char> {
        let width = width_at(&self.cursor, at)?;
        if width == 1 && self.cursor.bytes()[at] < 0x20 {
            return Err(self.cursor.fail_at(at, "控制字符必须转义"));
        }
        let text = char_at(&self.cursor, at, width)?;
        self.cursor.seek(at + width);
        Ok(text)
    }

    /// `\uXXXX`：普通码位直接取，孤立低位代理拒，高位代理交给 [`Reader::pair`]。
    fn unicode_escape(&mut self, open: usize) -> Result<char> {
        let code = self.hex4()?;
        if (0xdc00..=0xdfff).contains(&code) {
            return Err(self.cursor.fail_at(open, "孤立的低位代理"));
        }
        if (0xd800..=0xdbff).contains(&code) {
            return self.pair(open, code);
        }
        code_point(&self.cursor, open, code)
    }

    /// 高位代理之后必须紧跟 `\u` 低位代理，两者拼成 `0x1_0000 + …`。
    fn pair(&mut self, open: usize, high: u32) -> Result<char> {
        let at = self.cursor.pos();
        if self.cursor.bump() != Some(b'\\') || self.cursor.bump() != Some(b'u') {
            return Err(self.cursor.fail_at(at, "高位代理后必须紧跟 `\\u` 低位代理"));
        }
        let low = self.hex4()?;
        if !(0xdc00..=0xdfff).contains(&low) {
            return Err(self.cursor.fail_at(at, "低位代理必须落在 U+DC00..U+DFFF"));
        }
        code_point(
            &self.cursor,
            open,
            0x1_0000 + ((high - 0xd800) << 10) + (low - 0xdc00),
        )
    }

    /// 四位十六进制：不足 4 位与非十六进制字符用同一句报错。
    fn hex4(&mut self) -> Result<u32> {
        let at = self.cursor.pos();
        (0..4).try_fold(0u32, |value, _| {
            let byte = self
                .cursor
                .bump()
                .ok_or_else(|| self.cursor.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
            let digit = (byte as char)
                .to_digit(16)
                .ok_or_else(|| self.cursor.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
            Ok(value * 16 + digit)
        })
    }

    /// 一个数字：定形（符号、整数段、两段后缀）后交给 [`Reader::convert`]。
    fn number(&mut self) -> Result<Value> {
        let start = self.cursor.pos();
        self.cursor
            .advance(usize::from(self.cursor.peek() == Some(b'-')));
        let digits = self.cursor.pos();
        if !self.mantissa() {
            return Err(self.cursor.fail_at(start, "数字缺少整数部分"));
        }
        let whole = self.cursor.pos();
        let fractional = self.fraction(start)?;
        no_leading_zero(&self.cursor, start, digits, whole)?;
        let text = self
            .cursor
            .bytes()
            .get(start..self.cursor.pos())
            .unwrap_or_default();
        self.convert(start, text, fractional)
    }

    /// 小数与指数（都可缺，且按序）：返回是否出现过其中任何一段。
    fn fraction(&mut self, start: usize) -> Result<bool> {
        let dotted = self.suffix(start, b".", false, "小数点后缺少数字")?;
        let exponent = self.suffix(start, b"eE", true, "指数缺少数字")?;
        Ok(dotted | exponent)
    }

    /// 整数走 `i64`；小数与指数只有账本模式接受。
    fn convert(&self, start: usize, text: &[u8], fractional: bool) -> Result<Value> {
        if fractional {
            return self.decimal(start, text);
        }
        self.integer(start, text)
    }

    /// 账本模式的小数：保留原文——折算会把 `0.500000` 与 `0.5` 变成同一个值。
    fn decimal(&self, start: usize, text: &[u8]) -> Result<Value> {
        if !self.fractional {
            return Err(self
                .cursor
                .fail_at(start, "只接受整数：头部里的形状与偏移不能是小数或指数"));
        }
        Ok(Value::Decimal(String::from_utf8_lossy(text).into_owned()))
    }

    /// 整数：必须 ASCII，且收得进 `i64`。
    fn integer(&self, start: usize, text: &[u8]) -> Result<Value> {
        let Ok(shown) = std::str::from_utf8(text) else {
            return Err(self.cursor.fail_at(start, "整数不是 ASCII"));
        };
        shown.parse::<i64>().map(Value::Integer).map_err(|_| {
            self.cursor
                .fail_at(start, format!("整数 `{shown}` 超出 i64 范围"))
        })
    }

    /// 小数与指数（都可缺）：标记符之后（指数可带符号）必须至少一位数字。
    fn suffix(&mut self, start: usize, marks: &[u8], sign: bool, message: &str) -> Result<bool> {
        if !self.cursor.peek().is_some_and(|byte| marks.contains(&byte)) {
            return Ok(false);
        }
        self.cursor.advance(1);
        if sign {
            self.cursor
                .advance(usize::from(matches!(self.cursor.peek(), Some(b'+' | b'-'))));
        }
        if self.mantissa() {
            return Ok(true);
        }
        Err(self.cursor.fail_at(start, message))
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

/// 起始字节 → 字符宽度（1..=4）；不是合法首字节的显式拒。
fn width_at(cursor: &Cursor<'_>, at: usize) -> Result<usize> {
    match cursor.bytes().get(at) {
        Some(0x00..=0x7f) => Ok(1),
        Some(0xc0..=0xdf) => Ok(2),
        Some(0xe0..=0xef) => Ok(3),
        Some(0xf0..=0xf7) => Ok(4),
        _ => Err(cursor.fail_at(at, "非法的 UTF-8 起始字节")),
    }
}

/// 取一个完整字符：切片要落在字符边界内，`from_utf8` 与 `chars` 都必须成立。
fn char_at(cursor: &Cursor<'_>, at: usize, width: usize) -> Result<char> {
    cursor
        .bytes()
        .get(at..at + width)
        .and_then(|raw| std::str::from_utf8(raw).ok())
        .and_then(|raw| raw.chars().next())
        .ok_or_else(|| cursor.fail_at(at, "非法的 UTF-8 序列"))
}

/// 一个码位：`char` 的闭集之外（越界与代理区）显式拒。
fn code_point(cursor: &Cursor<'_>, at: usize, code: u32) -> Result<char> {
    match char::from_u32(code) {
        Some(ch) => Ok(ch),
        None => Err(cursor.fail_at(at, format!("U+{code:04X} 不是合法字符"))),
    }
}

/// 前导零判据：整数部分超过一位且以 `0` 开头即拒（报错用原文，含符号）。
fn no_leading_zero(cursor: &Cursor<'_>, start: usize, digits: usize, whole: usize) -> Result<()> {
    let head = cursor.bytes().get(digits..whole).unwrap_or_default();
    if head.len() > 1 && head[0] == b'0' {
        let shown = String::from_utf8_lossy(cursor.bytes().get(start..whole).unwrap_or_default());
        return Err(cursor.fail_at(start, format!("数字 `{shown}` 不能有前导零")));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/json.rs"]
mod tests;
