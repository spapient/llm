//! 受限 JSON 读取器：服务权重文件的头部（safetensors）与沉降账本（`llm.cert.v1`）。
//!
//! 全仓零外部依赖，所以这里手写一个只认结构、不追求通用的读取器：它接受完整 JSON 文法，但只
//! 产出这几处需要的几种值（`null` / 布尔 / 数字 / 字符串 / 数组 / 对象）。两个入口的唯一差别是
//! **数字的接受集**：[`parse`] 只认整数——头部里的形状与偏移都是整数，容忍 `1.0` 只会把一次输入
//! 错误推迟到下游；[`parse_ledger`] 另认小数与指数，且把它们按**原文**保留（不折算成 `f64`），
//! 于是账本的规范文本可以逐字比对。
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

use diagnostic::scan::Cursor;
use diagnostic::{Code, Error, Result};

/// 嵌套上限：头部只有"对象 → 条目对象 → 数组"三层，超过它的一定是输入有问题。
pub(crate) const MAX_DEPTH: usize = 16;

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
            Self::Decimal(_) => "小数",
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

    /// 取数字的原文写法：整数转成十进制文本，小数原样返回。
    ///
    /// # Errors
    ///
    /// 值不是数字时返回 [`Code::Invalid`]。
    pub fn number_text(&self) -> Result<String> {
        match self {
            Self::Integer(number) => Ok(number.to_string()),
            Self::Decimal(text) => Ok(text.clone()),
            other => Err(self.mismatch("数字", other)),
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
    crate::json_impl::parse_with(text, false)
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
    crate::json_impl::parse_with(text, true)
}
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

#[cfg(test)]
#[path = "../tests/json.rs"]
mod tests;
