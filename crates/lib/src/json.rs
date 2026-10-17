//! Minimal RFC 8259 JSON value, parser, and serializer.
//!
//! `lib::json` is a dependency-free JSON implementation owned by the
//! foundational layer. It exists so workspace crates that exchange JSON with
//! external tools (`format` reads `config.json` and safetensors headers,
//! `tokenizer` reads `tokenizer.json`, `checkpoint` reads and writes its
//! manifest) do not depend on a third-party serializer. The implementation is
//! deliberately small: one value tree, one recursive-descent parser, one
//! serializer, and no schema machinery.
//!
//! The parser accepts the RFC 8259 grammar, rejects trailing non-whitespace
//! content, and preserves integer precision up to `i64`. Numbers whose integer
//! magnitude exceeds `i64` are represented as reals when the exponent is
//! explicit, and rejected otherwise; the manifest domain (byte offsets and
//! counts) fits comfortably in `i64`.
//!
//! Object key order is preserved by construction: values are stored in a
//! `BTreeMap`, so serialization is canonical and deterministic for a given
//! tree, which keeps checkpoint manifests stable across runs.

use std::collections::BTreeMap;
use std::fmt;

/// A JSON number: an exact signed integer, an exact unsigned integer, or a
/// real.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum jsonnumber {
    /// An exact signed integer.
    integer(i64),
    /// An exact unsigned integer beyond `i64::MAX` (up to `u64::MAX`).
    unsigned(u64),
    /// A real number (contains `.`, `e`, or both).
    real(f64),
}

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum jsonvalue {
    /// `null`
    null,
    /// `true` / `false`
    boolean(bool),
    /// A number.
    number(jsonnumber),
    /// A UTF-8 string.
    string(String),
    /// An array, in document order.
    array(Vec<jsonvalue>),
    /// An object keyed by string.
    object(BTreeMap<String, jsonvalue>),
}

impl jsonvalue {
    /// Returns the string value, when this is a string.
    #[must_use]
    pub fn getstr(&self) -> Option<&str> {
        match self {
            Self::string(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the boolean value, when this is a boolean.
    #[must_use]
    pub fn getbool(&self) -> Option<bool> {
        match self {
            Self::boolean(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the integer value, when this is an exact integer.
    #[must_use]
    pub fn getinteger(&self) -> Option<i64> {
        match self {
            Self::number(jsonnumber::integer(value)) => Some(*value),
            Self::number(jsonnumber::unsigned(value)) => i64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Returns the unsigned integer value, when this is an exact integer.
    #[must_use]
    pub fn getunsigned(&self) -> Option<u64> {
        match self {
            Self::number(jsonnumber::integer(value)) => u64::try_from(*value).ok(),
            Self::number(jsonnumber::unsigned(value)) => Some(*value),
            _ => None,
        }
    }

    /// Returns the real value (or the integer widened), when this is a number.
    #[must_use]
    pub fn getreal(&self) -> Option<f64> {
        match self {
            Self::number(jsonnumber::integer(value)) => Some(*value as f64),
            Self::number(jsonnumber::unsigned(value)) => Some(*value as f64),
            Self::number(jsonnumber::real(value)) => Some(*value),
            _ => None,
        }
    }

    /// Returns the array, when this is an array.
    #[must_use]
    pub fn getarray(&self) -> Option<&[jsonvalue]> {
        match self {
            Self::array(values) => Some(values),
            _ => None,
        }
    }

    /// Returns the object, when this is an object.
    #[must_use]
    pub fn getobject(&self) -> Option<&BTreeMap<String, jsonvalue>> {
        match self {
            Self::object(values) => Some(values),
            _ => None,
        }
    }

    /// Looks up a member of an object value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&jsonvalue> {
        self.getobject().and_then(|object| object.get(key))
    }

    /// Returns whether this is the JSON null value.
    #[must_use]
    pub fn isnull(&self) -> bool {
        matches!(self, Self::null)
    }
}

/// A JSON syntax or validity error, with the byte offset of the failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct jsonerror {
    /// The one-based line of the error within the document.
    line: usize,
    /// The one-based column of the error within the line.
    column: usize,
    /// The offset of the failure within the document.
    offset: usize,
    /// A human-readable description.
    message: String,
}

impl jsonerror {
    /// Returns the one-based line of the failure.
    #[must_use]
    pub fn getline(&self) -> usize {
        self.line
    }

    /// Returns the one-based column of the failure.
    #[must_use]
    pub fn getcolumn(&self) -> usize {
        self.column
    }

    /// Returns the byte offset of the failure.
    #[must_use]
    pub fn getoffset(&self) -> usize {
        self.offset
    }

    fn new(input: &[u8], offset: usize, message: impl Into<String>) -> Self {
        let mut line = 1usize;
        let mut column = 1usize;
        for byte in &input[..offset.min(input.len())] {
            if *byte == b'\n' {
                line += 1;
                column = 1;
            } else {
                column += 1;
            }
        }
        Self {
            line,
            column,
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for jsonerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at line {}, column {}",
            self.message, self.line, self.column
        )
    }
}

impl std::error::Error for jsonerror {}

/// Parses a UTF-8 JSON document from bytes.
///
/// The entire input must be consumed; trailing non-whitespace content is an
/// error.
///
/// # Errors
///
/// Returns [`jsonerror`] on malformed input, invalid UTF-8, or trailing
/// content.
pub fn parse(bytes: &[u8]) -> Result<jsonvalue, jsonerror> {
    let mut parser = parser {
        input: bytes,
        offset: 0,
    };
    let value = parser.parsevalue()?;
    parser.skipwhitespace();
    if parser.offset != bytes.len() {
        return Err(parser.error("trailing content after JSON document"));
    }
    Ok(value)
}

/// Parses a UTF-8 JSON document from a string.
///
/// # Errors
///
/// Returns [`jsonerror`] on malformed input.
pub fn parse_str(document: &str) -> Result<jsonvalue, jsonerror> {
    parse(document.as_bytes())
}

/// Serializes a value to compact UTF-8 JSON.
///
/// Object keys are emitted in `BTreeMap` order, so output is canonical for a
/// given tree. Real numbers are written with Rust's shortest round-trip
/// formatting; integers are written exactly.
#[must_use]
pub fn tostring(value: &jsonvalue) -> String {
    let mut out = String::new();
    writevalue(value, &mut out);
    out
}

fn writevalue(value: &jsonvalue, out: &mut String) {
    match value {
        jsonvalue::null => out.push_str("null"),
        jsonvalue::boolean(value) => out.push_str(if *value { "true" } else { "false" }),
        jsonvalue::number(jsonnumber::integer(value)) => {
            out.push_str(&value.to_string());
        }
        jsonvalue::number(jsonnumber::unsigned(value)) => {
            out.push_str(&value.to_string());
        }
        jsonvalue::number(jsonnumber::real(value)) => {
            if value.is_finite() {
                out.push_str(&value.to_string());
            } else {
                out.push_str("null");
            }
        }
        jsonvalue::string(value) => writestring(value, out),
        jsonvalue::array(values) => {
            out.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                writevalue(item, out);
            }
            out.push(']');
        }
        jsonvalue::object(values) => {
            out.push('{');
            for (index, (key, item)) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                writestring(key, out);
                out.push(':');
                writevalue(item, out);
            }
            out.push('}');
        }
    }
}

fn writestring(value: &str, out: &mut String) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character < '\u{20}' => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

struct parser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> parser<'a> {
    fn error(&self, message: impl Into<String>) -> jsonerror {
        jsonerror::new(self.input, self.offset, message)
    }

    fn skipwhitespace(&mut self) {
        while self.offset < self.input.len()
            && matches!(self.input[self.offset], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.offset += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.offset).copied()
    }

    fn parsevalue(&mut self) -> Result<jsonvalue, jsonerror> {
        self.skipwhitespace();
        match self.peek() {
            Some(b'n') => self.parseliteral(b"null", jsonvalue::null),
            Some(b't') => self.parseliteral(b"true", jsonvalue::boolean(true)),
            Some(b'f') => self.parseliteral(b"false", jsonvalue::boolean(false)),
            Some(b'"') => self.parsestring().map(jsonvalue::string),
            Some(b'[') => self.parsearray(),
            Some(b'{') => self.parseobject(),
            Some(b'-' | b'0'..=b'9') => self.parsenumber(),
            Some(_) => Err(self.error("expected a JSON value")),
            None => Err(self.error("unexpected end of JSON document")),
        }
    }

    fn parseliteral(&mut self, literal: &[u8], value: jsonvalue) -> Result<jsonvalue, jsonerror> {
        if self.input.len() - self.offset < literal.len()
            || &self.input[self.offset..self.offset + literal.len()] != literal
        {
            return Err(self.error("invalid literal"));
        }
        self.offset += literal.len();
        Ok(value)
    }

    fn parsearray(&mut self) -> Result<jsonvalue, jsonerror> {
        self.offset += 1; // consume '['
        let mut values = Vec::new();
        loop {
            self.skipwhitespace();
            match self.peek() {
                Some(b']') => {
                    self.offset += 1;
                    return Ok(jsonvalue::array(values));
                }
                None => return Err(self.error("unterminated array")),
                _ => {}
            }
            values.push(self.parsevalue()?);
            self.skipwhitespace();
            match self.peek() {
                Some(b',') => {
                    self.offset += 1;
                    self.skipwhitespace();
                    if self.peek() == Some(b']') {
                        return Err(self.error("trailing comma in array"));
                    }
                }
                Some(b']') => {
                    self.offset += 1;
                    return Ok(jsonvalue::array(values));
                }
                _ => return Err(self.error("expected ',' or ']' in array")),
            }
        }
    }

    fn parseobject(&mut self) -> Result<jsonvalue, jsonerror> {
        self.offset += 1; // consume '{'
        let mut values = BTreeMap::new();
        loop {
            self.skipwhitespace();
            match self.peek() {
                Some(b'}') => {
                    self.offset += 1;
                    return Ok(jsonvalue::object(values));
                }
                Some(b'"') => {}
                None => return Err(self.error("unterminated object")),
                _ => return Err(self.error("expected a string object key")),
            }
            let key = self.parsestring()?;
            self.skipwhitespace();
            if self.peek() != Some(b':') {
                return Err(self.error("expected ':' after object key"));
            }
            self.offset += 1;
            let value = self.parsevalue()?;
            values.insert(key, value);
            self.skipwhitespace();
            match self.peek() {
                Some(b',') => {
                    self.offset += 1;
                    self.skipwhitespace();
                    if self.peek() == Some(b'}') {
                        return Err(self.error("trailing comma in object"));
                    }
                }
                Some(b'}') => {
                    self.offset += 1;
                    return Ok(jsonvalue::object(values));
                }
                _ => return Err(self.error("expected ',' or '}' in object")),
            }
        }
    }

    fn parsestring(&mut self) -> Result<String, jsonerror> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.offset += 1; // consume '"'
        let mut out = String::new();
        loop {
            let byte = match self.peek() {
                Some(byte) => byte,
                None => return Err(self.error("unterminated string")),
            };
            self.offset += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = match self.peek() {
                        Some(escape) => escape,
                        None => return Err(self.error("unterminated escape sequence")),
                    };
                    self.offset += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.parsehex4()?;
                            let character = if (0xd800..=0xdbff).contains(&first) {
                                // High surrogate: require a low surrogate escape.
                                if self.peek() != Some(b'\\') {
                                    return Err(self.error("unpaired high surrogate"));
                                }
                                self.offset += 1;
                                if self.peek() != Some(b'u') {
                                    return Err(self.error("expected low surrogate"));
                                }
                                self.offset += 1;
                                let second = self.parsehex4()?;
                                if !(0xdc00..=0xdfff).contains(&second) {
                                    return Err(self.error("unpaired high surrogate"));
                                }
                                let codepoint =
                                    0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00);
                                match char::from_u32(codepoint) {
                                    Some(character) => character,
                                    None => return Err(self.error("invalid surrogate pair")),
                                }
                            } else if (0xdc00..=0xdfff).contains(&first) {
                                return Err(self.error("unpaired low surrogate"));
                            } else {
                                match char::from_u32(first) {
                                    Some(character) => character,
                                    None => return Err(self.error("invalid unicode escape")),
                                }
                            };
                            out.push(character);
                        }
                        _ => return Err(self.error("invalid escape sequence")),
                    }
                }
                byte if byte < 0x80 => out.push(byte as char),
                _ => {
                    // Multibyte UTF-8 sequence: decode it as a whole character.
                    let remaining = &self.input[self.offset - 1..];
                    let len = utf8len(remaining);
                    if len == 0 || remaining.len() < len {
                        return Err(self.error("invalid UTF-8 in string"));
                    }
                    match std::str::from_utf8(&remaining[..len]) {
                        Ok(text) => {
                            out.push_str(text);
                            self.offset += len - 1;
                        }
                        Err(_) => return Err(self.error("invalid UTF-8 in string")),
                    }
                }
            }
        }
    }

    fn parsehex4(&mut self) -> Result<u32, jsonerror> {
        if self.offset + 4 > self.input.len() {
            return Err(self.error("truncated unicode escape"));
        }
        let mut value = 0u32;
        for byte in &self.input[self.offset..self.offset + 4] {
            let digit = match byte {
                b'0'..=b'9' => *byte - b'0',
                b'a'..=b'f' => *byte - b'a' + 10,
                b'A'..=b'F' => *byte - b'A' + 10,
                _ => return Err(self.error("invalid hex digit in unicode escape")),
            };
            value = value * 16 + digit as u32;
        }
        self.offset += 4;
        Ok(value)
    }

    fn parsenumber(&mut self) -> Result<jsonvalue, jsonerror> {
        let start = self.offset;
        if self.peek() == Some(b'-') {
            self.offset += 1;
        }
        match self.peek() {
            Some(b'0') => self.offset += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.offset += 1;
                }
            }
            _ => return Err(self.error("invalid number")),
        }
        let mut real = false;
        if self.peek() == Some(b'.') {
            real = true;
            self.offset += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected digit after decimal point"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            real = true;
            self.offset += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.offset += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected digit in exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }
        let text = std::str::from_utf8(&self.input[start..self.offset])
            .map_err(|_| self.error("invalid number"))?;
        if real {
            let value: f64 = text
                .parse()
                .map_err(|_| self.error("number out of range"))?;
            if !value.is_finite() {
                return Err(self.error("number out of range"));
            }
            Ok(jsonvalue::number(jsonnumber::real(value)))
        } else {
            match text.parse::<i64>() {
                Ok(value) => Ok(jsonvalue::number(jsonnumber::integer(value))),
                Err(_) => match text.parse::<u64>() {
                    Ok(value) => Ok(jsonvalue::number(jsonnumber::unsigned(value))),
                    Err(_) => Err(self.error("integer out of range")),
                },
            }
        }
    }
}

/// Returns the UTF-8 sequence length of the character beginning at `bytes`,
/// or 0 when the leading byte is invalid.
fn utf8len(bytes: &[u8]) -> usize {
    match bytes.first() {
        Some(byte) if *byte < 0x80 => 1,
        Some(byte) if (0xc2..=0xdf).contains(byte) => 2,
        Some(byte) if (0xe0..=0xef).contains(byte) => 3,
        Some(byte) if (0xf0..=0xf4).contains(byte) => 4,
        _ => 0,
    }
}
