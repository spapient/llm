//! [`BenchRecord`] 的受限 JSON 写出与回读（M2-4）。
//!
//! 全仓零外部依赖，所以这里手写一个只服务归档记录的读写层：写出侧只暴露记录已校验过的
//! 字段，未测量的实测与未测得的 roofline 上界写 `null`；回读侧按固定台账取回，并在解析期
//! 拒绝重复键、嵌套过深、截断输入与非有限数（`NaN` / `Infinity` 不是 JSON 词法）。
//!
//! **不拥有**：采样、基线结论、跨机对照；数值口径沿用 [`bench`](crate::bench) 各类型的构造自检。

use diagnostic::{Code, Error, Result};

use crate::bench::{BenchRecord, EntryKind, Geometry, Harness, Roofline};

/// 嵌套上限：归档记录只有「对象 → 几何 / 档案对象 → 形状数组」三层，超过它的一定是输入有问题。
const MAX_DEPTH: usize = 16;

// ---------------------------------------------------------------------------
// 写出
// ---------------------------------------------------------------------------

/// 把一条记录渲染成归档 JSON 文本；键序固定，解析器按键名读回，不依赖顺序。
pub fn to_json(record: &BenchRecord) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"kind\": ");
    write_string(&mut out, record.kind().as_str());
    out.push_str(",\n  \"name\": ");
    write_string(&mut out, record.name());
    out.push_str(",\n  \"backend\": ");
    write_string(&mut out, record.backend());
    write_geometry(&mut out, record.geometry());
    write_harness(&mut out, record.harness());
    write_roofline(&mut out, record.roofline());
    out.push_str(",\n  \"observed\": ");
    write_number_or_null(&mut out, record.observed());
    out.push_str("\n}\n");
    out
}

fn write_geometry(out: &mut String, geometry: &Geometry) {
    out.push_str(",\n  \"geometry\": {\n    \"shape\": [");
    let shape = geometry.shape();
    for (index, dim) in shape.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&dim.to_string());
    }
    out.push_str("],\n    \"batch\": ");
    out.push_str(&geometry.batch().to_string());
    out.push_str(",\n    \"dtype\": ");
    write_string(out, geometry.dtype());
    out.push_str("\n  }");
}

fn write_harness(out: &mut String, harness: &Harness) {
    out.push_str(",\n  \"harness\": {\n    \"host\": ");
    write_string(out, harness.host());
    out.push_str(",\n    \"cpu\": ");
    write_string(out, harness.cpu());
    out.push_str(",\n    \"os\": ");
    write_string(out, harness.os());
    out.push_str(",\n    \"arch\": ");
    write_string(out, harness.arch());
    out.push_str(",\n    \"toolchain\": ");
    write_string(out, harness.toolchain());
    out.push_str(",\n    \"driver\": ");
    write_string(out, harness.driver());
    out.push_str("\n  }");
}

fn write_roofline(out: &mut String, roofline: &Roofline) {
    out.push_str(",\n  \"roofline\": {\n    \"peak_bw_gb_s\": ");
    write_number_or_null(out, roofline.peak_bw_gb_s());
    out.push_str(",\n    \"peak_gflops\": ");
    write_number_or_null(out, roofline.peak_gflops());
    out.push_str("\n  }");
}

/// 写一个可选有限数；缺测写 `null`。记录构造已拒非有限数，这里不再二次判断。
fn write_number_or_null(out: &mut String, value: Option<f64>) {
    match value {
        Some(number) => out.push_str(&number.to_string()),
        None => out.push_str("null"),
    }
}

/// 写一个 JSON 字符串：引号、反斜杠与控制字符转义，其余 UTF-8 原样透出。
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match escaped(ch) {
            Some(literal) => out.push_str(literal),
            None if ch.is_control() => write_control(out, ch),
            None => out.push(ch),
        }
    }
    out.push('"');
}

/// 必须转义的结构字符 → JSON 转义字面量；其余字符返回 `None`。
fn escaped(ch: char) -> Option<&'static str> {
    match ch {
        '"' => Some("\\\""),
        '\\' => Some("\\\\"),
        _ => None,
    }
}

/// 控制字符：换行 / 回车 / 制表用短转义，其余按码位写 `\uXXXX`。
fn write_control(out: &mut String, ch: char) {
    let literal = match ch {
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
        _ => "",
    };
    if literal.is_empty() {
        out.push_str(&format!("\\u{:04x}", ch as u32));
    } else {
        out.push_str(literal);
    }
}

// ---------------------------------------------------------------------------
// 回读：受限词法读取器 + 台账取值
// ---------------------------------------------------------------------------

/// 回读一段完整 JSON 文本，按归档记录台账取回一条 [`BenchRecord`]。
///
/// # Errors
///
/// 词法 / 文法错误、重复键、嵌套过深、截断输入、非有限数、缺必填字段或类型不符，一律
/// [`Code::Invalid`]；解析期错误消息带出错字节位置，台账层错误消息带字段名。
pub fn from_json(text: &str) -> Result<BenchRecord> {
    let value = parse(text)?;
    let object = value.object()?;
    record_of(object)
}

fn record_of(object: &[(String, Value)]) -> Result<BenchRecord> {
    let (kind, name, backend) = head_of(object)?;
    let geometry = geometry_of(object)?;
    let harness = harness_of(object)?;
    let roofline = roofline_of(object)?;
    assemble(kind, name, backend, geometry, harness, roofline, object)
}

fn head_of(object: &[(String, Value)]) -> Result<(EntryKind, String, String)> {
    let kind = kind_of(field(object, "kind")?)?;
    let name = str_field(object, "name")?.to_string();
    let backend = str_field(object, "backend")?.to_string();
    Ok((kind, name, backend))
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    kind: EntryKind,
    name: String,
    backend: String,
    geometry: Geometry,
    harness: Harness,
    roofline: Roofline,
    object: &[(String, Value)],
) -> Result<BenchRecord> {
    let observed = optional_number_field(object, "observed")?;
    let record = BenchRecord::new(kind, name, backend, geometry, harness, roofline)?;
    apply_observed(record, observed)
}

fn apply_observed(record: BenchRecord, observed: Option<f64>) -> Result<BenchRecord> {
    match observed {
        Some(number) => record.with_observed(number),
        None => Ok(record),
    }
}

fn kind_of(value: &Value) -> Result<EntryKind> {
    let text = string_of(value)?;
    match text {
        "op" => Ok(EntryKind::Op),
        "layer" => Ok(EntryKind::Layer),
        "model" => Ok(EntryKind::Model),
        other => Err(invalid(format!(
            "条目层级 `{other}` 不在 op / layer / model 之内"
        ))),
    }
}

fn geometry_of(object: &[(String, Value)]) -> Result<Geometry> {
    let geo = nested(object, "geometry")?;
    let shape = shape_of(geo)?;
    build_geometry(geo, shape)
}

fn build_geometry(geo: &[(String, Value)], shape: Vec<usize>) -> Result<Geometry> {
    let batch = batch_of(geo)?;
    let dtype = str_field(geo, "dtype")?.to_string();
    Geometry::new(shape, batch, dtype)
}

fn shape_of(geo: &[(String, Value)]) -> Result<Vec<usize>> {
    let items = arr_field(geo, "shape")?;
    let mut shape = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        shape.push(dim_usize(item, &format!("shape[{index}]"))?);
    }
    Ok(shape)
}

fn batch_of(geo: &[(String, Value)]) -> Result<u32> {
    let batch = dim_usize(field(geo, "batch")?, "batch")?;
    u32::try_from(batch).map_err(|_| invalid("`batch` 超出 u32 范围"))
}

fn harness_of(object: &[(String, Value)]) -> Result<Harness> {
    let har = nested(object, "harness")?;
    let host = str_field(har, "host")?.to_string();
    let cpu = str_field(har, "cpu")?.to_string();
    let os = str_field(har, "os")?.to_string();
    build_harness(har, host, cpu, os)
}

fn build_harness(
    har: &[(String, Value)],
    host: String,
    cpu: String,
    os: String,
) -> Result<Harness> {
    let arch = str_field(har, "arch")?.to_string();
    let toolchain = str_field(har, "toolchain")?.to_string();
    let driver = str_field(har, "driver")?.to_string();
    Ok(Harness::new(host, cpu, os, arch, toolchain, driver))
}

fn roofline_of(object: &[(String, Value)]) -> Result<Roofline> {
    let roof = nested(object, "roofline")?;
    let bw = optional_number_field(roof, "peak_bw_gb_s")?;
    let gflops = optional_number_field(roof, "peak_gflops")?;
    Roofline::new(bw, gflops)
}

/// 数组 / 形状维度：非负整数；小数、负数与越界整数都在这里拒。
fn dim_usize(value: &Value, field: &str) -> Result<usize> {
    let number = number_of(value, field)?;
    if number < 0.0 || number.fract() != 0.0 {
        return Err(invalid(format!("字段 `{field}` 必须是非负整数")));
    }
    let integer = number as u128;
    if integer > usize::MAX as u128 {
        return Err(invalid(format!("字段 `{field}` 超出 usize 范围")));
    }
    Ok(integer as usize)
}

/// 可选数值：`null` 取回 `None`，有限数取回 `Some`；类型不符或非有限数报错。
fn optional_number_field(object: &[(String, Value)], key: &str) -> Result<Option<f64>> {
    let value = field(object, key)?;
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => finite_number(*number, key),
        other => Err(invalid(format!(
            "字段 `{key}` 期望数值或 null，得到{}",
            other.kind()
        ))),
    }
}

fn finite_number(number: f64, key: &str) -> Result<Option<f64>> {
    if !number.is_finite() {
        return Err(invalid(format!("字段 `{key}` 必须是有限数")));
    }
    Ok(Some(number))
}

fn number_of(value: &Value, field: &str) -> Result<f64> {
    match value {
        Value::Number(number) => Ok(*number),
        Value::Null => Err(invalid(format!("字段 `{field}` 不能是 null"))),
        other => Err(invalid(format!(
            "字段 `{field}` 期望数值，得到{}",
            other.kind()
        ))),
    }
}

fn str_field<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a str> {
    string_of(field(object, key)?)
}

fn arr_field<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a [Value]> {
    field(object, key)?.array()
}

fn nested<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a [(String, Value)]> {
    field(object, key)?.object()
}

/// 台账对象里按键名取字段；缺字段报字段名，不猜默认值。
fn field<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a Value> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| invalid(format!("缺少必填字段 `{key}`")))
}

fn string_of(value: &Value) -> Result<&str> {
    match value {
        Value::String(text) => Ok(text),
        other => Err(invalid(format!("期望字符串，得到{}", other.kind()))),
    }
}

// ---------------------------------------------------------------------------
// 词法读取器
// ---------------------------------------------------------------------------

/// 受限 JSON 值树：只取归档记录用到的几种形态。
#[derive(Debug, Clone, PartialEq)]
enum Value {
    /// `null`。
    Null,
    /// 有限数；整数与小数统一存 `f64`，整数口径在台账层再校。
    Number(f64),
    /// 字符串；转义与代理对已解码。
    String(String),
    /// 数组。
    Array(Vec<Value>),
    /// 对象：保持书写顺序，重复键在解析期即被拒绝。
    Object(Vec<(String, Value)>),
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Self::Array(_) => "数组",
            Self::Object(_) => "对象",
            Self::Null => "null",
            Self::Number(_) => "数值",
            Self::String(_) => "字符串",
        }
    }

    fn object(&self) -> Result<&[(String, Value)]> {
        match self {
            Self::Object(fields) => Ok(fields),
            other => Err(invalid(format!("期望对象，得到{}", other.kind()))),
        }
    }

    fn array(&self) -> Result<&[Value]> {
        match self {
            Self::Array(items) => Ok(items),
            other => Err(invalid(format!("期望数组，得到{}", other.kind()))),
        }
    }
}

/// 解析一段完整的 JSON 文本。
fn parse(text: &str) -> Result<Value> {
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

    fn fail(&self, message: impl Into<String>) -> Error {
        self.fail_at(self.pos, message)
    }

    fn fail_at(&self, position: usize, message: impl Into<String>) -> Error {
        located(position, message)
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
        self.by_opening(opened, depth)
    }

    /// 按开头字节分派：标量就地转换，结构化值交给容器读取器。
    fn by_opening(&mut self, opened: u8, depth: usize) -> Result<Value> {
        match opened {
            b'"' => self.string().map(Value::String),
            b'-' | b'0'..=b'9' => self.number(),
            b'n' => self.literal("null", Value::Null),
            _ => self.container(opened, depth),
        }
    }

    fn container(&mut self, opened: u8, depth: usize) -> Result<Value> {
        match opened {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            _ => {
                Err(self.fail("无法识别的值开头（归档记录不接受布尔、NaN / Infinity 或裸标识符）"))
            }
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

    fn member(
        &mut self,
        open: usize,
        depth: usize,
        seen: &[(String, Value)],
    ) -> Result<((String, Value), bool)> {
        let field = self.field(depth, seen)?;
        Ok((field, self.separator(open, b'}', "对象")?))
    }

    fn element(&mut self, open: usize, depth: usize) -> Result<(Value, bool)> {
        let value = self.value(depth + 1)?;
        Ok((value, self.separator(open, b']', "数组")?))
    }

    fn close(&mut self, closing: u8) -> bool {
        self.skip_space();
        if self.peek() != Some(closing) {
            return false;
        }
        self.pos += 1;
        true
    }

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

    fn fragment(&mut self, open: usize, at: usize, byte: u8) -> Result<String> {
        match byte {
            b'\\' => self.escape(open, at).map(String::from),
            0x00..=0x1f => Err(self.fail_at(at, "控制字符必须转义")),
            _ => self.utf8(at),
        }
    }

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

    fn hex_digit(&mut self, at: usize) -> Result<u32> {
        let byte = self
            .bump()
            .ok_or_else(|| self.fail_at(at, "`\\u` 需要 4 位十六进制"))?;
        (byte as char)
            .to_digit(16)
            .ok_or_else(|| self.fail_at(at, "`\\u` 需要 4 位十六进制"))
    }

    /// 读一个 JSON 数值：按文法切片，`f64` 解析后必须有限。
    fn number(&mut self) -> Result<Value> {
        let text = self.number_text()?.to_string();
        let at = self.pos;
        let number: f64 = text
            .parse()
            .map_err(|_| self.fail_at(at, format!("数值 `{text}` 无法解析")))?;
        if !number.is_finite() {
            return Err(self.fail_at(at, "数值必须是有限数（不接受 NaN / Infinity）"));
        }
        Ok(Value::Number(number))
    }

    /// 按 JSON 数词法切出字节片：符号、整数、小数与指数分段推进。
    fn number_text(&mut self) -> Result<&str> {
        let start = self.pos;
        self.number_sign()?;
        self.int_part(start)?;
        self.fraction()?;
        self.exponent()?;
        std::str::from_utf8(self.bytes.get(start..self.pos).unwrap_or_default())
            .map_err(|_| self.fail_at(start, "数值不是 ASCII"))
    }

    fn number_sign(&mut self) -> Result<()> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        Ok(())
    }

    /// 整数部分：`0` 或非零开头的一串数字；前导零在本地拒。
    fn int_part(&mut self, start: usize) -> Result<()> {
        let digit_at = self.pos;
        let first = self
            .peek()
            .ok_or_else(|| self.fail_at(digit_at, "数值整数部分缺少数字"))?;
        if first == b'0' {
            self.pos += 1;
            return self.reject_leading_zero(start);
        }
        if !matches!(first, b'1'..=b'9') {
            return Err(self.fail_at(digit_at, "数值整数部分缺少数字"));
        }
        self.consume_digits();
        Ok(())
    }

    fn reject_leading_zero(&mut self, start: usize) -> Result<()> {
        if matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(self.fail_at(start, "数值整数部分不能有前导零"));
        }
        Ok(())
    }

    fn consume_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
    }

    fn fraction(&mut self) -> Result<()> {
        if self.peek() != Some(b'.') {
            return Ok(());
        }
        self.pos += 1;
        let at = self.pos;
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(self.fail_at(at, "小数部分缺少数字"));
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        Ok(())
    }

    fn exponent(&mut self) -> Result<()> {
        if !matches!(self.peek(), Some(b'e' | b'E')) {
            return Ok(());
        }
        self.pos += 1;
        if matches!(self.peek(), Some(b'+' | b'-')) {
            self.pos += 1;
        }
        let at = self.pos;
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(self.fail_at(at, "指数部分缺少数字"));
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        Ok(())
    }
}

/// 带位置前缀的解析错误；台账层错误不带位置，另走 [`invalid`]。
fn located(at: usize, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("位置 {at}：{}", message.into()))
        .with("tool::bench_json::from_json")
}

/// 台账 / 类型校验错误；类别与上下文沿用 bench 层口径。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("tool::bench_json::from_json")
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

fn simple_escape(byte: u8) -> Option<char> {
    SIMPLE.get(byte as usize).copied().filter(|&ch| ch != '\0')
}

fn sequence_width(byte: u8) -> Option<usize> {
    match byte {
        0x00..=0x7f => Some(1),
        0xc0..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf7 => Some(4),
        _ => None,
    }
}

fn code_point(at: usize, code: u32) -> Result<char> {
    char::from_u32(code).ok_or_else(|| located(at, format!("U+{code:04X} 不是合法字符")))
}

fn is_high_surrogate(code: u32) -> bool {
    (0xd800..=0xdbff).contains(&code)
}

fn is_low_surrogate(code: u32) -> bool {
    (0xdc00..=0xdfff).contains(&code)
}
