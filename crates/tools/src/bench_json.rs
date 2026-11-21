//! [`BenchRecord`] 的受限 JSON 写出与回读（M2-4）。
//!
//! 全仓零外部依赖，所以这里手写一个只服务归档记录的读写层：写出侧只暴露记录已校验过的
//! 字段，未测量的实测与未测得的 roofline 上界写 `null`；回读侧按固定台账边扫边取字段，
//! 不保留通用对象 / 数组值树，也不按键名查表，并在解析期拒绝重复键、嵌套过深、截断输入与
//! 非有限数（`NaN` / `Infinity` 不是 JSON 词法）。
//!
//! **不拥有**：采样、基线结论、跨机对照；数值口径沿用 [`bench`](crate::bench) 各类型的构造自检。

use diagnostic::scan::Cursor;
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
// 回读：面向固定台账的流式读取器
// ---------------------------------------------------------------------------

/// 回读一段完整 JSON 文本，按归档记录台账取回一条 [`BenchRecord`]。
///
/// # Errors
///
/// 词法 / 文法错误、重复键、嵌套过深、截断输入、非有限数、缺必填字段或类型不符，一律
/// [`Code::Invalid`]；解析期错误消息带出错字节位置，台账层错误消息带字段名。
pub fn from_json(text: &str) -> Result<BenchRecord> {
    let mut reader = Reader::new(text);
    let record = reader.read_record()?;
    reader.cursor.skip_space();
    if reader.cursor.pos() != reader.cursor.bytes().len() {
        return Err(reader.cursor.fail("记录之后还有多余内容"));
    }
    Ok(record)
}

/// 台账草稿：顶层七个槽位，逐一在遍历时就地填好；`None` 表示该键尚未出现。
struct RecordDraft {
    kind: Option<EntryKind>,
    name: Option<String>,
    backend: Option<String>,
    geometry: Option<Geometry>,
    harness: Option<Harness>,
    roofline: Option<Roofline>,
    observed: Option<Option<f64>>,
}

impl RecordDraft {
    fn new() -> Self {
        Self {
            kind: None,
            name: None,
            backend: None,
            geometry: None,
            harness: None,
            roofline: None,
            observed: None,
        }
    }

    /// 七个槽位到齐后交回记录构造自检；`observed` 外层 `None` 表示键缺失。
    fn build(mut self) -> Result<BenchRecord> {
        let record = self.assemble()?;
        self.fill_observed(record)
    }

    fn assemble(&mut self) -> Result<BenchRecord> {
        let head = self.head()?;
        let rest = self.rest()?;
        BenchRecord::new(head.0, head.1, head.2, rest.0, rest.1, rest.2)
    }

    fn head(&mut self) -> Result<(EntryKind, String, String)> {
        Ok((
            require(self.kind.take(), "kind")?,
            require(self.name.take(), "name")?,
            require(self.backend.take(), "backend")?,
        ))
    }

    fn rest(&mut self) -> Result<(Geometry, Harness, Roofline)> {
        Ok((
            require(self.geometry.take(), "geometry")?,
            require(self.harness.take(), "harness")?,
            require(self.roofline.take(), "roofline")?,
        ))
    }

    fn fill_observed(self, record: BenchRecord) -> Result<BenchRecord> {
        match self.observed {
            Some(Some(number)) => record.with_observed(number),
            Some(None) => Ok(record),
            None => Err(invalid("缺少必填字段 `observed`")),
        }
    }
}

/// 几何草稿：形状、批次、数据类型三个槽位。
struct GeometryDraft {
    shape: Option<Vec<usize>>,
    batch: Option<usize>,
    dtype: Option<String>,
}

impl GeometryDraft {
    fn new() -> Self {
        Self {
            shape: None,
            batch: None,
            dtype: None,
        }
    }

    fn build(self) -> Result<Geometry> {
        let shape = require(self.shape, "shape")?;
        let batch = require(self.batch, "batch")?;
        let dtype = require(self.dtype, "dtype")?;
        let batch = u32::try_from(batch).map_err(|_| invalid("`batch` 超出 u32 范围"))?;
        Geometry::new(shape, batch, dtype)
    }
}

/// Roofline 草稿：两个可选峰值槽位。
struct RooflineSlots {
    bw: Option<Option<f64>>,
    gflops: Option<Option<f64>>,
}

impl RooflineSlots {
    fn new() -> Self {
        Self {
            bw: None,
            gflops: None,
        }
    }
}

/// 档案草稿：六个身份字段逐一填好。
struct HarnessDraft {
    host: Option<String>,
    cpu: Option<String>,
    os: Option<String>,
    arch: Option<String>,
    toolchain: Option<String>,
    driver: Option<String>,
}

impl HarnessDraft {
    fn new() -> Self {
        Self {
            host: None,
            cpu: None,
            os: None,
            arch: None,
            toolchain: None,
            driver: None,
        }
    }

    fn build(mut self) -> Result<Harness> {
        let head = self.head()?;
        let tail = self.tail()?;
        Ok(Harness::new(head.0, head.1, head.2, tail.0, tail.1, tail.2))
    }

    fn head(&mut self) -> Result<(String, String, String)> {
        Ok((
            require(self.host.take(), "host")?,
            require(self.cpu.take(), "cpu")?,
            require(self.os.take(), "os")?,
        ))
    }

    fn tail(&mut self) -> Result<(String, String, String)> {
        Ok((
            require(self.arch.take(), "arch")?,
            require(self.toolchain.take(), "toolchain")?,
            require(self.driver.take(), "driver")?,
        ))
    }
}

/// 逐字节推进的读取器：[`Cursor`] 负责字节与位置（位置同时是报错坐标），这里只多记一层
/// 容器深度。
struct Reader<'a> {
    cursor: Cursor<'a>,
    depth: usize,
}

impl<'a> Reader<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            cursor: Cursor::new(text, "tools::bench_json::from_json"),
            depth: 0,
        }
    }

    /// 开一个容器：先查深度，再消费开头字节；离开容器时调用方把 `depth` 减回。
    fn open_container(&mut self, opening: u8) -> Result<()> {
        if self.depth >= MAX_DEPTH {
            return Err(self.cursor.fail(format!("嵌套超过 {MAX_DEPTH} 层")));
        }
        self.depth += 1;
        self.cursor.eat(opening)
    }

    /// 容器已结束：跳过空白后若撞上收尾字节就消费掉，返回 `false`；否则返回 `true` 继续。
    fn close(&mut self, closing: u8) -> bool {
        self.cursor.skip_space();
        if self.cursor.peek() != Some(closing) {
            return false;
        }
        self.cursor.advance(1);
        true
    }

    /// 成员键：读字符串、查重、消费冒号，键名留在草稿分派。
    fn member_key(&mut self, seen: &mut Vec<String>) -> Result<String> {
        self.cursor.skip_space();
        let at = self.cursor.pos();
        let key = self.read_string()?;
        if seen.iter().any(|existing| existing == &key) {
            return Err(self.cursor.fail_at(at, format!("对象里出现重复键 `{key}`")));
        }
        seen.push(key.clone());
        self.cursor.skip_space();
        self.cursor.eat(b':')?;
        self.cursor.skip_space();
        Ok(key)
    }

    /// 成员之间：逗号续读；收尾括号留给循环的 `close` 消费，这里只做预判。
    fn require_next(&mut self, close: u8, what: &str) -> Result<()> {
        self.cursor.skip_space();
        match self.cursor.peek() {
            Some(b',') => {
                self.cursor.advance(1);
                Ok(())
            }
            Some(byte) if byte == close => Ok(()),
            Some(_) => Err(self.cursor.fail_at(
                self.cursor.pos(),
                format!("{what}里期望 `,` 或 `{}`", close as char),
            )),
            None => Err(self
                .cursor
                .fail_at(self.cursor.pos(), format!("{what}没有闭合"))),
        }
    }

    /// 遍历一个对象的成员：键交给 `visit` 分派到草稿槽位。
    fn walk_object<T>(
        &mut self,
        close: u8,
        what: &str,
        seen: &mut Vec<String>,
        ctx: &mut T,
        visit: fn(&mut Self, &str, &mut T) -> Result<()>,
    ) -> Result<()> {
        while !self.close(close) {
            let key = self.member_key(seen)?;
            visit(self, &key, ctx)?;
            self.require_next(close, what)?;
        }
        Ok(())
    }

    // ---- 台账结构 ---------------------------------------------------------

    fn read_record(&mut self) -> Result<BenchRecord> {
        self.open_container(b'{')?;
        let mut draft = RecordDraft::new();
        self.read_top_level(&mut draft)?;
        draft.build()
    }

    fn read_top_level(&mut self, draft: &mut RecordDraft) -> Result<()> {
        let mut seen = Vec::new();
        self.walk_object(b'}', "记录", &mut seen, draft, Self::visit_top)
    }

    fn visit_top(&mut self, key: &str, draft: &mut RecordDraft) -> Result<()> {
        self.assign_top_level(key, draft)
    }

    fn assign_top_level(&mut self, key: &str, draft: &mut RecordDraft) -> Result<()> {
        match key {
            "kind" => self.fill_kind(draft),
            "name" => self.fill_str(&mut draft.name),
            "backend" => self.fill_str(&mut draft.backend),
            "observed" => self.fill_observed_slot(draft),
            other => self.assign_nested_top(other, draft),
        }
    }

    fn assign_nested_top(&mut self, key: &str, draft: &mut RecordDraft) -> Result<()> {
        match key {
            "geometry" => self.fill_geometry(draft),
            "harness" => self.fill_harness(draft),
            "roofline" => self.fill_roofline(draft),
            other => Err(invalid(format!("未知字段 `{other}`"))),
        }
    }

    /// 把读到的字符串填进一个字符串槽位。
    fn fill_str(&mut self, slot: &mut Option<String>) -> Result<()> {
        *slot = Some(self.read_string()?);
        Ok(())
    }

    fn fill_kind(&mut self, draft: &mut RecordDraft) -> Result<()> {
        draft.kind = Some(self.read_kind()?);
        Ok(())
    }

    fn fill_observed_slot(&mut self, draft: &mut RecordDraft) -> Result<()> {
        draft.observed = Some(self.read_optional_number()?);
        Ok(())
    }

    fn fill_geometry(&mut self, draft: &mut RecordDraft) -> Result<()> {
        draft.geometry = Some(self.read_geometry()?);
        Ok(())
    }

    fn fill_harness(&mut self, draft: &mut RecordDraft) -> Result<()> {
        draft.harness = Some(self.read_harness()?);
        Ok(())
    }

    fn fill_roofline(&mut self, draft: &mut RecordDraft) -> Result<()> {
        draft.roofline = Some(self.read_roofline()?);
        Ok(())
    }

    fn read_geometry(&mut self) -> Result<Geometry> {
        self.open_container(b'{')?;
        let mut draft = GeometryDraft::new();
        self.read_geometry_members(&mut draft)?;
        self.depth -= 1;
        draft.build()
    }

    fn read_geometry_members(&mut self, draft: &mut GeometryDraft) -> Result<()> {
        let mut seen = Vec::new();
        self.walk_object(b'}', "几何", &mut seen, draft, Self::visit_geometry)
    }

    fn visit_geometry(&mut self, key: &str, draft: &mut GeometryDraft) -> Result<()> {
        self.assign_geometry(key, draft)
    }

    fn assign_geometry(&mut self, key: &str, draft: &mut GeometryDraft) -> Result<()> {
        match key {
            "shape" => self.fill_shape(draft),
            "batch" => self.fill_batch(draft),
            "dtype" => self.fill_str(&mut draft.dtype),
            other => Err(invalid(format!("几何对象里未知字段 `{other}`"))),
        }
    }

    fn fill_shape(&mut self, draft: &mut GeometryDraft) -> Result<()> {
        draft.shape = Some(self.read_shape()?);
        Ok(())
    }

    fn fill_batch(&mut self, draft: &mut GeometryDraft) -> Result<()> {
        draft.batch = Some(self.read_usize("batch")?);
        Ok(())
    }

    fn read_shape(&mut self) -> Result<Vec<usize>> {
        self.open_container(b'[')?;
        let mut shape = Vec::new();
        while !self.close(b']') {
            self.push_shape_item(&mut shape)?;
            self.require_next(b']', "形状")?;
        }
        self.depth -= 1;
        Ok(shape)
    }

    fn push_shape_item(&mut self, shape: &mut Vec<usize>) -> Result<()> {
        let index = shape.len();
        let number = self.read_number()?;
        let dim = dim_usize(number, &format!("shape[{index}]"))?;
        shape.push(dim);
        Ok(())
    }

    fn read_harness(&mut self) -> Result<Harness> {
        self.open_container(b'{')?;
        let mut draft = HarnessDraft::new();
        self.read_harness_members(&mut draft)?;
        self.depth -= 1;
        draft.build()
    }

    fn read_harness_members(&mut self, draft: &mut HarnessDraft) -> Result<()> {
        let mut seen = Vec::new();
        self.walk_object(b'}', "档案", &mut seen, draft, Self::visit_harness)
    }

    fn visit_harness(&mut self, key: &str, draft: &mut HarnessDraft) -> Result<()> {
        self.assign_harness(key, draft)
    }

    fn assign_harness(&mut self, key: &str, draft: &mut HarnessDraft) -> Result<()> {
        match key {
            "host" => self.fill_str(&mut draft.host),
            "cpu" => self.fill_str(&mut draft.cpu),
            "os" => self.fill_str(&mut draft.os),
            other => self.assign_harness_tail(other, draft),
        }
    }

    fn assign_harness_tail(&mut self, key: &str, draft: &mut HarnessDraft) -> Result<()> {
        match key {
            "arch" => self.fill_str(&mut draft.arch),
            "toolchain" => self.fill_str(&mut draft.toolchain),
            "driver" => self.fill_str(&mut draft.driver),
            other => Err(invalid(format!("档案对象里未知字段 `{other}`"))),
        }
    }

    fn read_roofline(&mut self) -> Result<Roofline> {
        self.open_container(b'{')?;
        let mut slots = RooflineSlots::new();
        let mut seen = Vec::new();
        self.walk_object(
            b'}',
            "roofline",
            &mut seen,
            &mut slots,
            Self::visit_roofline,
        )?;
        self.depth -= 1;
        Roofline::new(slots.bw.flatten(), slots.gflops.flatten())
    }

    fn visit_roofline(&mut self, key: &str, slots: &mut RooflineSlots) -> Result<()> {
        self.assign_roofline(key, slots)
    }

    fn assign_roofline(&mut self, key: &str, slots: &mut RooflineSlots) -> Result<()> {
        match key {
            "peak_bw_gb_s" => self.fill_bw(slots),
            "peak_gflops" => self.fill_gflops(slots),
            other => Err(invalid(format!("roofline 未知字段 `{other}`"))),
        }
    }

    fn fill_bw(&mut self, slots: &mut RooflineSlots) -> Result<()> {
        slots.bw = Some(self.read_optional_number()?);
        Ok(())
    }

    fn fill_gflops(&mut self, slots: &mut RooflineSlots) -> Result<()> {
        slots.gflops = Some(self.read_optional_number()?);
        Ok(())
    }

    // ---- 标量 ------------------------------------------------------------

    fn read_kind(&mut self) -> Result<EntryKind> {
        let text = self.read_string()?;
        match text.as_str() {
            "op" => Ok(EntryKind::Op),
            "layer" => Ok(EntryKind::Layer),
            "model" => Ok(EntryKind::Model),
            other => Err(invalid(format!(
                "条目层级 `{other}` 不在 op / layer / model 之内"
            ))),
        }
    }

    fn read_usize(&mut self, field: &str) -> Result<usize> {
        let number = self.read_number()?;
        dim_usize(number, field)
    }

    /// 可选数值：`null` 取回 `None`，有限数取回 `Some`；类型不符或非有限数报错。
    fn read_optional_number(&mut self) -> Result<Option<f64>> {
        if self.cursor.peek() == Some(b'n') {
            self.read_null()?;
            return Ok(None);
        }
        Ok(Some(self.read_number()?))
    }

    fn read_null(&mut self) -> Result<()> {
        let at = self.cursor.pos();
        let end = at + 4;
        let Some(slice) = self.cursor.bytes().get(at..end) else {
            return Err(self.cursor.fail_at(at, "`null` 不完整"));
        };
        if slice != b"null" {
            return Err(self.cursor.fail_at(
                at,
                "无法识别的值开头（归档记录不接受布尔、NaN / Infinity 或裸标识符）",
            ));
        }
        self.cursor.seek(end);
        Ok(())
    }

    /// 读一个 JSON 字符串；转义与 UTF-8 就地解码，不先落通用值树。
    fn read_string(&mut self) -> Result<String> {
        let open = self.cursor.pos();
        self.cursor.eat(b'"')?;
        let mut text = String::new();
        loop {
            let Some(byte) = self.cursor.bump() else {
                return Err(self.cursor.fail_at(open, "字符串没有闭合"));
            };
            if self.advance_string(&mut text, byte)? {
                break;
            }
        }
        Ok(text)
    }

    /// 推进一个字符串字节；返回 `true` 表示撞上收尾引号。
    fn advance_string(&mut self, text: &mut String, byte: u8) -> Result<bool> {
        match byte {
            b'"' => Ok(true),
            b'\\' => self.push_escape(text),
            0x00..=0x1f => Err(self
                .cursor
                .fail_at(self.cursor.pos() - 1, "控制字符必须转义")),
            _ => self.push_utf8(text, byte),
        }
    }

    fn push_escape(&mut self, text: &mut String) -> Result<bool> {
        text.push(self.read_escape()?);
        Ok(false)
    }

    fn push_utf8(&mut self, text: &mut String, first: u8) -> Result<bool> {
        text.push_str(self.utf8_run(first)?);
        Ok(false)
    }

    fn read_escape(&mut self) -> Result<char> {
        let at = self.cursor.pos();
        let Some(byte) = self.cursor.bump() else {
            return Err(self.cursor.fail_at(at, "转义序列不完整"));
        };
        match byte {
            b'u' => Err(self.cursor.fail_at(at, "本记录字符串不接受 `\\u` 转义")),
            _ => simple_escape(byte).ok_or_else(|| {
                self.cursor
                    .fail_at(at, format!("未知转义 `\\{}`", byte as char))
            }),
        }
    }

    /// 取出自当前 `pos` 起的一个合法 UTF-8 字符片段，并把指针推到片段末尾。
    fn utf8_run(&mut self, first: u8) -> Result<&str> {
        let Some(width) = sequence_width(first) else {
            return Err(self
                .cursor
                .fail_at(self.cursor.pos() - 1, "非法的 UTF-8 起始字节"));
        };
        let end = self.cursor.pos() - 1 + width;
        let Some(slice) = self.cursor.bytes().get(self.cursor.pos() - 1..end) else {
            return Err(self
                .cursor
                .fail_at(self.cursor.pos() - 1, "多字节字符被截断"));
        };
        let text = std::str::from_utf8(slice).map_err(|_| {
            self.cursor
                .fail_at(self.cursor.pos() - 1, "非法的 UTF-8 序列")
        })?;
        self.cursor.seek(end);
        Ok(text)
    }

    /// 读一个 JSON 数值：按文法切片，`f64` 解析后必须有限。
    fn read_number(&mut self) -> Result<f64> {
        self.cursor.skip_space();
        let text = self.number_text()?;
        self.parse_finite(&text)
    }

    fn number_text(&mut self) -> Result<String> {
        let start = self.cursor.pos();
        if self.cursor.peek() == Some(b'-') {
            self.cursor.advance(1);
        }
        self.consume_int(start)?;
        self.consume_fraction()?;
        self.consume_exponent()?;
        self.slice_text(start)
    }

    fn slice_text(&self, start: usize) -> Result<String> {
        let text = std::str::from_utf8(
            self.cursor
                .bytes()
                .get(start..self.cursor.pos())
                .unwrap_or_default(),
        )
        .map_err(|_| self.cursor.fail_at(start, "数值不是 ASCII"))?;
        Ok(text.to_string())
    }

    fn parse_finite(&self, text: &str) -> Result<f64> {
        let at = self.cursor.pos();
        let number: f64 = text
            .parse()
            .map_err(|_| self.cursor.fail_at(at, format!("数值 `{text}` 无法解析")))?;
        if !number.is_finite() {
            return Err(self
                .cursor
                .fail_at(at, "数值必须是有限数（不接受 NaN / Infinity）"));
        }
        Ok(number)
    }

    fn consume_int(&mut self, start: usize) -> Result<()> {
        let at = self.cursor.pos();
        let Some(first) = self.cursor.peek() else {
            return Err(self.cursor.fail_at(at, "数值整数部分缺少数字"));
        };
        match first {
            b'0' => {
                self.cursor.advance(1);
                if matches!(self.cursor.peek(), Some(b'0'..=b'9')) {
                    return Err(self.cursor.fail_at(start, "数值整数部分不能有前导零"));
                }
            }
            b'1'..=b'9' => self.consume_digits(),
            _ => return Err(self.cursor.fail_at(at, "数值整数部分缺少数字")),
        }
        Ok(())
    }

    fn consume_digits(&mut self) {
        while matches!(self.cursor.peek(), Some(b'0'..=b'9')) {
            self.cursor.advance(1);
        }
    }

    fn consume_fraction(&mut self) -> Result<()> {
        if self.cursor.peek() != Some(b'.') {
            return Ok(());
        }
        self.cursor.advance(1);
        let at = self.cursor.pos();
        if !matches!(self.cursor.peek(), Some(b'0'..=b'9')) {
            return Err(self.cursor.fail_at(at, "小数部分缺少数字"));
        }
        self.consume_digits();
        Ok(())
    }

    fn consume_exponent(&mut self) -> Result<()> {
        if !matches!(self.cursor.peek(), Some(b'e' | b'E')) {
            return Ok(());
        }
        self.cursor.advance(1);
        if matches!(self.cursor.peek(), Some(b'+' | b'-')) {
            self.cursor.advance(1);
        }
        let at = self.cursor.pos();
        if !matches!(self.cursor.peek(), Some(b'0'..=b'9')) {
            return Err(self.cursor.fail_at(at, "指数部分缺少数字"));
        }
        self.consume_digits();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 台账 / 词法辅助
// ---------------------------------------------------------------------------

/// 数组 / 形状维度：非负整数；小数、负数与越界整数都在这里拒。
fn dim_usize(number: f64, field: &str) -> Result<usize> {
    if number < 0.0 || number.fract() != 0.0 {
        return Err(invalid(format!("字段 `{field}` 必须是非负整数")));
    }
    let integer = number as u128;
    if integer > usize::MAX as u128 {
        return Err(invalid(format!("字段 `{field}` 超出 usize 范围")));
    }
    Ok(integer as usize)
}

fn require<T>(option: Option<T>, field: &str) -> Result<T> {
    option.ok_or_else(|| invalid(format!("缺少必填字段 `{field}`")))
}

/// 带位置前缀的解析错误；台账层错误不带位置，另走 [`invalid`]。
/// 台账 / 类型校验错误；类别与上下文沿用 bench 层口径。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("tools::bench_json::from_json")
}

/// `\X` 的第二个字节 → 字符；表里没有就是未知转义。
fn simple_escape(byte: u8) -> Option<char> {
    match byte {
        b'"' => Some('"'),
        b'\\' => Some('\\'),
        b'/' => Some('/'),
        other => escape_padding(other),
    }
}

fn escape_padding(byte: u8) -> Option<char> {
    match byte {
        b'b' => Some('\u{8}'),
        b'f' => Some('\u{c}'),
        other => escape_line(other),
    }
}

fn escape_line(byte: u8) -> Option<char> {
    match byte {
        b'n' => Some('\n'),
        b'r' => Some('\r'),
        b't' => Some('\t'),
        _ => None,
    }
}

/// UTF-8 起始字节 → 序列宽度；`None` 表示非法起始字节。
fn sequence_width(byte: u8) -> Option<usize> {
    match byte.leading_ones() {
        0 => Some(1),
        2 => Some(2),
        3 => Some(3),
        4 => Some(4),
        _ => None,
    }
}
