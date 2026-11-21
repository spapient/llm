//! 账本正文的文本形态：一端 [`Settled::render`] 写，一端 [`verify_root`] 核回。
//!
//! 从 [`crate::settle`] 拆出：那里的 [`crate::settle::Settlement::render`] 只负责拼外壳，这里负责
//! 「一格里该写什么字节」以及「读回来的那格是不是同一份」，写作与校验因此永远在同一份代码里。

use std::fmt::Write as _;

use diagnostic::Result;

use crate::bad;
use crate::json::Value;
use crate::ledger::SCHEMA;
use crate::settle::{Settled, Settlement};

impl Settled {
    /// 单条目的规范 JSON（六格缩进，与 [`Settlement::render`] 拼在一起）。
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("    {\n");
        let _ = writeln!(out, "      \"name\": \"{}\",", escape(&self.name));
        let _ = writeln!(out, "      \"layer\": {},", layer_json(self.layer));
        let _ = writeln!(out, "      \"raw_len\": {},", self.raw_len);
        let _ = writeln!(out, "      \"volume_bytes\": {},", self.volume);
        let _ = writeln!(out, "      \"volume_ratio\": {:.6},", self.volume_ratio());
        let _ = writeln!(out, "      \"entropy_H\": {:.6},", self.entropy);
        let (bits, group) = self.quant.unwrap_or((0, 0));
        let _ = writeln!(out, "      \"quant_bits\": {bits},");
        let _ = writeln!(out, "      \"quant_group_size\": {group},");
        let _ = writeln!(out, "      \"verify_bitwise_equal\": {},", self.bitwise);
        let _ = writeln!(out, "      \"is_lossless\": {},", self.lossless);
        let _ = writeln!(out, "      \"decision\": \"{}\",", self.decision());
        let _ = writeln!(out, "      \"codec_reason\": \"{}\",", self.reason.as_str());
        let _ = writeln!(out, "      \"cert\": \"{}\"", self.cert.as_str());
        out.push_str("    }");
        out
    }
}

/// 层号的 JSON 写法：没有层号就写 `null`，不写 0（0 是真实层号，混同会丢信息）。
pub(crate) fn layer_json(layer: Option<u32>) -> String {
    match layer {
        Some(layer) => layer.to_string(),
        None => "null".to_string(),
    }
}

/// JSON 字符串转义：张量名来自文件，不能默认"里面没有引号与控制字符"。
pub(crate) fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        push_escaped(&mut out, ch);
    }
    out
}

/// 逐字符转义：引号与反斜杠必须转义，换行用短形式，其余控制字符留给 [`push_char`]。
fn push_escaped(out: &mut String, ch: char) {
    match ch {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        other => push_char(out, other),
    }
}

/// 控制字符走 `\u00xx`（JSON 合法形式），其余原样。
fn push_char(out: &mut String, ch: char) {
    if (ch as u32) < 0x20 {
        let _ = write!(out, "\\u{:04x}", ch as u32);
    } else {
        out.push(ch);
    }
}

/// 顶层三格 + 条目数：多一格、少一格、数目不符都在这里作答。
pub(crate) fn verify_root(root: &[(String, Value)], settled: &Settlement) -> Result<()> {
    verify_header(root, settled)?;
    let listed = field(root, "tensors", "账本")?.array()?;
    verify_count(listed, settled.tensors().len())?;
    verify_rows(listed, settled.tensors())
}

/// 段表逐条比对：数目已在本层之前对齐，这里只问「每条账与它那一行是否同源」。
fn verify_rows(listed: &[Value], entries: &[Settled]) -> Result<()> {
    entries
        .iter()
        .zip(listed)
        .try_for_each(|(entry, row)| verify_row(row.object()?, entry))
}

/// 顶层的两个标量格：`schema` 与 `file_len` 都得按规范写法与这份字节相等。
fn verify_header(root: &[(String, Value)], settled: &Settlement) -> Result<()> {
    only(root, &["schema", "file_len", "tensors"], "账本")?;
    check(root, "schema", SCHEMA, "账本")?;
    check(root, "file_len", &settled.file_len().to_string(), "账本")
}

/// 条目数必须与段表一致：多一条、少一条都是错位的账。
fn verify_count(listed: &[Value], actual: usize) -> Result<()> {
    if listed.len() == actual {
        return Ok(());
    }
    Err(bad(format!(
        "账本列了 {} 个张量，这份字节有 {actual} 段",
        listed.len()
    )))
}

/// 一个条目的十二格：先挡掉多余的格，再逐格比规范写法。
fn verify_row(fields: &[(String, Value)], entry: &Settled) -> Result<()> {
    let want = expected(entry);
    let names: Vec<&str> = want.iter().map(|(key, _)| *key).collect();
    only(fields, &names, entry.name())?;
    for (key, value) in &want {
        check(fields, key, value, entry.name())?;
    }
    Ok(())
}

/// 键集合不得超出 `allowed`：账本只能描述这份字节里有的格。
fn only(fields: &[(String, Value)], allowed: &[&str], owner: &str) -> Result<()> {
    match fields
        .iter()
        .find(|(name, _)| !allowed.contains(&name.as_str()))
    {
        None => Ok(()),
        Some((extra, _)) => Err(bad(format!(
            "{owner} 的账本多出 `{extra}`：这份字节没有这一格"
        ))),
    }
}

/// 一格账的比对：键必须在、且按**规范写法**逐字等于现算值。
fn check(fields: &[(String, Value)], key: &str, want: &str, owner: &str) -> Result<()> {
    let text = written(field(fields, key, owner)?)?;
    if text == want {
        return Ok(());
    }
    Err(bad(format!(
        "{owner} 的 `{key}` 是 `{text}`，这份字节现算的是 `{want}`"
    )))
}

/// 按名取一格；缺键就是账本不完整（不是「这格可以不写」）。
fn field<'a>(fields: &'a [(String, Value)], key: &str, owner: &str) -> Result<&'a Value> {
    fields
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| bad(format!("{owner} 的账本缺 `{key}`")))
}

/// 值在文本里的写法：字符串取内容、布尔与 `null` 取字面、数字取原文。
///
/// 与 [`Settled::render`] 的写法是同一份约定——两边各写一套，逐格比对就变成两种格式之争。
fn written(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Bool(flag) => Ok(flag.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Integer(_) | Value::Decimal(_) => value.number_text(),
        other => Err(bad(format!("账本的这一格必须是标量，得到{}", other.kind()))),
    }
}

/// 一条目期望的十三格：键名与 [`Settled::render`] 同源同序，值按规范写法成文。
fn expected(entry: &Settled) -> [(&'static str, String); 13] {
    let (bits, group) = entry.quant().unwrap_or((0, 0));
    [
        ("name", entry.name().to_string()),
        ("layer", layer_json(entry.layer())),
        ("raw_len", entry.raw_len().to_string()),
        ("volume_bytes", entry.volume().to_string()),
        ("volume_ratio", format!("{:.6}", entry.volume_ratio())),
        ("entropy_H", format!("{:.6}", entry.entropy())),
        ("quant_bits", bits.to_string()),
        ("quant_group_size", group.to_string()),
        ("verify_bitwise_equal", entry.bitwise().to_string()),
        ("is_lossless", entry.lossless().to_string()),
        ("decision", entry.decision().to_string()),
        ("codec_reason", entry.reason().as_str().to_string()),
        ("cert", entry.cert().as_str().to_string()),
    ]
}
