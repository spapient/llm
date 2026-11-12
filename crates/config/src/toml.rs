//! 手写受限 TOML 解析器。
//!
//! 本模块是 workspace 内**唯一**的 TOML 实现：声明层（`ops/*.toml`）、档案层（`devices/*.toml`）
//! 与运行时配置文件（`cutile.toml`）都通过它解析。放在 `llm-config` 而不是某个领域 crate 里，
//! 是因为它是与领域无关的基础设施：任何成员都可以用，且不必因此反向依赖 `cutile`。
//!
//! DESIGN §18 的 D3 决定：P0 不引入 `serde`/`toml`，先用最小解析器证明可行性，再评估
//! 是否换依赖。因此这里只实现声明层真正用到的子集：
//!
//! - `[section]` 与 `[[array-of-tables]]`（可嵌套点号路径）；
//! - `key = value`，键可带点号（`extra_needs.shuffle = {...}`）；
//! - 值：字符串、整数、浮点、布尔、数组、内联表；
//! - `#` 行尾注释（字符串内的 `#` 原样保留）。
//!
//! 刻意不支持的：多行字符串、日期、数字分隔符、数组嵌套数组。遇到即报错并给出行号。

use core::fmt;

/// 解析后的表：保持插入顺序，便于稳定打印。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    entries: Vec<(String, Value)>,
}

/// TOML 值。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 字符串。
    Str(String),
    /// 整数。
    Int(i64),
    /// 浮点。
    Float(f64),
    /// 布尔。
    Bool(bool),
    /// 数组。
    Array(Vec<Value>),
    /// 表（`[section]` 或内联 `{ ... }`）。
    Table(Table),
}

impl Table {
    /// 空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 全部键值对，按插入顺序。
    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    /// 取一个值。
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// 按键取子表。
    pub fn table(&self, key: &str) -> Option<&Table> {
        match self.get(key) {
            Some(Value::Table(table)) => Some(table),
            _ => None,
        }
    }

    /// 按键取字符串。
    pub fn string(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Value::Str(text)) => Some(text),
            _ => None,
        }
    }

    /// 按键取整数。
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(Value::Int(value)) => Some(*value),
            _ => None,
        }
    }

    /// 按键取浮点（整数也接受）。
    pub fn float(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(Value::Float(value)) => Some(*value),
            Some(Value::Int(value)) => Some(*value as f64),
            _ => None,
        }
    }

    /// 按键取布尔。
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(Value::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    /// 按键取数组。
    pub fn array(&self, key: &str) -> Option<&[Value]> {
        match self.get(key) {
            Some(Value::Array(items)) => Some(items),
            _ => None,
        }
    }

    /// 按键取字符串数组；不是数组或含非字符串项即为 `None`。
    pub fn string_array(&self, key: &str) -> Option<Vec<&str>> {
        let items = self.array(key)?;
        items
            .iter()
            .map(|item| match item {
                Value::Str(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// 按键取表数组（`[[variants]]`）。
    pub fn table_array(&self, key: &str) -> Option<Vec<&Table>> {
        let items = self.array(key)?;
        items
            .iter()
            .map(|item| match item {
                Value::Table(table) => Some(table),
                _ => None,
            })
            .collect()
    }

    /// 键是否存在。
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    fn insert(&mut self, key: String, value: Value) {
        match self.entries.iter().position(|(name, _)| name == &key) {
            Some(index) => self.entries[index].1 = value,
            None => self.entries.push((key, value)),
        }
    }

    fn value_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }
}

/// 解析错误：只记行号与说明，不记列（DESIGN §4.2 的精度承诺）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1 起始的行号。
    pub line: usize,
    /// 说明。
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "第 {} 行：{}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// 解析一段 TOML 子集文本。
pub fn parse(source: &str) -> Result<Table, ParseError> {
    let mut root = Table::new();
    let mut section: Vec<String> = Vec::new();

    for (index, raw) in source.lines().enumerate() {
        let line_no = index + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(inner) = line.strip_prefix("[[") {
            let path = inner
                .strip_suffix("]]")
                .ok_or_else(|| err(line_no, "`[[` 缺少配对的 `]]`"))?;
            section = split_path(path);
            push_table_array(&mut root, &section, line_no)?;
            continue;
        }
        if let Some(inner) = line.strip_prefix('[') {
            let path = inner
                .strip_suffix(']')
                .ok_or_else(|| err(line_no, "`[` 缺少配对的 `]`"))?;
            section = split_path(path);
            descend(&mut root, &section, line_no)?;
            continue;
        }

        let (key, raw_value) = line
            .split_once('=')
            .ok_or_else(|| err(line_no, "缺少 `=`：只有 `key = value` 与段落头是合法语句"))?;
        let value_text = raw_value.trim();
        let value = parse_value(value_text)
            .map_err(|message| err(line_no, &format!("值 `{value_text}`：{message}")))?;

        let mut path = section.clone();
        let key = key.trim();
        if key.is_empty() {
            return Err(err(line_no, "键为空"));
        }
        path.extend(split_path(key));
        let (last, parents) = path.split_last().ok_or_else(|| err(line_no, "空键"))?;
        let parent = descend(&mut root, parents, line_no)?;
        parent.insert(last.clone(), value);
    }

    Ok(root)
}

fn err(line: usize, message: &str) -> ParseError {
    ParseError {
        line,
        message: message.to_string(),
    }
}

fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in line.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '#' => return &line[..index],
            _ => {}
        }
    }
    line
}

fn split_path(path: &str) -> Vec<String> {
    path.split('.')
        .map(|segment| segment.trim().trim_matches('"').to_string())
        .filter(|segment| !segment.is_empty())
        .collect()
}

/// 递归下钻：缺失的表自动创建；遇到表数组则进入最后一个元素。
fn descend<'a>(
    table: &'a mut Table,
    keys: &[String],
    line: usize,
) -> Result<&'a mut Table, ParseError> {
    let Some((key, rest)) = keys.split_first() else {
        return Ok(table);
    };
    if !table.contains(key) {
        table.insert(key.clone(), Value::Table(Table::new()));
    }
    let Some(value) = table.value_mut(key) else {
        return Err(err(line, &format!("`{key}` 刚刚创建却取不到")));
    };
    match value {
        Value::Table(child) => descend(child, rest, line),
        Value::Array(items) => match items.last_mut() {
            Some(Value::Table(child)) => descend(child, rest, line),
            _ => Err(err(line, &format!("`{key}` 不是表数组"))),
        },
        Value::Str(_) | Value::Int(_) | Value::Float(_) | Value::Bool(_) => {
            Err(err(line, &format!("`{key}` 已是标量，不能当段落")))
        }
    }
}

/// 在 `keys` 指向的位置新建一个表数组元素。
fn push_table_array(root: &mut Table, keys: &[String], line: usize) -> Result<(), ParseError> {
    let (last, parents) = keys
        .split_last()
        .ok_or_else(|| err(line, "`[[]]` 里没有键名"))?;
    let parent = descend(root, parents, line)?;
    if !parent.contains(last) {
        parent.insert(last.clone(), Value::Array(Vec::new()));
    }
    match parent.value_mut(last) {
        Some(Value::Array(items)) => {
            items.push(Value::Table(Table::new()));
            Ok(())
        }
        Some(_) => Err(err(line, &format!("`{last}` 已被占用，不能作为表数组"))),
        None => Err(err(line, &format!("`{last}` 刚刚创建却取不到"))),
    }
}

fn parse_value(text: &str) -> Result<Value, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("值为空".to_string());
    }
    if let Some(rest) = text.strip_prefix('"') {
        let inner = rest
            .strip_suffix('"')
            .ok_or_else(|| "字符串缺少闭合引号".to_string())?;
        return Ok(Value::Str(unescape(inner)));
    }
    if let Some(rest) = text.strip_prefix('[') {
        let inner = rest
            .strip_suffix(']')
            .ok_or_else(|| "数组缺少闭合 `]`".to_string())?;
        let mut items = Vec::new();
        for part in split_top_level(inner) {
            if part.trim().is_empty() {
                continue;
            }
            items.push(parse_value(&part)?);
        }
        return Ok(Value::Array(items));
    }
    if let Some(rest) = text.strip_prefix('{') {
        let inner = rest
            .strip_suffix('}')
            .ok_or_else(|| "内联表缺少闭合 `}`".to_string())?;
        let mut table = Table::new();
        for part in split_top_level(inner) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (key, raw) = part
                .split_once('=')
                .ok_or_else(|| format!("内联表项 `{part}` 缺少 `=`"))?;
            table.insert(key.trim().to_string(), parse_value(raw)?);
        }
        return Ok(Value::Table(table));
    }
    match text {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        _ => {}
    }
    if let Ok(value) = text.parse::<i64>() {
        return Ok(Value::Int(value));
    }
    if let Ok(value) = text.parse::<f64>() {
        return Ok(Value::Float(value));
    }
    Err("既不是字符串、数字、布尔，也不是数组/内联表".to_string())
}

fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn split_top_level(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if in_string {
            current.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => {
                in_string = true;
                current.push(ch);
            }
            '[' | '{' | '(' => {
                depth += 1;
                current.push(ch);
            }
            ']' | '}' | ')' => {
                depth = depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sections_scalars_and_arrays() {
        let table = parse(
            r#"
            [op]
            name = "gemv_i4"          # 注释
            enabled = true

            [axes]
            GMAX = [32, 64]
            "#,
        )
        .unwrap();

        let op = table.table("op").unwrap();
        assert_eq!(op.string("name"), Some("gemv_i4"));
        assert_eq!(op.bool("enabled"), Some(true));
        assert_eq!(table.table("axes").unwrap().array("GMAX").unwrap().len(), 2);
    }

    #[test]
    fn parses_inline_tables_and_dotted_keys() {
        let table = parse(
            r#"
            [needs]
            mma = { m = "16..64", dt = ["f16", "bf16"], space = "FAST" }

            [[variants]]
            name = "pipe2"
            extra_needs.load_async = { bytes = "<=512" }
            "#,
        )
        .unwrap();

        let mma = table.table("needs").unwrap().table("mma").unwrap();
        assert_eq!(mma.string("m"), Some("16..64"));
        assert_eq!(mma.string_array("dt").unwrap(), vec!["f16", "bf16"]);

        let variants = table.table_array("variants").unwrap();
        assert_eq!(variants.len(), 1);
        let extra = variants[0].table("extra_needs").unwrap();
        assert!(extra.table("load_async").is_some());
    }

    #[test]
    fn keeps_hash_inside_strings() {
        let table = parse("[op]\nname = \"a#b\"\n").unwrap();
        assert_eq!(table.table("op").unwrap().string("name"), Some("a#b"));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let table = parse("\n# only a comment\n[meta]\nid = \"sm100\"\n").unwrap();
        assert_eq!(table.table("meta").unwrap().string("id"), Some("sm100"));
    }

    #[test]
    fn reports_line_numbers_on_bad_syntax() {
        let error = parse("[op]\nname \"x\"\n").unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.message.contains("`=`"), "{}", error.message);
    }

    #[test]
    fn rejects_unbalanced_brackets() {
        let error = parse("[op\nname = \"x\"\n").unwrap_err();
        assert_eq!(error.line, 1);
    }

    #[test]
    fn parses_floats() {
        let table = parse("[[variants]]\nname = \"pipe2\"\nmma_share = 0.82\n").unwrap();
        let variants = table.table_array("variants").unwrap();
        assert_eq!(variants[0].float("mma_share"), Some(0.82));
    }

    #[test]
    fn table_arrays_accumulate_in_order() {
        let table = parse(
            r#"
            [[variants]]
            name = "a"
            [[variants]]
            name = "b"
            "#,
        )
        .unwrap();
        let variants = table.table_array("variants").unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].string("name"), Some("a"));
        assert_eq!(variants[1].string("name"), Some("b"));
    }

    #[test]
    fn duplicate_keys_last_wins() {
        let table = parse("[op]\nname = \"a\"\nname = \"b\"\n").unwrap();
        assert_eq!(table.table("op").unwrap().string("name"), Some("b"));
        assert_eq!(table.table("op").unwrap().entries().len(), 1);
    }

    #[test]
    fn escaped_quotes_do_not_end_string_early() {
        let table = parse("[op]\nsummary = \"a\\\"b # c\"\n").unwrap();
        assert_eq!(
            table.table("op").unwrap().string("summary"),
            Some("a\"b # c")
        );
    }

    #[test]
    fn scalar_in_place_of_section_is_an_error() {
        let error = parse("[op]\nname = \"x\"\n[op.name]\n").unwrap_err();
        assert!(error.message.contains("已是标量"), "{}", error.message);
    }

    #[test]
    fn nested_section_paths() {
        let table = parse("[a.b]\nc = 1\n").unwrap();
        assert_eq!(
            table.table("a").unwrap().table("b").unwrap().int("c"),
            Some(1)
        );
    }
}
