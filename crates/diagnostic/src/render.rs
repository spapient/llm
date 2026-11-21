//! 人读渲染：带标题的两列面板、Markdown 表、顿号列表、字节量与单次耗时。
//!
//! **拥有**：把已经算好的数排成人读文本的原语 —— [`Panel`]（标题 + 标签/值两列）、
//! [`Sheet`]（Markdown 表：表头 / 分隔行 / 逐列对齐）、[`join_names`]（顿号连接，空列表写 `—`）、
//! [`human_bytes`]（1024 进制，B / KiB / MiB / GiB）、[`per_run`]（亚微秒用 ns，否则 ms）。
//! **不拥有**：任何判定与任何领域数据；这里只排版，量纲口径、该不该出现某一行都由调用方给。
//!
//! 与 [`crate::error`] 共处一个 crate 的理由同样是依赖方向：错误文本与报告文本都是“给人看的
//! 一段字”，都必须比任何数据层更低；消费者（L7 组合根 `cli`）本来就依赖本 crate，
//! 因此这里不再单独成成员，也不再出现第二个只装几个字符串函数的叶。
//!
//! 同一组数在入口里只有一个单位：`cache gc` 的上限、工作集与基准台账的耗时都从这里取，
//! 不会出现同一次测量被说成两种单位；表头与分隔行的写法也只有 [`Sheet`] 这一份。

/// 一份带标题的面板：标题一行，之后每行一个“标签 + 值”，标签按最宽者对齐。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Panel {
    title: String,
    rows: Vec<(String, String)>,
}

impl Panel {
    /// 由标题构造。
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            rows: Vec::new(),
        }
    }

    /// 追加一行（链式）。
    #[must_use]
    pub fn row(mut self, label: impl Into<String>, value: impl Into<String>) -> Self {
        self.rows.push((label.into(), value.into()));
        self
    }

    /// 排成文本：标题一行，之后按“标签 + 值”两列输出，末尾带换行。
    pub fn render(&self) -> String {
        let widest = self
            .rows
            .iter()
            .map(|(label, _)| span(label))
            .max()
            .unwrap_or(0);
        let mut text = format!("{}\n", self.title);
        for (label, value) in &self.rows {
            let pad = widest.saturating_sub(span(label));
            text.push_str(&format!("  {label}{}  {value}\n", " ".repeat(pad)));
        }
        text
    }
}

/// 一段文本在等宽终端里占的列数。
fn span(text: &str) -> usize {
    text.chars().map(char_span).sum()
}

/// 单字符占的列数：宽/全角两列，组合符与零宽不占列，其余一列。
///
/// 按字节数或 `chars().count()` 量宽度都会把「条目数」量成 3 列（实际 6 列），中英混排的标签
/// 于是逐行错开——而这些面板的读者是人，错开的第一眼就是坏掉的一眼。表只覆盖需要的那几段：
/// 本工作区的标签是中文加拉丁，再加少量数学符号（`×`/`≤` 属歧义宽度，终端按一列排）。
fn char_span(ch: char) -> usize {
    match ch as u32 {
        0x0300..=0x036F | 0x200B..=0x200F | 0xFEFF => 0,
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xA000..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE10..=0xFE19
        | 0xFE30..=0xFE6F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1F300..=0x1F64F
        | 0x1F900..=0x1F9FF
        | 0x20000..=0x3FFFD => 2,
        _ => 1,
    }
}

/// 单元格的对齐方向：Markdown 分隔行里 `---` 与 `---:` 的唯一来源。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Align {
    /// 左对齐（也是缺省）。
    #[default]
    Left,
    /// 右对齐（数值列）。
    Right,
}

/// 一份 Markdown 表：表头、逐列对齐与数据行。
///
/// 表头与分隔行只在这里拼：调用方给数据，不给分隔行的字面量——否则同一个
/// `|---|:---|` 会在每个报告函数里各写一遍，列数一变就有地方漏改。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sheet {
    head: Vec<String>,
    align: Vec<Align>,
    rows: Vec<Vec<String>>,
}

impl Sheet {
    /// 由表头构造；列宽与列数由表头定，分隔行随之生成。
    pub fn new(head: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            head: head.into_iter().map(Into::into).collect(),
            align: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// 逐列对齐（链式）；少给的列按左对齐补齐。
    #[must_use]
    pub fn align(mut self, align: impl IntoIterator<Item = Align>) -> Self {
        self.align = align.into_iter().collect();
        self
    }

    /// 追加一行（链式）；缺的格子补空，多的格子按表头列数截断。
    #[must_use]
    pub fn row(mut self, cells: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut row: Vec<String> = cells.into_iter().map(Into::into).collect();
        row.resize(self.head.len(), String::new());
        row.truncate(self.head.len());
        self.rows.push(row);
        self
    }

    /// 排成 Markdown：表头一行、分隔一行、之后每行一条，末尾带换行。
    pub fn render(&self) -> String {
        let mut text = format!("| {} |\n|", self.head.join(" | "));
        for column in 0..self.head.len() {
            let mark = match self.align.get(column).copied().unwrap_or_default() {
                Align::Left => "---|",
                Align::Right => "---:|",
            };
            text.push_str(mark);
        }
        text.push('\n');
        for row in &self.rows {
            text.push_str(&format!("| {} |\n", row.join(" | ")));
        }
        text
    }
}

/// 把名字并成一行，用顿号分隔；空列表返回 `—`。
pub fn join_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let joined = names.collect::<Vec<_>>().join("、");
    if joined.is_empty() {
        "—".to_string()
    } else {
        joined
    }
}

/// 人读的字节数（工作集与容量对照用）：1024 进制，B / KiB / MiB / GiB。
///
/// 排版口径只此一份：入口与基准台账都从这里取，避免同一组字节被说成两种单位。
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// 单次耗时：亚微秒内核用 ns，否则用 ms（避免一串 `0.000`）。
pub fn per_run(ns: f64) -> String {
    if ns < 1e3 {
        format!("{ns:.1} ns/次")
    } else {
        format!("{:.3} ms/次", ns / 1e6)
    }
}

#[cfg(test)]
#[path = "../tests/render_tests.rs"]
mod tests;
