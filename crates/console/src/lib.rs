//! 人可读输出：带标题的两列面板、Markdown 表、名字列表、字节量与单次耗时。
//!
//! **拥有**：把已经算好的数排成人读文本的全部原语 —— [`Panel`]（标题 + 标签/值两列）、
//! [`Table`]（Markdown 表头 / 对齐 / 行）、[`join_names`]（顿号连接，空列表写 `—`）、
//! [`human_bytes`]（1024 进制，B / KiB / MiB / GiB）、[`per_run`]（亚微秒用 ns，否则 ms）。
//! **不拥有**：任何判定与任何领域数据；这里只排版，量纲口径、该不该出现某一行都由调用方给。
//! **依赖**：无（L1 叶，零依赖）。
//!
//! 单独成成员的理由是收敛依赖方向：人读输出同时被 L7 组合根（`cli`）与工具成员（`musa`）使用，
//! 无论留在哪一边，另一边都会为了几个字符串函数而依赖它的全部依赖——历史形态是
//! `cli → tools` 一条运行时边，把静态分析器拖进了产品二进制。
//!
//! 同一组数在入口里只有一个单位：`cache gc` 的上限、工作集与基准台账的耗时都从这里取，
//! 不会出现同一次测量被说成两种单位；表头与分隔行的写法也只有 [`Table`] 这一份。

#![deny(unsafe_code)]

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
        let width = self
            .rows
            .iter()
            .map(|(label, _)| label.chars().count())
            .max()
            .unwrap_or(0);
        let mut text = format!("{}\n", self.title);
        for (label, value) in &self.rows {
            let pad = width.saturating_sub(label.chars().count());
            text.push_str(&format!("  {label}{}  {value}\n", " ".repeat(pad)));
        }
        text
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
pub struct Table {
    head: Vec<String>,
    align: Vec<Align>,
    rows: Vec<Vec<String>>,
}

impl Table {
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
