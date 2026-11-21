//! 人可读排版：两列键值块、名字列表、字节量与单次耗时。
//!
//! **拥有**：文本排版的四个纯函数与它们的载体类型 —— [`Report`]（标题 + 标签/值两列）、
//! [`join_names`]（顿号连接，空列表写 `—`）、[`human_bytes`]（1024 进制，B / KiB / MiB / GiB）、
//! [`per_run`]（亚微秒用 ns，否则 ms）。
//! **不拥有**：任何判定与任何领域数据；这里只把已经算好的数排成一行，量纲口径由调用方给。
//! **依赖**：无（L1 叶，零依赖）。
//!
//! 单独成成员的理由是收敛依赖方向：排版同时被 L7 组合根（`cli`）与仓库工具（`tools`）使用，
//! 无论留在哪一边，另一边都会为了四个字符串函数而依赖它的全部依赖——历史形态是
//! `cli → tools` 一条运行时边，把静态分析器拖进了产品二进制。
//!
//! 同一组数在入口里只有一个单位：`cache gc` 的上限、工作集与基准台账的耗时都从这里取，
//! 不会出现同一次测量被说成两种单位。

#![deny(unsafe_code)]

/// 一份带标题的键值报告：标题一行，之后每行一个“标签 + 值”。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    title: String,
    rows: Vec<(String, String)>,
}

impl Report {
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

    /// 行数。
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// 是否只有标题。
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
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
mod tests;
