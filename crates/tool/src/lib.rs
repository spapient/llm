//! 呈现层工具：[`Report`] 与 [`join_names`]——把各层的诊断信息排成人类可读的文本。
//!
//! **拥有**：`Report`、`join_names`。**不拥有**：任何领域数据类型（它只接受字符串与列标签）、
//! 退出码（由调用方映射）。**依赖**：无。**被依赖**：`llm`（组装点的摘要输出）。
//!
//! 为什么值得单独一层：格式化是最容易在三个地方各写一遍的东西（`cutile` 的 CLI、`llm` 的摘要、
//! 将来的服务日志）。把它收成两个函数后，"同一份信息在两处排版不同"这类噪声就不存在了。
//!
//! # 例子
//!
//! ```
//! use llm_tool::Report;
//!
//! let report = Report::new("llm —— 组装点")
//!     .row("声明层", "1 个算子")
//!     .row("档案层", "4 份设备档案");
//! let text = report.render();
//! assert!(text.starts_with("llm —— 组装点\n"));
//! assert!(text.contains("  声明层  1 个算子"));
//! ```

#![deny(unsafe_code)]

use std::fmt::Write as _;

/// 一份带标题的键值报告：标题一行，之后每行一个"标签 + 值"。
///
/// 排版规则固定为标题 + 两空格缩进 + 标签补齐到同一列，因此调用方不需要关心宽度——
/// 也就不会出现两份输出对齐不一致的问题。
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

    /// 排成文本：标题一行，之后按"标签 + 值"两列输出，末尾带换行。
    pub fn render(&self) -> String {
        let width = self
            .rows
            .iter()
            .map(|(label, _)| label.chars().count())
            .max()
            .unwrap_or(0);
        let mut text = String::new();
        writeln!(text, "{}", self.title).expect("写入 String 不会失败");
        for (label, value) in &self.rows {
            writeln!(text, "  {label:<width$}  {value}").expect("写入 String 不会失败");
        }
        text
    }
}

/// 把名字并成一行，用顿号分隔；空列表返回 `—`（而不是空字符串，否则输出里会少一列）。
pub fn join_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let joined = names.collect::<Vec<_>>().join("、");
    if joined.is_empty() {
        "—".to_string()
    } else {
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_aligns_labels_to_the_widest_row() {
        let text = Report::new("标题")
            .row("a", "1")
            .row("longer", "2")
            .render();
        assert_eq!(text, "标题\n  a       1\n  longer  2\n");
    }

    #[test]
    fn an_empty_report_is_only_its_title() {
        let report = Report::new("标题");
        assert!(report.is_empty());
        assert_eq!(report.len(), 0);
        assert_eq!(report.render(), "标题\n");
    }

    #[test]
    fn names_are_joined_and_an_empty_list_shows_a_dash() {
        assert_eq!(join_names(["a", "b"].into_iter()), "a、b");
        assert_eq!(join_names([].into_iter()), "—");
    }
}
