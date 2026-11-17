//! 仓库工具：配置驱动的静态质量门与人可读输出。
//!
//! **拥有**：词法分析（[`lex`]）、圈复杂度与函数长度（[`ccn`]）、文件行数（[`dup`]）、
//! 禁止调用形态（[`call`]）、文档漂移（[`docs`]）、测试代码剔除（[`strip`]）、质量门阈值（[`gate`]）、
//! 回退判定（[`regress`]）、统一基准记录（[`bench`]）、人可读排版（[`Report`]）。
//! **不拥有**：任何运行时领域数据类型、退出码（由调用方映射）。
//! **依赖**：`diagnostic`（错误契约与数据树）。
//!
//! 设计原则（参考 burn）：标准是数据（项目 `lints/vocabulary.toml` 的 `[gate]`），分析器是纯函数，
//! 结果可对拍；分析器本身不打印、不退出，门禁是否失败由调用方按 [`Gate`] 判定。
//!
//! 词汇表在**构建期**由 [`build.rs`](../build.rs) 用构建依赖 `toml` 解析 `lints/vocabulary.toml`，
//! 展开成 `diagnostic::data::{Table, Value}` 字面量写进 `$OUT_DIR/vocabulary.rs`，再由
//! [`vocabulary`] 暴露（ADR 0022）。运行期只遍历这棵树：不读配置文件，也没有任何 TOML 解析器。

#![deny(unsafe_code)]

pub mod bench;
pub mod call;
pub mod ccn;
pub mod docs;
pub mod dup;
pub mod gate;
pub mod lex;
pub mod regress;
pub mod strip;

/// 构建期生成的词汇表（`build.rs` 读 `lints/vocabulary.toml` 展开成 `Table` 字面量）。
#[allow(clippy::all, missing_docs)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/vocabulary.rs"));
}

pub use ccn::{Finding, analyze};
pub use gate::Gate;
pub use generated::vocabulary;
pub use lex::{Delim, Pos, Token, parse};
pub use strip::strip_tests;

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
