//! 解析后的爱因斯坦求和表达式：若干输入项 + 一个输出项。
//!
//! 文法与 `numpy.einsum` 的**显式**模式一致，只保留能穷举的那部分：
//!
//! ```text
//! expr   = term { "," term } "->" output
//! term   = { label }                # 空项表示标量缓冲
//! output = { label }
//! label  = "a".."z" | "A".."Z"
//! ```
//!
//! 不支持省略号（广播）、同项重复字母（对角提取）、省略 `->`（隐式模式）：它们都能用两条
//! 显式式子改写，而支持它们会让这份文档变成唯一事实源之外的第二个解释器。

use diagnostic::{Code, Error};
use std::fmt;
use std::str::FromStr;

/// 轴字母的闭集：`a..z` / `A..Z`；[`super::extents::Extents`] 也用这一条。
pub(crate) fn is_label(label: char) -> bool {
    label.is_ascii_alphabetic()
}

/// 不是合法轴字母时的公共措辞（调用点负责补上下文）。
pub(crate) const NOT_LABEL: &str = "不是字母（闭集是 `a..z` / `A..Z`）";

/// 解析后的爱因斯坦求和表达式：若干输入项 + 一个输出项。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    text: String,
    terms: Vec<Vec<char>>,
    output: Vec<char>,
}

impl Expr {
    /// 输出分隔符。
    pub const ARROW: &'static str = "->";

    /// 解析显式模式的表达式，例如 `"nm,m,ng,n->n"`。
    ///
    /// # Errors
    ///
    /// - 表达式为空、缺 `->` 或出现多个 `->`；
    /// - 某个字母不是 `a..z` / `A..Z`；
    /// - 同一项里出现重复字母（对角提取不在本层支持范围内）；
    /// - 输出项里有重复字母，或输出项里的字母没有出现在任何输入项里。
    pub fn parse(text: &str) -> Result<Self, Error> {
        let trimmed = text.trim();
        let (inputs, output) = sides(trimmed)?;
        let terms = inputs
            .split(',')
            .enumerate()
            .map(|(idx, term)| term_labels(term, idx))
            .collect::<Result<Vec<_>, _>>()?;
        let output = output_labels(output)?;
        check_output(&terms, &output, trimmed)?;

        Ok(Self {
            text: trimmed.to_string(),
            terms,
            output,
        })
    }

    /// 原文（去掉首尾空白），用于诊断。
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 输入项的个数，即操作数的个数。
    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    /// 逐个输入项的字母，顺序与操作数一致。
    pub fn terms(&self) -> impl Iterator<Item = &[char]> + '_ {
        self.terms.iter().map(Vec::as_slice)
    }

    /// 输出项的字母，顺序即输出的轴顺序。
    pub fn output(&self) -> &[char] {
        &self.output
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl FromStr for Expr {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

fn bad(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("einsum::Expr::parse")
}

/// `->` 的左右两侧：表达式非空，箭头恰好一个。
fn sides(trimmed: &str) -> Result<(&str, &str), Error> {
    if trimmed.is_empty() {
        return Err(bad("表达式为空"));
    }
    if trimmed.matches(Expr::ARROW).count() > 1 {
        return Err(bad(format!(
            "`->` 只能出现一次：`{trimmed}`（要拆成多个算子，请分别声明）"
        )));
    }
    trimmed.split_once(Expr::ARROW).ok_or_else(|| {
        bad(format!(
            "缺少 `->`：`{trimmed}`（显式模式必须写出输出轴，例如 `nm,m->n`）"
        ))
    })
}

fn term_labels(term: &str, idx: usize) -> Result<Vec<char>, Error> {
    let mut labels = Vec::with_capacity(term.len());
    for label in term.chars() {
        if !is_label(label) {
            return Err(bad(format!(
                "第 {idx} 项（`{term}`）：`{label}` {NOT_LABEL}"
            )));
        }
        if labels.contains(&label) {
            return Err(bad(format!(
                "第 {idx} 项（`{term}`）里字母 `{label}` 重复：对角提取请先 reshape，再写成两条式子"
            )));
        }
        labels.push(label);
    }
    Ok(labels)
}

fn output_labels(output: &str) -> Result<Vec<char>, Error> {
    let mut labels = Vec::with_capacity(output.len());
    for label in output.chars() {
        if !is_label(label) {
            return Err(bad(format!("输出项（`{output}`）：`{label}` {NOT_LABEL}")));
        }
        if labels.contains(&label) {
            return Err(bad(format!(
                "输出项（`{output}`）里字母 `{label}` 重复：输出轴必须唯一"
            )));
        }
        labels.push(label);
    }
    Ok(labels)
}

fn check_output(terms: &[Vec<char>], output: &[char], text: &str) -> Result<(), Error> {
    for label in output {
        if !terms.iter().any(|term| term.contains(label)) {
            return Err(bad(format!(
                "表达式 `{text}` 的输出轴 `{label}` 没有出现在任何输入项里：算不出来"
            )));
        }
    }
    Ok(())
}
