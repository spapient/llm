//! 表达式解析：文本 → [`Expr`]，只做文法与闭集校验，不做任何形状推断。

use diagnostic::{Code, Error};
use std::fmt;
use std::str::FromStr;

/// 解析后的爱因斯坦求和表达式：若干输入项 + 一个输出项。
///
/// 只保存**结构**（项与字母），不保存任何长度；长度由调用方给出的
/// [`crate::einsum::Extents`] 决定，因此同一份表达式可以复用到任意形状。
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
            .map(|(index, term)| term_labels(term, index))
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

/// 轴字母的闭集：`a..z` / `A..Z`；[`crate::einsum::Extents`] 也用这一条。
///
/// 这里只回答"是不是"，不产出错误：这一层不产生 `String` 错误，文本由各调用点按自己的上下文给
/// （`Expr` 说"第几项"，`Extents` 说"哪个轴"）。
pub(crate) fn is_label(label: char) -> bool {
    label.is_ascii_alphabetic()
}

/// 不是合法轴字母时的公共措辞（调用点负责补上下文）。
pub(crate) const NOT_LABEL: &str = "不是字母（闭集是 `a..z` / `A..Z`）";

fn bad(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("einsum::Expr::parse")
}

/// `->` 的左右两侧：表达式非空，箭头恰好一个。
///
/// 只回答"文本切不切得开"，不回答轴字母是否合法——那是 [`term_labels`]/[`output_labels`] 的事。
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

fn term_labels(term: &str, index: usize) -> Result<Vec<char>, Error> {
    let mut labels = Vec::with_capacity(term.len());
    for label in term.chars() {
        if !is_label(label) {
            return Err(bad(format!(
                "第 {index} 项（`{term}`）：`{label}` {NOT_LABEL}"
            )));
        }
        if labels.contains(&label) {
            return Err(bad(format!(
                "第 {index} 项（`{term}`）里字母 `{label}` 重复：对角提取请先 reshape，再写成两条式子"
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

#[cfg(test)]
#[path = "expr_tests.rs"]
mod tests;
