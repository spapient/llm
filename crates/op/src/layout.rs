//! 布局：4 个构造子的闭集与声明式签名（cutile README §6）。
//!
//! 布局是**声明层的词汇表**：`[layouts]` 的取值就是这里的闭集，声明解析（[`crate::decl`]）
//! 要用它，因此事实源在本 crate，机制层（`cutile`）只消费。
//!
//! 布局不做推导，只给出一份稳定的声明式签名：缓存指纹与 `doctor` 都用它。声明式就是布局
//! 在构建期的身份——模板里 `{{w}}` 插的是它，而不是某个具体形状下的步长。
//!
//! 正规形（`(order, shape, stride, swizzle, pad)`）需要槽位的两级形状，而构建期没有形状：
//! tile 是轴取值（`GMAX=32`），不是 `w[n, m/2]` 这样的槽位形状。用编出来的形状去实例化，
//! 只能让缓存键和诊断看起来覆盖了形状，实际上描述了一次没有发生的编译。因此 P0 只保留闭集
//! 与声明式；等有真实形状（以及片段声明的布局）时再加回正规形与判定。
//!
//! 契约判定（模板给 vs 片段要）也不在 P0：片段层至今没有声明自己要求的布局（CPU 片段只收
//! 指针），没有第二个操作数可比较。设备侧唯一的真实约束是 bank 数，由
//! `cutile::space::check_alignment` 在构建期执行。

use core::fmt;
use diagnostic::{Code, Error};

/// 布局构造子闭集（cutile README §6.1）。`Swz`/`Pad` 只能包裹 `Row`/`Col`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutSpec {
    /// 行主序紧凑。
    Row,
    /// 列主序紧凑。
    Col,
    /// 在基布局上做 XOR swizzle（`b` 个 bank 位、`m` 个行位、`s` 个段位）。
    Swz {
        /// bank 位宽。
        b: u8,
        /// 行位宽。
        m: u8,
        /// 段位宽。
        s: u8,
    },
    /// 在最后一维插入填充元素。
    Pad(u32),
}

/// 布局写法不合法：闭集外的取值一律是 `invalid`。
fn invalid(detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, detail)
}

/// `swz<b,m,s>`：三个参数是位宽、块宽、步长，各一个 `u8`。
///
/// `rest` 是 `<` 之后的文本（含收尾的 `>`）。
fn swz(text: &str, rest: &str) -> diagnostic::Result<LayoutSpec> {
    let [b, m, s] = widths(text, closed(text, rest, '>')?)?;
    Ok(LayoutSpec::Swz { b, m, s })
}

/// 去掉收尾的分隔符；缺了就是声明写坏了。
fn closed<'a>(text: &str, rest: &'a str, end: char) -> diagnostic::Result<&'a str> {
    rest.strip_suffix(end)
        .ok_or_else(|| invalid(format!("`{text}` 缺少 `{end}`")))
}

/// 逗号分隔的三个位宽（顺序：位宽、块宽、步长）。
fn widths(text: &str, inner: &str) -> diagnostic::Result<[u8; 3]> {
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    let [b, m, s] = parts
        .try_into()
        .map_err(|_| invalid(format!("`{text}` 需要三个参数：swz<b,m,s>")))?;
    Ok([width(b)?, width(m)?, width(s)?])
}

/// 一个位宽：`u8` 十进制。
fn width(value: &str) -> diagnostic::Result<u8> {
    value
        .parse::<u8>()
        .map_err(|_| invalid(format!("`{value}` 不是合法的位宽")))
}

/// `pad(p)`：括号里是填充长度（允许首尾空白）。
fn pad(text: &str, rest: &str) -> diagnostic::Result<LayoutSpec> {
    let trimmed = closed(text, rest, ')')?.trim();
    trimmed
        .parse::<u32>()
        .map(LayoutSpec::Pad)
        .map_err(|_| invalid(format!("`{trimmed}` 不是合法的填充长度")))
}

impl LayoutSpec {
    /// 解析声明层写法：`row`、`col`、`swz<3,3,3>`、`pad(4)`。
    pub fn parse(text: &str) -> diagnostic::Result<Self> {
        let text = text.trim();
        match text {
            "row" => return Ok(Self::Row),
            "col" => return Ok(Self::Col),
            _ => {}
        }
        if let Some(rest) = text.strip_prefix("swz<") {
            return swz(text, rest);
        }
        if let Some(rest) = text.strip_prefix("pad(") {
            return pad(text, rest);
        }
        Err(invalid(format!(
            "`{text}` 不在布局闭集内（闭集：row | col | swz<b,m,s> | pad(p)）"
        )))
    }

    /// 渲染回声明层写法。
    ///
    /// 只有这一处写对照表：`Display` 直接写进调用方的缓冲，不在中间先建一个 `String`。
    #[must_use]
    pub fn as_str(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for LayoutSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Row => f.write_str("row"),
            Self::Col => f.write_str("col"),
            Self::Swz { b, m, s } => write!(f, "swz<{b},{m},{s}>"),
            Self::Pad(pad) => write!(f, "pad({pad})"),
        }
    }
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
