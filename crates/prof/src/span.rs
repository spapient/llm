//! 职责：闭区间：exact/at_most 取值约束。

use crate::atom::*;

/// 一个闭区间（含两端）。`exact(16)` 表示"只能 16"，`at_most(512)` 表示"≤ 512"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    lo: u32,
    hi: u32,
}
impl Span {
    /// 两端构造；四个构造口径共用的私有入口。
    const fn new(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }
    /// 不限。
    pub const fn any() -> Self {
        Self::new(0, u32::MAX)
    }
    /// 精确值。
    pub const fn exact(value: u32) -> Self {
        Self::new(value, value)
    }
    /// 上界（下界 0）。
    pub const fn at_most(value: u32) -> Self {
        Self::new(0, value)
    }
    /// 下界（上界不限）。
    pub const fn at_least(value: u32) -> Self {
        Self::new(value, u32::MAX)
    }
    /// 区间；`lo > hi` 视为非法。
    pub fn range(lo: u32, hi: u32) -> diagnostic::Result<Self> {
        if lo > hi {
            return Err(Error::new(
                Code::Invalid,
                format!("区间下界 {lo} 大于上界 {hi}"),
            ));
        }
        Ok(Self { lo, hi })
    }
    /// 下界。
    pub const fn lo(self) -> u32 {
        self.lo
    }
    /// 上界。
    pub const fn hi(self) -> u32 {
        self.hi
    }
    /// 是否不限。
    pub const fn is_any(self) -> bool {
        self.lo == 0 && self.hi == u32::MAX
    }
    /// `self` 是否覆盖 `need`（区间包含）。
    pub const fn covers(self, need: Self) -> bool {
        self.lo <= need.lo && need.hi <= self.hi
    }
    /// 解析 `16`、`16..64`、`<=512`、`16..` 四种写法；越界或写法不认识时返回 `invalid`。
    pub fn parse(text: &str) -> diagnostic::Result<Self> {
        let text = text.trim();
        if let Some(rest) = text.strip_prefix("<=") {
            return parse_u32(rest).map(Self::at_most);
        }
        match text.split_once("..") {
            Some((lo, hi)) => Self::parse_range(lo, hi),
            None => parse_u32(text).map(Self::exact),
        }
    }
    /// 区间写法 `lo..hi` / `lo..`（空上界表示不限）。
    fn parse_range(lo: &str, hi: &str) -> diagnostic::Result<Self> {
        let lo = parse_u32(lo)?;
        if hi.trim().is_empty() {
            return Ok(Self::at_least(lo));
        }
        Self::range(lo, parse_u32(hi)?)
    }
}
fn parse_u32(text: &str) -> diagnostic::Result<u32> {
    text.trim().parse::<u32>().map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("`{}` 不是合法的非负整数", text.trim()),
        )
    })
}
impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_any() {
            return f.write_str("*");
        }
        if self.lo == self.hi {
            return write!(f, "{}", self.lo);
        }
        if self.lo == 0 {
            return write!(f, "<={}", self.hi);
        }
        if self.hi == u32::MAX {
            return write!(f, "{}..", self.lo);
        }
        write!(f, "{}..{}", self.lo, self.hi)
    }
}
