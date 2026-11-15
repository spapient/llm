//! 轴长度表：字母 → 元素个数，供一份表达式复用到任意形状。

use crate::einsum::expr::{NOT_LABEL, is_label};
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::fmt;

/// 轴长度表。
///
/// 长度**不属于表达式**：同一份 `"nm,m->n"` 可以算 8×64 也可以算 4096×4096，差别只在这张表。
/// 用 `BTreeMap` 而不是 `HashMap`：诊断输出（`Display`）必须稳定，否则测试与 `doctor` 的差异
/// 会随哈希种子变化。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Extents {
    entries: BTreeMap<char, usize>,
    groups: BTreeMap<char, Group>,
}

/// 一条分组轴的来历：它是 `base` 按 `divisor` 切出来的一面。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    base: char,
    divisor: usize,
}

/// 分组声明的静态判据：轴名合法、不以自己为基准、因子非零。
///
/// 这三条只读声明的字面量，不读表；「基准有没有长度、能不能整除」属于 [`Extents::base_length`]。
fn declared(label: char, base: char, divisor: usize) -> Result<(), Error> {
    if !is_label(label) {
        return Err(blame(format!("轴 `{label}` {NOT_LABEL}")));
    }
    if label == base {
        return Err(blame(format!("轴 `{label}` 不能以自己为分组基准")));
    }
    if divisor != 0 {
        return Ok(());
    }
    Err(blame(format!("轴 `{label}` 的分组因子不能为 0")))
}

/// 分组错误的统一边框：类别恒为 `invalid`，上下文固定到 [`Extents::group`]。
fn blame(message: String) -> Error {
    Error::new(Code::Invalid, message).with("einsum::Extents::group")
}

impl Extents {
    /// 由 `(字母, 长度)` 构造。
    ///
    /// # Errors
    ///
    /// - 字母不是 `a..z` / `A..Z`；
    /// - 长度为 `0`（空轴没有代价可言，通常是上游形状算错）；
    /// - 同一个字母给了两次（后者会静默覆盖前者，因此拒绝）。
    pub fn new(pairs: impl IntoIterator<Item = (char, usize)>) -> Result<Self, Error> {
        let mut extents = Self::default();
        for (label, length) in pairs {
            if !is_label(label) {
                return Err(
                    Error::new(Code::Invalid, format!("轴 `{label}` {NOT_LABEL}"))
                        .with("einsum::Extents::new"),
                );
            }
            if length == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("轴 `{label}` 的长度为 0：空轴没有可估算的代价"),
                )
                .with("einsum::Extents::new"));
            }
            if extents.entries.insert(label, length).is_some() {
                return Err(
                    Error::new(Code::Invalid, format!("轴 `{label}` 被声明了两次"))
                        .with("einsum::Extents::new"),
                );
            }
        }
        Ok(extents)
    }

    /// 取一个轴的长度。
    pub fn get(&self, label: char) -> Option<usize> {
        self.entries.get(&label).copied()
    }

    /// 声明一条**分组轴**：`label` 是 `base` 按 `divisor` 切出来的一面，长度由前者推出。
    ///
    /// 例如量化 GEMV 里 `scales[n, g]` 的 `g` 并不是独立的求和维，而是 `m` 每 `GROUP` 个元素
    /// 一组：`g` 只影响 `scales` 的**流量**（它确实只有 `n · m/GROUP` 个元素），不能参与运算
    /// 次数。用推导而不是手写长度，正是为了让这两件事不再被混为一谈。
    ///
    /// # Errors
    ///
    /// - `label` 不是字母、与 `base` 同名，或已经声明过；
    /// - `base` 没有长度；
    /// - `divisor` 为 `0`，或 `base` 的长度不是它的整数倍。
    pub fn group(mut self, label: char, base: char, divisor: usize) -> Result<Self, Error> {
        declared(label, base, divisor)?;
        let length = self.base_length(label, base, divisor)?;
        if self.entries.insert(label, length / divisor).is_some() {
            return Err(blame(format!("轴 `{label}` 被声明了两次")));
        }
        self.groups.insert(label, Group { base, divisor });
        Ok(self)
    }

    /// 基准轴的分组长度：基准必须先声明，且长度能被因子整除。
    fn base_length(&self, label: char, base: char, divisor: usize) -> Result<usize, Error> {
        let Some(length) = self.entries.get(&base).copied() else {
            return Err(blame(format!(
                "轴 `{label}` 以 `{base}` 为分组基准，但 `{base}` 没有长度（先声明基准轴）"
            )));
        };
        if length.is_multiple_of(divisor) {
            return Ok(length);
        }
        Err(blame(format!(
            "轴 `{base}` 的长度 {length} 不是分组因子 {divisor} 的整数倍"
        )))
    }

    /// 一条轴的分组来历；普通轴返回 `None`。
    pub fn grouped(&self, label: char) -> Option<(char, usize)> {
        self.groups
            .get(&label)
            .map(|group| (group.base, group.divisor))
    }

    /// 轴的数量（含分组轴）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否一个轴都没有（纯标量表达式）。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按字母序遍历所有轴（顺序稳定，供诊断使用）。
    pub fn labels(&self) -> impl Iterator<Item = char> + '_ {
        self.entries.keys().copied()
    }
}

impl fmt::Display for Extents {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (label, length)) in self.entries.iter().enumerate() {
            if index > 0 {
                formatter.write_str(" ")?;
            }
            write!(formatter, "{label}={length}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/extents_tests.rs"]
mod tests;
