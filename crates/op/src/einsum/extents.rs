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
        let blame =
            |message: String| Error::new(Code::Invalid, message).with("einsum::Extents::group");
        if !is_label(label) {
            return Err(blame(format!("轴 `{label}` {NOT_LABEL}")));
        }
        if label == base {
            return Err(blame(format!("轴 `{label}` 不能以自己为分组基准")));
        }
        if divisor == 0 {
            return Err(blame(format!("轴 `{label}` 的分组因子不能为 0")));
        }
        let Some(length) = self.entries.get(&base).copied() else {
            return Err(blame(format!(
                "轴 `{label}` 以 `{base}` 为分组基准，但 `{base}` 没有长度（先声明基准轴）"
            )));
        };
        if !length.is_multiple_of(divisor) {
            return Err(blame(format!(
                "轴 `{base}` 的长度 {length} 不是分组因子 {divisor} 的整数倍"
            )));
        }
        if self.entries.insert(label, length / divisor).is_some() {
            return Err(blame(format!("轴 `{label}` 被声明了两次")));
        }
        self.groups.insert(label, Group { base, divisor });
        Ok(self)
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
mod tests {
    use super::*;

    #[test]
    fn lengths_are_looked_up_by_letter() {
        let extents = Extents::new([('n', 8), ('m', 64), ('g', 2)]).expect("合法");
        assert_eq!(extents.get('n'), Some(8));
        assert_eq!(extents.get('m'), Some(64));
        assert_eq!(extents.get('z'), None);
        assert_eq!(extents.len(), 3);
        assert!(!extents.is_empty());
    }

    #[test]
    fn the_display_is_stable_and_sorted() {
        let extents = Extents::new([('n', 8), ('g', 2), ('m', 64)]).expect("合法");
        assert_eq!(extents.to_string(), "g=2 m=64 n=8");
    }

    #[test]
    fn an_empty_table_is_allowed_because_a_scalar_has_no_axes() {
        let extents = Extents::new([]).expect("合法");
        assert!(extents.is_empty());
        assert_eq!(extents.to_string(), "");
    }

    #[test]
    fn a_zero_length_axis_is_rejected() {
        let error = Extents::new([('n', 0)]).expect_err("空轴应当被拒绝");
        assert!(error.to_string().contains("长度为 0"), "{error}");
    }

    #[test]
    fn a_duplicate_axis_is_rejected_instead_of_overwritten() {
        let error = Extents::new([('n', 8), ('n', 16)]).expect_err("重复轴应当被拒绝");
        assert!(error.to_string().contains("两次"), "{error}");
    }

    #[test]
    fn a_non_letter_axis_is_rejected() {
        let error = Extents::new([('1', 8)]).expect_err("非字母应当被拒绝");
        assert!(error.to_string().contains("不是字母"), "{error}");
    }

    #[test]
    fn a_grouped_axis_derives_its_length_and_remembers_its_base() {
        let extents = Extents::new([('m', 64)])
            .expect("合法")
            .group('g', 'm', 32)
            .expect("64 能被 32 整除");
        assert_eq!(extents.get('g'), Some(2));
        assert_eq!(extents.grouped('g'), Some(('m', 32)));
        assert_eq!(extents.grouped('m'), None);
        assert_eq!(extents.len(), 2);
        assert_eq!(extents.to_string(), "g=2 m=64");
    }

    #[test]
    fn grouping_rejects_a_base_that_cannot_be_divided_evenly() {
        let error = Extents::new([('m', 33)])
            .expect("合法")
            .group('g', 'm', 32)
            .expect_err("33 不能被 32 整除");
        assert!(error.to_string().contains("整数倍"), "{error}");
    }

    #[test]
    fn grouping_rejects_a_missing_base_and_a_zero_divisor() {
        let missing = Extents::new([('n', 8)])
            .expect("合法")
            .group('g', 'm', 32)
            .expect_err("没有 m");
        assert!(missing.to_string().contains("没有长度"), "{missing}");

        let zero = Extents::new([('m', 64)])
            .expect("合法")
            .group('g', 'm', 0)
            .expect_err("因子不能为 0");
        assert!(zero.to_string().contains("不能为 0"), "{zero}");
    }

    #[test]
    fn grouping_rejects_a_taken_label_and_a_self_base() {
        let taken = Extents::new([('m', 64), ('g', 3)])
            .expect("合法")
            .group('g', 'm', 32)
            .expect_err("g 已经存在");
        assert!(taken.to_string().contains("两次"), "{taken}");

        let itself = Extents::new([('m', 64)])
            .expect("合法")
            .group('m', 'm', 32)
            .expect_err("不能以自己为基准");
        assert!(itself.to_string().contains("自己"), "{itself}");
    }
}
