//! 轴模型：角色分解、flops 口径与轴长度表（含派生分组轴）。

use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::fmt;

use super::expr::{NOT_LABEL, is_label};

/// 一条轴在这次计算里的角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 批轴：出现在输出里，且被至少两个**只读**操作数共享。
    Batch,
    /// 自由轴：出现在输出里，且只被一个操作数拥有。
    Free,
    /// 规约轴：不出现在输出里，求和后被消掉。
    Reduced,
    /// 分组轴：由另一条轴按固定因子切出（[`Extents::group`]）。
    Grouped,
}

/// 一条轴：字母、长度与角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    pub(crate) label: char,
    pub(crate) extent: usize,
    pub(crate) role: AxisRole,
}

impl Axis {
    /// 轴字母。
    pub fn label(&self) -> char {
        self.label
    }

    /// 轴长度（元素个数）。
    pub fn extent(&self) -> usize {
        self.extent
    }

    /// 轴角色。
    pub fn role(&self) -> AxisRole {
        self.role
    }
}

/// 表达式的轴分解：输出轴在前（按输出顺序），规约轴在后（按首次出现顺序）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axes {
    pub(crate) axes: Vec<Axis>,
}

impl Axes {
    /// 全部轴，顺序稳定（输出轴优先）。
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// 输出轴。
    pub fn output(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| matches!(axis.role, AxisRole::Batch | AxisRole::Free))
    }

    /// 规约轴。
    pub fn reduced(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Reduced)
    }

    /// 分组轴（由其他轴按因子切出，只计流量）。
    pub fn grouped(&self) -> impl Iterator<Item = &Axis> + '_ {
        self.axes
            .iter()
            .filter(|axis| axis.role == AxisRole::Grouped)
    }

    /// 是否有真正的规约轴（决定 `flops` 用哪种口径）。
    pub fn reduces(&self) -> bool {
        self.reduced().next().is_some()
    }

    /// 输出形状（按输出轴的声明顺序）。
    pub fn output_shape(&self) -> Vec<usize> {
        self.output().map(Axis::extent).collect()
    }

    /// 输出元素个数；标量输出为 `1`。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn output_elements(&self) -> Result<u128, Error> {
        self.output()
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 参与运算的轴的乘积，即"展开后的标量运算次数"的分母。
    ///
    /// 分组轴被排除：它是另一条轴的分组看法，不是独立的循环维。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn all_elements(&self) -> Result<u128, Error> {
        self.axes
            .iter()
            .filter(|axis| axis.role != AxisRole::Grouped)
            .try_fold(1u128, |total, axis| multiply(total, axis.extent))
    }

    /// 运算次数：有规约轴时按"每个规约元素一次乘加（2 flops）"，否则只算输出元素。
    ///
    /// # Errors
    ///
    /// 乘积超出 `u128` 时返回说明。
    pub fn flops(&self) -> Result<f64, Error> {
        if self.reduces() {
            return Ok(2.0 * self.all_elements()? as f64);
        }
        Ok(self.output_elements()? as f64)
    }
}

/// 轴长度乘积：溢出即形状不可能是真的。
pub(crate) fn multiply(total: u128, length: usize) -> Result<u128, Error> {
    total.checked_mul(length as u128).ok_or_else(overflow)
}

/// 维度乘积或流量溢出的统一错误帧。
pub(crate) fn overflow() -> Error {
    Error::new(Code::Invalid, "维度乘积或流量超出 u128：形状不可能是真的").with("einsum::Formula")
}

// ---------------------------------------------------------------------------
// 轴长度表：每条轴的长度，以及分组轴（由基准轴按固定因子切出）的来历。

/// 轴长度表。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Extents {
    entries: BTreeMap<char, Entry>,
}

/// 表中的一条轴：长度，以及（分组轴才有的）分组来历。长度与来历同住一格，声明分组轴时
/// 不再需要把「长度」和「这是什么分组」两处分别写入、再靠调用方保证它们同步。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
    /// 元素个数。
    len: usize,
    /// 分组轴的分组来历：`(基准轴, 因子)`；普通轴为 `None`。
    group: Option<(char, usize)>,
}

/// 分组声明的静态判据：轴名合法、不以自己为基准、因子非零。
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
            let entry = Entry {
                len: length,
                group: None,
            };
            if extents.entries.insert(label, entry).is_some() {
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
        self.entries.get(&label).map(|entry| entry.len)
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
        let entry = Entry {
            len: length / divisor,
            group: Some((base, divisor)),
        };
        if self.entries.insert(label, entry).is_some() {
            return Err(blame(format!("轴 `{label}` 被声明了两次")));
        }
        Ok(self)
    }

    /// 基准轴的分组长度：基准必须先声明，且长度能被因子整除。
    fn base_length(&self, label: char, base: char, divisor: usize) -> Result<usize, Error> {
        let Some(length) = self.get(base) else {
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
        self.entries.get(&label).and_then(|entry| entry.group)
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
        for (idx, (label, entry)) in self.entries.iter().enumerate() {
            if idx > 0 {
                formatter.write_str(" ")?;
            }
            write!(formatter, "{label}={}", entry.len)?;
        }
        Ok(())
    }
}
