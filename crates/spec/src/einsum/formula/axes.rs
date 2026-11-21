use super::arith::multiply;
use super::*;

/// 一条轴在这次计算里的角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRole {
    /// 批轴：出现在输出里，且被至少两个**只读**操作数共享。
    Batch,
    /// 自由轴：出现在输出里，且只被一个操作数拥有。
    Free,
    /// 规约轴：不出现在输出里，求和后被消掉。
    Reduced,
    /// 分组轴：由另一条轴按固定因子切出（[`crate::einsum::Extents::group`]）。
    Grouped,
}

/// 一条轴：字母、长度与角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    pub(super) label: char,
    pub(super) extent: usize,
    pub(super) role: AxisRole,
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
    pub(super) axes: Vec<Axis>,
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
