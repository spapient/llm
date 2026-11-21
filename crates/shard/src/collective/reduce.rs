//! 归约种类闭集：[`Reduce`] 与它的算术。

/// 归约种类闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reduce {
    /// 求和（梯度同步的默认选择）。
    Sum,
    /// 取最大（张量并行里的数值稳定选项）。
    Max,
    /// 取最小。
    Min,
    /// 取乘积。
    Prod,
}

impl Reduce {
    /// 全部种类，供日志与测试遍历。
    pub const ALL: [Self; 4] = [Self::Sum, Self::Max, Self::Min, Self::Prod];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
            Self::Prod => "prod",
        }
    }

    /// 单位元：与任何值 `apply` 之后都不改变它。
    pub fn identity(self) -> f32 {
        match self {
            Self::Sum => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
            Self::Prod => 1.0,
        }
    }

    /// 二元运算：满足结合律，且对 [`Reduce::identity`] 中立。
    pub fn apply(self, left: f32, right: f32) -> f32 {
        match self {
            Self::Sum => left + right,
            Self::Prod => left * right,
            Self::Max => extreme(left, right, true),
            Self::Min => extreme(left, right, false),
        }
    }

    /// 把一个非空切片折成一个值；空切片返回 `None`。
    pub fn fold(self, values: &[f32]) -> Option<f32> {
        let (first, rest) = values.split_first()?;
        Some(
            rest.iter()
                .fold(*first, |acc, value| self.apply(acc, *value)),
        )
    }
}

/// 极值归约：任一侧是 NaN 就传播 NaN，否则取 `max` / `min`。
fn extreme(left: f32, right: f32, upper: bool) -> f32 {
    if left.is_nan() || right.is_nan() {
        return f32::NAN;
    }
    if upper {
        left.max(right)
    } else {
        left.min(right)
    }
}

#[cfg(test)]
#[path = "../../tests/collective_reduce_tests.rs"]
mod tests;
