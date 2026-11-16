//! 归约种类闭集：[`Reduce`] 与它的算术。
//!
//! 归约的算术只有一份：[`Reduce::identity`]（单位元）与 [`Reduce::apply`]（二元运算）在这里定义，
//! `fold` 与集合通信实现都从这两个派生。分开写会漂移——把 `Max` 的单位元写成 `f32::MIN`、
//! 或在 `fold` 里对空输入返回单位元，都是只在全负输入或空缓冲上才露头的错。
//!
//! NaN 的口径写死为**传播**：`Max` / `Min` 用 `f32::max` / `f32::min` 会静默吃掉 NaN，
//! 而"某一路算出 NaN，另一路把它抹掉"在梯度同步里是最难查的一类静默错误。

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
    ///
    /// `Max` / `Min` 的单位元是无穷而不是 `f32::MAX` / `f32::MIN`：后者会把"所有值都比它大"
    /// 这一正常情形误判成一个具体的有限数。
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
    ///
    /// 空输入不给单位元：`Max` 的空归约在数学上没有答案，返回 `-inf` 会让调用方以为
    /// "这一组确实取到了负无穷"，而不是"没有数据"。
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
#[path = "../tests/reduce_tests.rs"]
mod tests;
