//! 规划配置：反向排序与显存策略的纯数据约定。
//!
//! [`Config`] 是一次规划的**全部可调旋钮**：闭集枚举、可比较、可缓存，不持有缓冲、不认识
//! 设备。排序策略决定步骤之间的次序，显存策略决定在预算内偏好全留还是分段重放；两者互不
//! 耦合，分别进入 [`crate::Schedule`] 与 [`crate::Plan`]。

use crate::plan::MemoryPolicy;
use crate::schedule::Ordering;

/// 一次规划的配置：排序策略 + 显存策略。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Config {
    ordering: Ordering,
    policy: MemoryPolicy,
}

impl Config {
    /// 按两个策略构造配置；默认 [`Ordering::Depth`] + [`MemoryPolicy::Auto`]。
    pub const fn new(ordering: Ordering, policy: MemoryPolicy) -> Self {
        Self { ordering, policy }
    }

    /// 反向步骤的排序策略。
    pub const fn ordering(self) -> Ordering {
        self.ordering
    }

    /// 显存规划策略。
    pub const fn policy(self) -> MemoryPolicy {
        self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_depth_and_auto() {
        let config = Config::default();
        assert_eq!(config.ordering(), Ordering::Depth);
        assert_eq!(config.policy(), MemoryPolicy::Auto);
    }

    #[test]
    fn config_carries_both_strategies() {
        let config = Config::new(Ordering::Insertion, MemoryPolicy::Recompute);
        assert_eq!(config.ordering(), Ordering::Insertion);
        assert_eq!(config.policy(), MemoryPolicy::Recompute);
    }
}
