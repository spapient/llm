//! 并行策略配置：[`ParallelConfig`] 收齐三维并行度、归约口径、平均与通信优先级，是
//! [`crate::Distributed`](train 侧) 装配编排的唯一策略入口。
//!
//! 本模块只持有**部署形态的纯数据**：不读 toml、不决定配置来源、不发起通信；与内核工具链
//! 的 [`crate::RuntimeConfig`](crate::config::RuntimeConfig) 分属两条旋钮，互不混入。
//!
//! 三维度数与 [`Topology`](crate::Topology) 同一条整除/上界判据：构造 [`Topology`] 时才
//! 拒绝零维与 `u32` 溢出，这里只把数值原样收下。

use crate::collective::Reduce;
use crate::topology::Topology;
use diagnostic::Result;

/// 集合通信的调度优先级。
///
/// 判别式就是它在 [`Self::TABLE`] 里的行号（`#[repr(usize)]`）：短名与序数走同一张行表，
/// 不再各写一个三臂 `match`。序数越小越先调度；优先级只决定同一批次里集合操作的排队顺序，
/// 不改变任何归约数值。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum Priority {
    /// 高：梯度同步等阻塞主步的集合操作。
    High = 0,
    /// 普通：默认组内通信。
    Normal = 1,
    /// 低：后台/预取类通信，可让位于高优先级操作。
    Low = 2,
}

impl Priority {
    /// 全部优先级，供日志、配置解析与测试遍历。
    pub const ALL: [Self; 3] = [Self::High, Self::Normal, Self::Low];

    /// 一行：短名、调度序数。行序必须与判别式一致，由测试
    /// `priority_row_matches_discriminant` 钉住。
    const TABLE: [(&'static str, u8); 3] = [("high", 0), ("normal", 1), ("low", 2)];

    /// 全部优先级的稳定短名。
    pub fn as_str(self) -> &'static str {
        Self::TABLE[self as usize].0
    }

    /// 调度序数：越小越先排。
    pub fn ordinal(self) -> u8 {
        Self::TABLE[self as usize].1
    }

    /// 由短名反解；未知短名返回 `None`。
    pub fn from_name(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|priority| priority.as_str() == value)
    }
}

/// 一份并行策略：三维度数 + 归约口径 + 平均开关 + 通信优先级。
///
/// 度数是拓扑的输入；`reduce` / `average` / `priority` 是集合通信的口径。除度数外的字段
/// 都有合理默认，装配方只需覆盖要改的项。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    /// 张量并行度（同一层权重按列/行切给多卡）。
    tp: usize,
    /// 流水并行度（层按段切给多卡）。
    pp: usize,
    /// 数据并行度（同一份权重、不同微批）。
    dp: usize,
    /// 组内归约种类：梯度同步默认求和。
    reduce: Reduce,
    /// 归约后是否按数据并行组数取平均。
    average: bool,
    /// 集合通信调度优先级。
    priority: Priority,
}

impl ParallelConfig {
    /// 单进程策略：1×1×1、求和、平均、普通优先级。恒合法，无需包 `Result`。
    pub const fn single() -> Self {
        Self {
            tp: 1,
            pp: 1,
            dp: 1,
            reduce: Reduce::Sum,
            average: true,
            priority: Priority::Normal,
        }
    }

    /// 由三维度数构造；其余口径取默认（求和、平均、普通优先级）。
    ///
    /// # Errors
    ///
    /// 度数为零或 `tp · pp · dp` 超出 `u32` 时，交由 [`Topology::new`] 的同一判据拒绝。
    pub fn new(tp: usize, pp: usize, dp: usize) -> Result<Self> {
        let config = Self {
            tp,
            pp,
            dp,
            reduce: Reduce::Sum,
            average: true,
            priority: Priority::Normal,
        };
        config.topology()?;
        Ok(config)
    }

    /// 覆盖归约种类。
    pub fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// 覆盖平均开关：`false` 保留求和梯度。
    pub fn with_average(mut self, average: bool) -> Self {
        self.average = average;
        self
    }

    /// 覆盖通信优先级。
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// 张量并行度。
    pub fn tp(&self) -> usize {
        self.tp
    }

    /// 流水并行度。
    pub fn pp(&self) -> usize {
        self.pp
    }

    /// 数据并行度。
    pub fn dp(&self) -> usize {
        self.dp
    }

    /// 归约种类。
    pub fn reduce(&self) -> Reduce {
        self.reduce
    }

    /// 是否按组数取平均。
    pub fn average(&self) -> bool {
        self.average
    }

    /// 通信优先级。
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// 由度数派生三维拓扑；零维与上界溢出在此拒绝，口径与 [`Topology::new`] 一致。
    pub fn topology(&self) -> Result<Topology> {
        Topology::new(self.tp, self.pp, self.dp)
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::single()
    }
}

#[cfg(test)]
#[path = "../tests/policy_tests.rs"]
mod tests;
