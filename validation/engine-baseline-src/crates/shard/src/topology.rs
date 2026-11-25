//! 并行编排：三维拓扑、rank 计划、部署策略与优先级队列。
//!
//! rank 的唯一布局规则是 `rank = dp_idx * (tp * pp) + pp_idx * tp + tp_idx`；
//! 正解、反解、进程组与层段都从这条规则派生。

use crate::collective::{Group, Reduce};
use diagnostic::Result;

const DIM: usize = 3;

/// 三种并行维度；判别式是它在 `[tp, pp, dp]` 中的位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Parallel {
    /// 张量并行：同一层的权重按列或按行切分。
    Tensor = 0,
    /// 流水并行：层按段切分。
    Pipeline = 1,
    /// 数据并行：同一权重对应不同微批。
    Data = 2,
}

impl Parallel {
    /// 全部维度，顺序与判别式一致。
    pub const ALL: [Self; DIM] = [Self::Tensor, Self::Pipeline, Self::Data];
    const NAMES: [&'static str; DIM] = ["tp", "pp", "dp"];

    /// 稳定短名。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        Self::NAMES[self as usize]
    }
}

/// 三维并行拓扑。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    tp: usize,
    pp: usize,
    dp: usize,
    world: usize,
}

impl Topology {
    /// 构造非零三维拓扑；总进程数必须能用 `u32` 表示。
    ///
    /// # Errors
    ///
    /// 任一维度为零，或 `tp · pp · dp` 溢出 `usize`／超过 `u32::MAX`。
    pub fn new(tp: usize, pp: usize, dp: usize) -> Result<Self> {
        if [tp, pp, dp].contains(&0) {
            fail!(
                "shard::Topology::new",
                "并行度必须非零（tp = {tp}，pp = {pp}，dp = {dp}）"
            );
        }
        let world = tp
            .checked_mul(pp)
            .and_then(|value| value.checked_mul(dp))
            .filter(|world| *world <= u32::MAX as usize)
            .ok_or_else(|| {
                invalid!(
                    "shard::Topology::new",
                    "tp · pp · dp = {tp} · {pp} · {dp} 超出 u32：rank 布局无解，启动再多进程也对不上"
                )
            })?;
        Ok(Self { tp, pp, dp, world })
    }

    /// 单进程恒等拓扑。
    #[must_use]
    pub const fn single() -> Self {
        Self {
            tp: 1,
            pp: 1,
            dp: 1,
            world: 1,
        }
    }

    accessors! {
        /// 张量并行度。
        tp -> usize = |s| s.tp;
        /// 流水并行度。
        pp -> usize = |s| s.pp;
        /// 数据并行度。
        dp -> usize = |s| s.dp;
        /// 总进程数。
        world -> usize = |s| s.world;
    }

    /// 校验启动进程数与拓扑一致。
    ///
    /// # Errors
    ///
    /// 实际进程数不等于 [`Self::world`]。
    pub fn check_world(&self, world: usize) -> Result<()> {
        if world != self.world {
            fail!(
                "shard::Topology::check_world",
                "启动 {world} 个进程，但拓扑要求 {}",
                self.world
            );
        }
        Ok(())
    }

    /// 将三个轴下标编码为全局 rank。
    ///
    /// # Errors
    ///
    /// 任一下标超出对应维度。
    pub fn rank(&self, tp_idx: usize, pp_idx: usize, dp_idx: usize) -> Result<u32> {
        for (axis, idx) in Parallel::ALL.into_iter().zip([tp_idx, pp_idx, dp_idx]) {
            self.check_axis(axis, idx)?;
        }
        let flat = self.weigh([tp_idx, pp_idx, dp_idx]);
        u32::try_from(flat).map_err(|_| invalid!("shard::Topology::rank", "rank {flat} 超出 u32"))
    }

    /// 将全局 rank 反解为 `(tp_idx, pp_idx, dp_idx)`。
    ///
    /// # Errors
    ///
    /// `rank` 不在 `0..world`。
    pub fn idxs(&self, rank: u32) -> Result<(usize, usize, usize)> {
        let position = rank as usize;
        if position >= self.world {
            fail!(
                "shard::Topology::idxs",
                "rank {rank} 超出 0..{}",
                self.world
            );
        }
        let [tp, pp, dp] = self.digits(position, None);
        Ok((tp, pp, dp))
    }

    /// 构造第 `idx` 个指定轴进程组；组内只沿该轴移动。
    ///
    /// # Errors
    ///
    /// `idx` 超出该轴组数 `world / degree`。
    pub fn group(&self, parallel: Parallel, idx: usize) -> Result<Group> {
        let own = parallel as usize;
        let degree = self.degree(parallel);
        let count = self.world / degree;
        if idx >= count {
            fail!(
                "shard::Topology::group",
                "{} 组下标 {idx} 超出 0..{count}",
                parallel.as_str()
            );
        }
        let stride = self.strides()[own];
        let head = self.weigh(self.digits(idx, Some(own)));
        let ranks = (0..degree)
            .map(|step| u32::try_from(head + step * stride).ok())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid!("shard::Topology::group", "组内 rank 超出 u32"))?;
        Group::new(ranks)
    }

    fn radices(&self) -> [usize; DIM] {
        [self.tp, self.pp, self.dp]
    }

    fn degree(&self, parallel: Parallel) -> usize {
        self.radices()[parallel as usize]
    }

    fn strides(&self) -> [usize; DIM] {
        let radices = self.radices();
        let mut strides = [1; DIM];
        for axis in 1..DIM {
            strides[axis] = strides[axis - 1] * radices[axis - 1];
        }
        strides
    }

    fn weigh(&self, idxs: [usize; DIM]) -> usize {
        idxs.iter().zip(self.strides()).map(|(n, s)| n * s).sum()
    }

    fn digits(&self, mut flat: usize, skip: Option<usize>) -> [usize; DIM] {
        let radices = self.radices();
        let mut digits = [0; DIM];
        for axis in 0..DIM {
            if Some(axis) == skip {
                continue;
            }
            digits[axis] = flat % radices[axis];
            flat /= radices[axis];
        }
        digits
    }

    fn check_axis(&self, parallel: Parallel, idx: usize) -> Result<()> {
        let limit = self.degree(parallel);
        if idx >= limit {
            fail!(
                "shard::Topology::rank",
                "{} 下标 {idx} 超出 0..{limit}",
                parallel.as_str()
            );
        }
        Ok(())
    }
}

/// 一个 rank 的三轴下标与各轴通信组视图。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    topology: Topology,
    tp_idx: usize,
    pp_idx: usize,
    dp_idx: usize,
}

impl Plan {
    /// 单进程恒等计划。
    #[must_use]
    pub const fn single() -> Self {
        Self {
            topology: Topology::single(),
            tp_idx: 0,
            pp_idx: 0,
            dp_idx: 0,
        }
    }

    /// 由拓扑与全局 rank 派生计划。
    ///
    /// # Errors
    ///
    /// `rank` 不在 `0..world`。
    pub fn new(topology: Topology, rank: u32) -> Result<Self> {
        let (tp_idx, pp_idx, dp_idx) = topology.idxs(rank)?;
        Ok(Self {
            topology,
            tp_idx,
            pp_idx,
            dp_idx,
        })
    }

    accessors! {
        /// 拓扑只读投影。
        topology -> Topology = |s| s.topology;
        /// 本 rank 的 TP 份号。
        tp_idx -> usize = |s| s.tp_idx;
        /// 本 rank 的 PP 段号。
        pp_idx -> usize = |s| s.pp_idx;
        /// 本 rank 的 DP 副本号。
        dp_idx -> usize = |s| s.dp_idx;
    }

    /// 本 rank 在任意指定轴上的通信组。
    ///
    /// # Errors
    ///
    /// 拓扑不变量被破坏、无法构造进程组时返回说明。
    pub fn group(&self, parallel: Parallel) -> Result<Group> {
        let idx = match parallel {
            Parallel::Tensor => self.dp_idx * self.topology.pp + self.pp_idx,
            Parallel::Pipeline => self.dp_idx * self.topology.tp + self.tp_idx,
            Parallel::Data => self.pp_idx * self.topology.tp + self.tp_idx,
        };
        self.topology.group(parallel, idx)
    }

    /// 张量并行组。
    ///
    /// # Errors
    ///
    /// 与 [`Self::group`] 相同。
    pub fn tp_group(&self) -> Result<Group> {
        self.group(Parallel::Tensor)
    }

    /// 流水并行组。
    ///
    /// # Errors
    ///
    /// 与 [`Self::group`] 相同。
    pub fn pp_group(&self) -> Result<Group> {
        self.group(Parallel::Pipeline)
    }

    /// 数据并行组。
    ///
    /// # Errors
    ///
    /// 与 [`Self::group`] 相同。
    pub fn dp_group(&self) -> Result<Group> {
        self.group(Parallel::Data)
    }

    /// 本 rank 负责的流水层段 `[start, end)`。
    ///
    /// # Errors
    ///
    /// `layers` 为零或不能被 PP 度数整除。
    pub fn layer_range(&self, layers: usize) -> Result<(usize, usize)> {
        let pp = self.topology.pp;
        if layers == 0 {
            fail!("shard::Plan::layer_range", "层数不能为 0");
        }
        if !layers.is_multiple_of(pp) {
            fail!(
                "shard::Plan::layer_range",
                "层数 {layers} 不能被流水并行度 {pp} 整除：各段层数会不一致",
            );
        }
        let width = layers / pp;
        let start = self.pp_idx * width;
        Ok((start, start + width))
    }
}

/// 集合通信调度优先级；序数越小越先提交。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(usize)]
pub enum Priority {
    /// 高优先级：梯度同步等阻塞主步的操作。
    High = 0,
    /// 普通优先级：默认组内通信。
    Normal = 1,
    /// 低优先级：后台与预取操作。
    Low = 2,
}

impl Priority {
    /// 全部优先级，顺序与调度序数一致。
    pub const ALL: [Self; 3] = [Self::High, Self::Normal, Self::Low];
    const NAMES: [&'static str; 3] = ["high", "normal", "low"];

    /// 稳定短名。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    /// 调度序数，直接由判别式派生，不维护第二份序号表。
    #[must_use]
    pub const fn ordinal(self) -> u8 {
        self as u8
    }

    /// 由稳定短名反解；未知名称返回 `None`。
    #[must_use]
    pub fn from_name(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|priority| priority.as_str() == value)
    }
}

/// 三维并行与集合通信策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    tp: usize,
    pp: usize,
    dp: usize,
    reduce: Reduce,
    average: bool,
    priority: Priority,
}

impl ParallelConfig {
    /// 单进程策略：1×1×1、求和、平均、普通优先级。
    #[must_use]
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

    /// 由三维度数构造，其余口径取单进程默认值。
    ///
    /// # Errors
    ///
    /// 度数非法，判据与 [`Topology::new`] 相同。
    pub fn new(tp: usize, pp: usize, dp: usize) -> Result<Self> {
        let config = Self {
            tp,
            pp,
            dp,
            ..Self::single()
        };
        config.topology()?;
        Ok(config)
    }

    /// 覆盖归约种类。
    #[must_use]
    pub const fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// 覆盖是否按 DP 组大小取平均。
    #[must_use]
    pub const fn with_average(mut self, average: bool) -> Self {
        self.average = average;
        self
    }

    /// 覆盖集合通信优先级。
    #[must_use]
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    accessors! {
        /// 张量并行度。
        tp -> usize = |s| s.tp;
        /// 流水并行度。
        pp -> usize = |s| s.pp;
        /// 数据并行度。
        dp -> usize = |s| s.dp;
        /// 归约种类。
        reduce -> Reduce = |s| s.reduce;
        /// 是否按 DP 组大小取平均。
        average -> bool = |s| s.average;
        /// 集合通信优先级。
        priority -> Priority = |s| s.priority;
    }

    /// 由度数派生拓扑。
    ///
    /// # Errors
    ///
    /// 与 [`Topology::new`] 相同。
    pub fn topology(&self) -> Result<Topology> {
        Topology::new(self.tp, self.pp, self.dp)
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::single()
    }
}

/// 一条带优先级的泛型待提交操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job<T> {
    priority: Priority,
    payload: T,
}

impl<T> Job<T> {
    /// 绑定操作与优先级。
    #[must_use]
    pub const fn new(priority: Priority, payload: T) -> Self {
        Self { priority, payload }
    }

    accessors! {
        /// 调度优先级。
        priority -> Priority = |s| s.priority;
    }

    /// 解出载荷。
    #[must_use]
    pub fn into_payload(self) -> T {
        self.payload
    }
}

/// 三档稳定 FIFO 队列；排空为 O(n)，不需要排序。
pub struct Queue<T> {
    lanes: [Vec<Job<T>>; 3],
    len: usize,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Queue<T> {
    /// 空队列。
    #[must_use]
    pub fn new() -> Self {
        Self {
            lanes: std::array::from_fn(|_| Vec::new()),
            len: 0,
        }
    }

    /// 按优先级入队，同档保持 FIFO。
    pub fn submit(&mut self, priority: Priority, payload: T) {
        self.lanes[priority as usize].push(Job::new(priority, payload));
        self.len += 1;
    }

    /// 待提交条数。
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// 是否没有待提交操作。
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 按高、普通、低优先级排空，同档保持入队先后。
    #[must_use]
    pub fn drain(self) -> Vec<Job<T>> {
        let mut jobs = Vec::with_capacity(self.len);
        for lane in self.lanes {
            jobs.extend(lane);
        }
        jobs
    }
}

#[cfg(test)]
#[path = "../tests/topology_tests.rs"]
mod tests;
