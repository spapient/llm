//! 分布式训练编排：拓扑绑定 + 梯度同步后端，配置化挂到训练会话上。
//!
//! 训练侧只认 [`Collective`] 契约：数据并行组内把各槽梯度按 [`Reduce::Sum`] 归约（可选除组数
//! 取平均），张量/流水并行的组划分由 [`Topology`] 派生。后端实现不同（进程内 [`Local`]、
//! 远端占位 [`Remote`]、后续 NCCL/Gloo），上层调用点不变。
//!
//! 一条泛型装配路径：[`Distributed::none`] 是单进程恒等回路（不持有通信句柄）；接真实后端时
//! 构造 [`Distributed::new`]，把拓扑、本 rank、通信后端一起登记进会话编排。

use diagnostic::{Code, Error, Result};
use shard::{Collective, Group, Parallel, Reduce, Topology};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Distributed";

/// 分布式编排配置：拓扑、本进程 rank 与通信后端；梯度同步口径随配置走。
///
/// `E` 是通信后端实现；单进程场景用 [`Distributed::none`]，不必套 `Option` 分支穿透会话。
pub struct Distributed<E: Collective> {
    topology: Topology,
    rank: u32,
    collective: Option<E>,
    /// 归约后是否除以数据并行组数（`true` = 平均梯度，`false` = 求和梯度）。
    average: bool,
    reduce: Reduce,
}

impl Distributed<Remote> {
    /// 单进程恒等编排：无拓扑、无后端，梯度不同步。后续装配多进程时用 [`Self::new`]。
    pub fn none() -> Self {
        Self {
            topology: Topology::single(),
            rank: 0,
            collective: None,
            average: true,
            reduce: Reduce::Sum,
        }
    }
}

impl<E: Collective> Distributed<E> {
    /// 由拓扑、本 rank 与通信后端构造；rank 必须落在拓扑世界内，否则归约组派生时会越界。
    ///
    /// # Errors
    ///
    /// `rank` 超出 `0..world`（[`Code::Invalid`]）。
    pub fn new(topology: Topology, rank: u32, collective: E) -> Result<Self> {
        if rank as usize >= topology.world() {
            return Err(Error::new(
                Code::Invalid,
                format!("rank {rank} 超出拓扑世界 0..{}", topology.world()),
            )
            .with(SITE));
        }
        Ok(Self {
            topology,
            rank,
            collective: Some(collective),
            average: true,
            reduce: Reduce::Sum,
        })
    }

    /// 归约后取平均（默认）；`false` 保留求和梯度。
    pub fn with_average(mut self, average: bool) -> Self {
        self.average = average;
        self
    }

    /// 归约种类；默认 [`Reduce::Sum`]（梯度同步的标准选择）。
    pub fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// 拓扑只读投影。
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// 本进程全局 rank。
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// 数据并行组：同一份权重、不同微批的进程集合；单进程退化成 `[0]`。
    pub fn data_group(&self) -> Result<Group> {
        let index = self.topology.indices(self.rank)?.2;
        self.topology.group(Parallel::Data, index)
    }

    /// 组大小（数据并行度）；单进程为 1。
    pub fn data_parallel(&self) -> usize {
        self.topology.dp()
    }

    /// 是否接了真实通信后端：`false` 时同步是恒等回路，不产生网络动作。
    pub fn is_distributed(&self) -> bool {
        self.collective.is_some()
    }

    /// 把全部槽位梯度在数据并行组内同步：先组内归约，再按组数取平均（配置开启时）。
    ///
    /// 缓冲长度与槽位对齐由调用方保证；后端返回的错误原样透传。单进程无后端时直接返回。
    ///
    /// # Errors
    ///
    /// 组派生越界、后端归约失败，或平均除数为 0 时返回对应 [`Code`]。
    pub fn sync_grads(&mut self, slots: &mut [&mut [f32]]) -> Result<()> {
        let group = self.data_group()?;
        let Some(collective) = self.collective.as_mut() else {
            return Ok(());
        };
        reduce_slots(collective, &group, self.reduce, slots)?;
        if self.average {
            average_slots(slots, &group)?;
        }
        Ok(())
    }
}

/// 组内对每个槽位梯度按 [`Reduce`] 归约；后端错误原样透传。
fn reduce_slots<E: Collective>(
    collective: &mut E,
    group: &Group,
    reduce: Reduce,
    slots: &mut [&mut [f32]],
) -> Result<()> {
    for grads in slots.iter_mut() {
        collective.all_reduce(group, reduce, grads)?;
    }
    Ok(())
}

/// 归约后按数据并行组数取平均；组大小为 0 是装配口径错误。
fn average_slots(slots: &mut [&mut [f32]], group: &Group) -> Result<()> {
    let divisor = group.size() as f32;
    if divisor == 0.0 {
        return Err(Error::new(Code::Internal, "数据并行组大小为 0").with(SITE));
    }
    for grads in slots.iter_mut() {
        for slot in grads.iter_mut() {
            *slot /= divisor;
        }
    }
    Ok(())
}

/// 远端占位后端别名：单进程 [`Distributed::none`] 用它占位，不挂真实通信栈。
pub use shard::{Local, Remote};
