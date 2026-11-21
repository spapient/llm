//! 分布式训练编排：拓扑绑定 + 梯度同步后端，配置化挂到训练会话上。

use diagnostic::{Code, Error, Result};
use shard::{Collective, Group, Parallel, Reduce, Topology};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Distributed";

/// 分布式编排配置：拓扑、本进程 rank 与通信后端；梯度同步口径随配置走。
#[derive(Debug, Clone, PartialEq)]
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
        synchronize(collective, &group, self.reduce, self.average, slots)?;
        Ok(())
    }
}

/// 组内归约各槽梯度；配置开启平均时再按数据并行组数取均值。
fn synchronize<E: Collective>(
    collective: &mut E,
    group: &Group,
    reduce: Reduce,
    average: bool,
    slots: &mut [&mut [f32]],
) -> Result<()> {
    for grads in slots.iter_mut() {
        collective.all_reduce(group, reduce, grads)?;
    }
    if average {
        scale(slots, group)?;
    }
    Ok(())
}

/// 按组大小缩小平坦后的梯度；组大小为 0 是构造错误，不是可降级情形。
fn scale(slots: &mut [&mut [f32]], group: &Group) -> Result<()> {
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

#[cfg(test)]
#[path = "../tests/dist_tests.rs"]
mod tests;
