//! 并行计划：由拓扑与本 rank 派生三轴下标、各维通信组与流水层段。
//!
//! [`Topology`] 只回答"进程在三维世界的哪里"；[`Plan`] 是装配侧查询视图，回答"这个 rank
//! 拿哪一片"——TP 份号、PP 层段、DP 副本号。纯只读数据，不持缓冲、不发起通信。层段与
//! [`crate::Tiling`] 同口径要求整除，摊余数在构造期拒绝，不保留不均匀切分。

use crate::collective::Group;
use crate::topology::{Parallel, Topology};
use diagnostic::Result;

/// 一个 rank 在三维拓扑里的并行归属：三轴下标 + 各维通信组 + 流水层段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    topology: Topology,
    tp_idx: usize,
    pp_idx: usize,
    dp_idx: usize,
}

impl Plan {
    /// 单进程计划：拓扑 1×1×1，三轴下标全 0，恒合法。
    pub fn single() -> Self {
        Self {
            topology: Topology::single(),
            tp_idx: 0,
            pp_idx: 0,
            dp_idx: 0,
        }
    }

    /// 由拓扑与全局 rank 派生；rank 越界复用 [`Topology::idxs`] 的同一条判据。
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

    /// 拓扑只读投影。
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// 张量并行下标：本 rank 在 TP 组内的份号。
    pub fn tp_idx(&self) -> usize {
        self.tp_idx
    }

    /// 流水并行下标：本 rank 负责的 PP 段号。
    pub fn pp_idx(&self) -> usize {
        self.pp_idx
    }

    /// 数据并行下标：本 rank 是第几个 DP 副本。
    pub fn dp_idx(&self) -> usize {
        self.dp_idx
    }

    /// 张量并行组：沿 TP 维移动，固定 PP 段与 DP 副本。
    pub fn tp_group(&self) -> Result<Group> {
        let idx = self.dp_idx * self.topology.pp() + self.pp_idx;
        self.topology.group(Parallel::Tensor, idx)
    }

    /// 流水并行组：沿 PP 段移动，固定 TP 份与 DP 副本。
    pub fn pp_group(&self) -> Result<Group> {
        let idx = self.dp_idx * self.topology.tp() + self.tp_idx;
        self.topology.group(Parallel::Pipeline, idx)
    }

    /// 数据并行组：沿 DP 副本移动，固定 TP 份与 PP 段。
    pub fn dp_group(&self) -> Result<Group> {
        let idx = self.pp_idx * self.topology.tp() + self.tp_idx;
        self.topology.group(Parallel::Data, idx)
    }

    /// 本 rank 在流水并行下负责的层段 `[start, end)`；`layers` 必须能被 PP 度数整除。
    ///
    /// 各段层数一致才是合法的流水划分；有余数说明层序切不开，交给装配侧拒绝，不在运行时
    /// 悄悄把余数塞给前几段。
    ///
    /// # Errors
    ///
    /// `layers` 为 0，或不能被 `pp` 整除。
    pub fn layer_range(&self, layers: usize) -> Result<(usize, usize)> {
        let pp = self.topology.pp();
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

#[cfg(test)]
#[path = "../tests/plan_tests.rs"]
mod tests;
