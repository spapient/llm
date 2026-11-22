//! 三维并行拓扑：[`Parallel`] 与 [`Topology`]。

use crate::collective::Group;
use diagnostic::Result;

/// 三种并行维度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parallel {
    /// 张量并行：同一层的权重按列切。
    Tensor,
    /// 流水并行：层按段切。
    Pipeline,
    /// 数据并行：同一份权重、不同的批。
    Data,
}

impl Parallel {
    /// 全部维度，供日志与 `doctor` 遍历。
    pub const ALL: [Self; 3] = [Self::Tensor, Self::Pipeline, Self::Data];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tensor => "tp",
            Self::Pipeline => "pp",
            Self::Data => "dp",
        }
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
    /// 三个维度都必须非零，且乘积（总进程数）必须能用 `u32` 表示。
    ///
    /// # Errors
    ///
    /// 任一维度为零，或 `tp · pp · dp` 溢出 `usize` / 超过 `u32::MAX` 时返回说明。
    pub fn new(tp: usize, pp: usize, dp: usize) -> Result<Self> {
        if [tp, pp, dp].contains(&0) {
            fail!(
                "shard::Topology::new",
                "并行度必须非零（tp = {tp}，pp = {pp}，dp = {dp}）"
            );
        }
        let world = tp
            .checked_mul(pp)
            .and_then(|plan| plan.checked_mul(dp))
            .filter(|world| *world <= u32::MAX as usize)
            .ok_or_else(|| {
                invalid!(
                    "shard::Topology::new",
                    "tp · pp · dp = {tp} · {pp} · {dp} 超出 u32：rank 布局无解，启动再多进程也对不上"
                )
            })?;
        Ok(Self { tp, pp, dp, world })
    }

    /// 单进程恒合法拓扑（1×1×1）：维度非零、乘积 1 在 `u32` 内，是 [`Self::new`] 的合法特例，
    /// 无需再包一层 `Result`。供进程内恒等编排直接取常量，避免在已知合法的输入上 `expect`。
    pub const fn single() -> Self {
        Self {
            tp: 1,
            pp: 1,
            dp: 1,
            world: 1,
        }
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

    /// 总进程数。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 校验启动的进程数与拓扑一致（不一致时继续跑只会得到静默的错误结果）。
    pub fn check_world(&self, world: usize) -> Result<()> {
        if world != self.world {
            fail!(
                "shard::Topology::check",
                "启动 {world} 个进程，但拓扑要求 {}",
                self.world
            );
        }
        Ok(())
    }

    /// 由三个下标算出全局 rank：`dp_idx · (tp · pp) + pp_idx · tp + tp_idx`。
    ///
    /// # Errors
    ///
    /// 任一下标超出它那一维的并行度。
    pub fn rank(&self, tp_idx: usize, pp_idx: usize, dp_idx: usize) -> Result<u32> {
        self.check_axis(Parallel::Tensor, tp_idx)?;
        self.check_axis(Parallel::Pipeline, pp_idx)?;
        self.check_axis(Parallel::Data, dp_idx)?;
        let flat = dp_idx * self.plan() + pp_idx * self.tp + tp_idx;
        // `new` 已把 world 限在 u32 以内，而 flat < world，故这次收窄不会失败。
        u32::try_from(flat).map_err(|_| invalid!("shard::Topology::rank", "rank {flat} 超出 u32"))
    }

    /// [`Topology::rank`] 的反解：`(tp_idx, pp_idx, dp_idx)`。
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
        let within = position % self.plan();
        Ok((within % self.tp, within / self.tp, position / self.plan()))
    }

    /// 第 `idx` 个指定维度的进程组：组内只沿该维移动，另外两维固定。
    ///
    /// # Errors
    ///
    /// `idx` 超出该维的组数（即 `world / 该维并行度`）。
    pub fn group(&self, parallel: Parallel, idx: usize) -> Result<Group> {
        let count = self.world / self.degree(parallel);
        if idx >= count {
            fail!(
                "shard::Topology::group",
                "{} 组下标 {idx} 超出 0..{count}",
                parallel.as_str()
            );
        }
        let ranks = (0..self.degree(parallel))
            .map(|step| {
                let [tp, pp, dp] = self.place(parallel, idx, step);
                self.rank(tp, pp, dp)
            })
            .collect::<Result<Vec<u32>>>()?;
        Group::new(ranks)
    }

    /// `tp · pp`：数据维之外的世界大小（也是 `Data` 组内的 rank 步长）。
    fn plan(&self) -> usize {
        self.tp * self.pp
    }

    /// 某一维的并行度，也是它每个组的大小。
    fn degree(&self, parallel: Parallel) -> usize {
        match parallel {
            Parallel::Tensor => self.tp,
            Parallel::Pipeline => self.pp,
            Parallel::Data => self.dp,
        }
    }

    /// 组的第 `step` 个成员在三个维上的下标：`idx` 是组号，`step` 是组内下标。
    ///
    /// 组号由**另外两维**按 rank 布局的次要→主要次序解出（先除小组、后除大组），本维只由
    /// `step` 决定——"这一组只沿本维移动"这条判据就在这里落地。
    fn place(&self, parallel: Parallel, idx: usize, step: usize) -> [usize; 3] {
        match parallel {
            Parallel::Tensor => [step, idx % self.pp, idx / self.pp],
            Parallel::Pipeline => [idx % self.tp, step, idx / self.tp],
            Parallel::Data => [idx % self.tp, idx / self.tp, step],
        }
    }

    /// 某一维的下标必须落在它自己的并行度以内。
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

#[cfg(test)]
#[path = "../tests/topology_tests.rs"]
mod tests;
