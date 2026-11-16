//! 三维并行拓扑：[`Parallel`] 与 [`Topology`]。
//!
//! rank 的唯一布局规则写在这里：`rank = dp_index * (tp * pp) + pp_index * tp + tp_index`。
//!
//! 正反两个方向都从这一条规则派生：[`Topology::rank`] 由三个下标算 rank，
//! [`Topology::indices`] 由 rank 反解三个下标，[`Topology::group`] 则把「某一维的第 `index` 组」
//! 还原成三个下标再逐 rank 求值。于是"某个并行维的进程组"就是一串等差数列，
//! [`crate::group::Group`] 的升序要求也自然满足——不需要第二套 stride/span 公式。

use crate::group::Group;
use diagnostic::{Code, Error, Result};

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
///
/// `world` 在构造时算好并校验过：三个维度的乘积必须能用 `u32` 表示。rank 在 [`Group`]（与
/// `exec::Stream`）里是 `u32`，所以更大的拓扑本来就派不出一套合法的 rank 布局；让构造期报错，
/// 好过让 [`Topology::world`] 退回成饱和值（"一共 `usize::MAX` 个进程"）而 [`Topology::group`]
/// 用回绕的乘积发出一串看起来合法的 rank。
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
            return Err(Error::new(
                Code::Invalid,
                format!("并行度必须非零（tp = {tp}，pp = {pp}，dp = {dp}）"),
            )
            .with("shard::Topology::new"));
        }
        let world = tp
            .checked_mul(pp)
            .and_then(|plan| plan.checked_mul(dp))
            .filter(|world| *world <= u32::MAX as usize)
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!(
                        "tp · pp · dp = {tp} · {pp} · {dp} 超出 u32：rank 布局无解，启动再多进程也对不上"
                    ),
                )
                .with("shard::Topology::new")
            })?;
        Ok(Self { tp, pp, dp, world })
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
    ///
    /// 构造期已经校验过它能用 `u32` 表示，所以这里是精确值，不存在"饱和成 `usize::MAX`"那种
    /// 谎报——否则 [`Topology::check`] 也会跟着接受一个不可能的进程数。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 校验启动的进程数与拓扑一致（不一致时继续跑只会得到静默的错误结果）。
    pub fn check(&self, world: usize) -> Result<()> {
        if world != self.world() {
            return Err(Error::new(
                Code::Invalid,
                format!("启动 {world} 个进程，但拓扑要求 {}", self.world()),
            )
            .with("shard::Topology::check"));
        }
        Ok(())
    }

    /// 由三个下标算出全局 rank：`dp_index · (tp · pp) + pp_index · tp + tp_index`。
    ///
    /// # Errors
    ///
    /// 任一下标超出它那一维的并行度。
    pub fn rank(&self, tp_index: usize, pp_index: usize, dp_index: usize) -> Result<u32> {
        self.check_axis(Parallel::Tensor, tp_index)?;
        self.check_axis(Parallel::Pipeline, pp_index)?;
        self.check_axis(Parallel::Data, dp_index)?;
        rank_u32(dp_index * self.plan() + pp_index * self.tp + tp_index)
    }

    /// [`Topology::rank`] 的反解：`(tp_index, pp_index, dp_index)`。
    ///
    /// # Errors
    ///
    /// `rank` 不在 `0..world`。
    pub fn indices(&self, rank: u32) -> Result<(usize, usize, usize)> {
        let position = rank as usize;
        if position >= self.world() {
            return Err(Error::new(
                Code::Invalid,
                format!("rank {rank} 超出 0..{}", self.world()),
            )
            .with("shard::Topology::indices"));
        }
        let plan = self.plan();
        let within = position % plan;
        Ok((within % self.tp, within / self.tp, position / plan))
    }

    /// 第 `index` 个指定维度的进程组。
    ///
    /// - `Tensor`：`index` 是 `dp_index * pp + pp_index`，组大小 `tp`；
    /// - `Pipeline`：`index` 是 `dp_index`，组大小 `tp * pp`；
    /// - `Data`：`index` 是 `pp_index * tp + tp_index`，组大小 `dp`。
    pub fn group(&self, parallel: Parallel, index: usize) -> Result<Group> {
        Group::new(self.ranks(parallel, index)?)
    }

    /// `tp · pp`：数据维之外的世界大小（也是 `Data` 组内的 rank 步长）。
    fn plan(&self) -> usize {
        self.tp * self.pp
    }

    /// 某一维的组大小。
    fn size(&self, parallel: Parallel) -> usize {
        match parallel {
            Parallel::Tensor => self.tp,
            Parallel::Pipeline => self.plan(),
            Parallel::Data => self.dp,
        }
    }

    /// 某一维的组数：`world / 组大小`。
    fn groups(&self, parallel: Parallel) -> usize {
        self.world() / self.size(parallel)
    }

    /// 某一维第 `index` 组的 rank 列表（组内升序，已转成 `u32`）。
    fn ranks(&self, parallel: Parallel, index: usize) -> Result<Vec<u32>> {
        let count = self.groups(parallel);
        if index >= count {
            return Err(self.out_of_range(parallel, index, count));
        }
        (0..self.size(parallel))
            .map(|step| {
                let (tp, pp, dp) = self.place(parallel, index, step);
                self.rank(tp, pp, dp)
            })
            .collect()
    }

    /// 组的第 `step` 个成员在三个维上的下标：`index` 是组号，`step` 是组内下标。
    ///
    /// 三个维度共用一条布局规则（[`Topology::rank`]），这里只回答"这一步动的是哪一维"：
    /// `Tensor` 动 tp、`Pipeline` 动 pp、`Data` 动 dp，另两维由组号定死。
    fn place(&self, parallel: Parallel, index: usize, step: usize) -> (usize, usize, usize) {
        match parallel {
            Parallel::Tensor => (step, index % self.pp, index / self.pp),
            Parallel::Pipeline => (step % self.tp, step / self.tp, index),
            Parallel::Data => (index % self.tp, index / self.tp, step),
        }
    }

    /// 某一维的下标必须落在它自己的并行度以内。
    fn check_axis(&self, parallel: Parallel, index: usize) -> Result<()> {
        let limit = match parallel {
            Parallel::Tensor => self.tp,
            Parallel::Pipeline => self.pp,
            Parallel::Data => self.dp,
        };
        if index >= limit {
            return Err(Error::new(
                Code::Invalid,
                format!("{} 下标 {index} 超出 0..{limit}", parallel.as_str()),
            )
            .with("shard::Topology::rank"));
        }
        Ok(())
    }

    fn out_of_range(&self, parallel: Parallel, index: usize, count: usize) -> Error {
        Error::new(
            Code::Invalid,
            format!("{} 组下标 {index} 超出 0..{count}", parallel.as_str()),
        )
        .with("shard::Topology::group")
    }
}

/// rank 必须是 `u32`：`world` 在构造期已被限在 `u32` 以内，这里只是把类型收窄。
fn rank_u32(rank: usize) -> Result<u32> {
    u32::try_from(rank).map_err(|_| {
        Error::new(Code::Invalid, format!("rank {rank} 超出 u32")).with("shard::Topology::group")
    })
}

#[cfg(test)]
#[path = "../tests/topology_tests.rs"]
mod tests;
