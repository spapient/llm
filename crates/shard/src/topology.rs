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
mod tests {
    use super::*;

    #[test]
    fn the_world_is_the_product_of_the_three_axes() {
        let topology = Topology::new(4, 2, 3).expect("合法");
        assert_eq!(topology.world(), 24);
        assert!(topology.check(24).is_ok());
        assert_eq!(topology.check(12).unwrap_err().code(), Code::Invalid);
        assert_eq!(Topology::new(0, 1, 1).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn each_axis_has_its_own_group_stride() {
        let topology = Topology::new(2, 2, 2).expect("合法");
        let tensor: Vec<Vec<u32>> = (0..4)
            .map(|index| {
                topology
                    .group(Parallel::Tensor, index)
                    .expect("张量组")
                    .ranks()
                    .to_vec()
            })
            .collect();
        assert_eq!(tensor, [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
        assert_eq!(
            topology
                .group(Parallel::Pipeline, 1)
                .expect("流水组")
                .ranks(),
            [4, 5, 6, 7]
        );
        assert_eq!(
            topology.group(Parallel::Data, 1).expect("数据组").ranks(),
            [1, 5]
        );
    }

    #[test]
    fn an_index_outside_the_axis_is_rejected() {
        let topology = Topology::new(2, 2, 2).expect("合法");
        assert_eq!(
            topology.group(Parallel::Tensor, 4).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            topology.group(Parallel::Data, 4).unwrap_err().code(),
            Code::Invalid
        );
    }

    /// 乘积溢出时不能只把 `world()` 饱和掉：那样 `check` 会跟着接受一个不可能的进程数，
    /// 而 `group` 又用回绕的乘积发出一串"看起来合法"的 rank。
    #[test]
    fn a_topology_whose_world_does_not_fit_in_u32_is_refused() {
        let error = Topology::new(usize::MAX, 2, 2).expect_err("溢出");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("u32"), "{error}");
        // 恰好超出 u32 也算无解：rank 是 u32，最后一个 rank 写不出来。
        assert_eq!(
            Topology::new(1 << 16, 1 << 16, 1).unwrap_err().code(),
            Code::Invalid
        );
        // 卡在 u32::MAX 之下的乘积仍然合法，且 world 是精确值而不是饱和值。
        let huge = Topology::new(65_535, 65_535, 1).expect("乘积 4_294_836_225 < u32::MAX");
        assert_eq!(huge.world(), 4_294_836_225);
        assert!(huge.check(4_294_836_225).is_ok());
        assert_eq!(huge.check(usize::MAX).unwrap_err().code(), Code::Invalid);
    }

    /// 三个维度的组必须各自把 `0..world` 恰切一遍：不多不少、不重叠。
    /// 布局规则一旦算错，这里会先于训练发现。
    #[test]
    fn every_axis_partitions_the_ranks_exactly_once() {
        for (tp, pp, dp) in [(1, 1, 1), (2, 2, 2), (3, 5, 7), (4, 1, 3), (1, 6, 2)] {
            let topology = Topology::new(tp, pp, dp).expect("合法");
            let world = topology.world();
            for parallel in Parallel::ALL {
                let count = match parallel {
                    Parallel::Tensor => dp * pp,
                    Parallel::Pipeline => dp,
                    Parallel::Data => tp * pp,
                };
                let mut seen = vec![0usize; world];
                for index in 0..count {
                    for &rank in topology.group(parallel, index).expect("组存在").ranks() {
                        seen[rank as usize] += 1;
                    }
                }
                assert!(
                    seen.iter().all(|count| *count == 1),
                    "{parallel:?} 在 {tp}·{pp}·{dp} 上没有恰好划开：{seen:?}"
                );
                assert_eq!(
                    topology.group(parallel, count).unwrap_err().code(),
                    Code::Invalid,
                    "{parallel:?} 的第 {count} 组不存在"
                );
            }
        }
    }

    #[test]
    fn axis_names_are_stable() {
        let names: Vec<&str> = Parallel::ALL.iter().map(|axis| axis.as_str()).collect();
        assert_eq!(names, ["tp", "pp", "dp"]);
    }

    /// 布局规则只有一个方向是可读的公式，反解必须与它逐点互逆——否则 `group` 与
    /// "我在第几段"会各说一套。
    #[test]
    fn rank_and_indices_are_inverses() {
        for (tp, pp, dp) in [(1, 1, 1), (2, 2, 2), (3, 1, 4), (1, 5, 2)] {
            let topology = Topology::new(tp, pp, dp).expect("合法");
            for rank in 0..topology.world() {
                let (tp_index, pp_index, dp_index) = topology
                    .indices(u32::try_from(rank).expect("world 收窄"))
                    .expect("反解");
                assert!(tp_index < tp && pp_index < pp && dp_index < dp);
                assert_eq!(
                    topology.rank(tp_index, pp_index, dp_index),
                    Ok(u32::try_from(rank).expect("world 收窄")),
                    "{tp}·{pp}·{dp} 的 rank {rank} 反解后不再等于自己"
                );
            }
            assert_eq!(
                topology
                    .indices(u32::try_from(topology.world()).expect("world 收窄"))
                    .unwrap_err()
                    .code(),
                Code::Invalid
            );
        }
    }

    #[test]
    fn an_index_outside_its_own_degree_is_refused() {
        let topology = Topology::new(2, 3, 4).expect("合法");
        assert_eq!(topology.rank(2, 0, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(topology.rank(0, 3, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(topology.rank(0, 0, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(topology.rank(1, 2, 3), Ok(23));
    }
}
