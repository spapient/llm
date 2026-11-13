//! 三维并行拓扑：[`Parallel`] 与 [`Topology`]。
//!
//! rank 的唯一布局规则写在这里：`rank = dp_index * (tp * pp) + pp_index * tp + tp_index`。
//! 有了它，"某个并行维的进程组"就是一串等差数列，[`crate::group::Group`] 的升序要求也自然满足。

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
        if tp == 0 || pp == 0 || dp == 0 {
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

    /// 第 `index` 个指定维度的进程组。
    ///
    /// - `Tensor`：`index` 是 `dp_index * pp + pp_index`，组大小 `tp`；
    /// - `Pipeline`：`index` 是 `dp_index`，组大小 `tp * pp`；
    /// - `Data`：`index` 是 `pp_index * tp + tp_index`，组大小 `dp`。
    pub fn group(&self, parallel: Parallel, index: usize) -> Result<Group> {
        // `plan` 与下面各式的乘积都以 `world` 为上界，而 `world` 在构造期已校验能用 u32 表示，
        // 所以这里不会回绕——这正是把上限放在 `new` 里的理由。
        let plan = self.tp * self.pp;
        let ranks: Vec<usize> = match parallel {
            Parallel::Tensor => {
                if index >= self.dp * self.pp {
                    return Err(self.out_of_range(parallel, index, self.dp * self.pp));
                }
                let base = index * self.tp;
                (0..self.tp).map(|step| base + step).collect()
            }
            Parallel::Pipeline => {
                if index >= self.dp {
                    return Err(self.out_of_range(parallel, index, self.dp));
                }
                let base = index * plan;
                (0..plan).map(|step| base + step).collect()
            }
            Parallel::Data => {
                if index >= plan {
                    return Err(self.out_of_range(parallel, index, plan));
                }
                (0..self.dp).map(|step| step * plan + index).collect()
            }
        };
        let mut converted = Vec::with_capacity(ranks.len());
        for rank in ranks {
            converted.push(u32::try_from(rank).map_err(|_| {
                Error::new(Code::Invalid, format!("rank {rank} 超出 u32"))
                    .with("shard::Topology::group")
            })?);
        }
        Group::new(converted)
    }

    fn out_of_range(&self, parallel: Parallel, index: usize, count: usize) -> Error {
        Error::new(
            Code::Invalid,
            format!("{} 组下标 {index} 超出 0..{count}", parallel.as_str()),
        )
        .with("shard::Topology::group")
    }
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
}
