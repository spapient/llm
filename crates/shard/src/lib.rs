//! 切分拓扑：[`Topology`]、[`Parallel`]、[`Partition`]。
//!
//! **拥有**：`Topology`、`Parallel`、`Partition`。**不拥有**：通信后端（`comm::Collective`）、
//! 参数切分本身（`weights` 给出条目，按 `Partition` 取片）。
//! **依赖**：`comm`、`diagnostic`。**被依赖**：`train`。
//!
//! rank 的唯一布局规则写在这里：`rank = dp_index * (tp * pp) + pp_index * tp + tp_index`。
//! 有了它，"某个并行维的进程组"就是一串等差数列，`comm::Group` 的升序要求也自然满足。
//!
//! # 例子
//!
//! ```
//! use llm_shard::{Parallel, Topology};
//!
//! let topology = Topology::new(2, 2, 2).expect("合法");
//! assert_eq!(topology.world(), 8);
//! let tensor = topology.group(Parallel::Tensor, 0).expect("第一个张量并行组");
//! assert_eq!(tensor.ranks(), [0, 1]);
//! let data = topology.group(Parallel::Data, 0).expect("第一个数据并行组");
//! assert_eq!(data.ranks(), [0, 4]);
//! ```

#![deny(unsafe_code)]

use comm::Group;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    tp: usize,
    pp: usize,
    dp: usize,
}

impl Topology {
    /// 三个维度都必须非零。
    pub fn new(tp: usize, pp: usize, dp: usize) -> Result<Self> {
        if tp == 0 || pp == 0 || dp == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("并行度必须非零（tp = {tp}，pp = {pp}，dp = {dp}）"),
            )
            .with("shard::Topology::new"));
        }
        Ok(Self { tp, pp, dp })
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
        self.tp.saturating_mul(self.pp).saturating_mul(self.dp)
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
        let plan = self.tp.saturating_mul(self.pp);
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

/// 单个维度上的均匀切分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Partition {
    axis: usize,
    parts: usize,
}

impl Partition {
    /// 由轴与份数构造；份数必须非零。
    pub fn new(axis: usize, parts: usize) -> Result<Self> {
        if parts == 0 {
            return Err(Error::new(Code::Invalid, "切分份数不能为 0").with("shard::Partition::new"));
        }
        Ok(Self { axis, parts })
    }

    /// 被切的轴。
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// 份数。
    pub fn parts(&self) -> usize {
        self.parts
    }

    /// 第 `part` 份的 `(起点, 长度)`；不整除直接报错（静默丢弃尾巴会让各进程看到的张量不一致）。
    pub fn slice(&self, dimension: usize, part: usize) -> Result<(usize, usize)> {
        if part >= self.parts {
            return Err(Error::new(
                Code::Invalid,
                format!("份下标 {part} 超出 0..{}", self.parts),
            )
            .with("shard::Partition::slice"));
        }
        if !dimension.is_multiple_of(self.parts) {
            return Err(Error::new(
                Code::Invalid,
                format!("长度 {dimension} 不能被 {} 整除", self.parts),
            )
            .with("shard::Partition::slice"));
        }
        let width = dimension / self.parts;
        Ok((part * width, width))
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

    #[test]
    fn a_partition_requires_an_even_split() {
        let partition = Partition::new(0, 4).expect("切分");
        assert_eq!(partition.slice(16, 0), Ok((0, 4)));
        assert_eq!(partition.slice(16, 3), Ok((12, 4)));
        assert_eq!(partition.slice(15, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(partition.slice(16, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(Partition::new(0, 0).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn axis_names_are_stable() {
        let names: Vec<&str> = Parallel::ALL.iter().map(|axis| axis.as_str()).collect();
        assert_eq!(names, ["tp", "pp", "dp"]);
    }
}
