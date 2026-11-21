//! 张量并行的权重切分：[`Sharding`] 与 [`Tiling`]。
//!
//! 张量并行只有两种线性层切法，差别在于"切哪一维"以及"合起来需不需要通信"：
//!
//! | 切法 | 切的维 | 每个 rank 的权重 | 合成整层输出 |
//! |---|---|---|---|
//! | [`Sharding::Column`] | 输出维（轴 0） | `[out / tp, in]` | 不需要：下一层本来就吃分片输入 |
//! | [`Sharding::Row`] | 输入维（轴 1） | `[out, in / tp]` | `all_reduce(Sum)`：每个 rank 是一份部分和 |
//!
//! 这条规则写成表，而不是散在每个模型的 `linear` 调用里：同一份权重换一种切法只改变体，
//! 而"这一层要不要归约"由 [`Tiling::merge`] 回答，调用方不需要自己记。
//!
//! 切分必须整除：TP 各 rank 的张量形状必须一致，所以不整除在这里就报错，而不是交给
//! [`crate::Partition`] 摊余数——摊出来的不等宽分片会让各 rank 的权重形状对不上。

pub use partition::Partition;

use crate::collective::Reduce;
use diagnostic::{Code, Error, Result};

/// 张量并行里线性层的两种切法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sharding {
    /// 按列切：切输出维，分片直接进下一层，不需要通信。
    Column,
    /// 按行切：切输入维，每个 rank 产出一份部分和，必须求和归约。
    Row,
}

impl Sharding {
    /// 全部切法，供日志与测试遍历。
    pub const ALL: [Self; 2] = [Self::Column, Self::Row];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Column => "column",
            Self::Row => "row",
        }
    }

    /// 被切的是权重 `[out, in]` 的哪一维。
    pub fn axis(self) -> usize {
        match self {
            Self::Column => 0,
            Self::Row => 1,
        }
    }

    /// 合并各 rank 的输出需要哪种归约；`Column` 不需要（`None`）。
    pub fn merge(self) -> Option<Reduce> {
        match self {
            Self::Column => None,
            Self::Row => Some(Reduce::Sum),
        }
    }
}

/// 一层线性权重的 TP 切分计划：切法、份数与完整的 `[out, in]` 形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    sharding: Sharding,
    partition: Partition,
    out: usize,
    inp: usize,
}

impl Tiling {
    /// 由切法、份数与完整权重形状构造；被切的那一维必须能被份数整除。
    ///
    /// # Errors
    ///
    /// 份数为零、形状里出现零宽，或被切维不能被份数整除。
    pub fn new(sharding: Sharding, parts: usize, out: usize, inp: usize) -> Result<Self> {
        let partition = Partition::new(sharding.axis(), parts)?;
        if out == 0 || inp == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("权重形状 [{out}, {inp}] 含零宽：切出来是空分片"),
            )
            .with("shard::Tiling::new"));
        }
        let split = Self::dimension(sharding, out, inp);
        if !split.is_multiple_of(parts) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "{} 的第 {} 维长度 {split} 不能被 {parts} 整除：各 rank 的权重形状会不一致",
                    sharding.as_str(),
                    sharding.axis()
                ),
            )
            .with("shard::Tiling::new"));
        }
        Ok(Self {
            sharding,
            partition,
            out,
            inp,
        })
    }

    /// 切法。
    pub fn sharding(&self) -> Sharding {
        self.sharding
    }

    /// 份数（通常等于 `Topology::tp()`）。
    pub fn parts(&self) -> usize {
        self.partition.parts()
    }

    /// 第 `rank` 份的权重形状 `[out, in]`。
    pub fn weight_shape(&self, rank: usize) -> Result<[usize; 2]> {
        let (_, width) = self.range(rank)?;
        Ok(match self.sharding {
            Sharding::Column => [width, self.inp],
            Sharding::Row => [self.out, width],
        })
    }

    /// 第 `rank` 份在完整权重里的 `(起点, 长度)`，沿 [`Sharding::axis`] 那一维。
    pub fn range(&self, rank: usize) -> Result<(usize, usize)> {
        let split = Self::dimension(self.sharding, self.out, self.inp);
        self.partition.slice(split, rank)
    }

    /// 合并各 rank 的输出需要哪种归约；不需要通信时是 `None`。
    pub fn merge(&self) -> Option<Reduce> {
        self.sharding.merge()
    }

    /// 被切的维在完整权重里的长度。
    fn dimension(sharding: Sharding, out: usize, inp: usize) -> usize {
        match sharding {
            Sharding::Column => out,
            Sharding::Row => inp,
        }
    }
}

mod partition {
    //! 单个维度上的均匀切分：[`Partition`]。
    //!
    //! 切分是**恰好覆盖**而不是"尽量均分"：`0..parts` 的区间首尾相接、互不重叠，并集恰好是
    //! `0..dimension`。余数按份序摊给前面几份（`10` 切 `3` 份得 `4 / 3 / 3`）。
    //!
    //! 这个口径服务两件不同的事，因此**不提供**第二套"整除才合法"的规则：
    //!
    //! - 流水并行的层分配本来就不整除（`32` 层切 `3` 段），摊余数是唯一正确的做法；
    //! - 张量并行要求整除（各 rank 的张量形状必须一致），这条要求由 [`crate::Tiling`] 在构造时
    //!   显式校验，而不是靠切分器在底层替它猜——切分器不该知道调用方能不能容忍不等宽。

    use diagnostic::{Code, Error, Result};

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
                return Err(
                    Error::new(Code::Invalid, "切分份数不能为 0").with("shard::Partition::new")
                );
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

        /// 第 `part` 份的 `(起点, 长度)`；前 `dimension % parts` 份各多拿一个元素。
        pub fn slice(&self, dimension: usize, part: usize) -> Result<(usize, usize)> {
            if part >= self.parts {
                return Err(Error::new(
                    Code::Invalid,
                    format!("份下标 {part} 超出 0..{}", self.parts),
                )
                .with("shard::Partition::slice"));
            }
            let width = dimension / self.parts;
            let extra = dimension % self.parts;
            // 前 `extra` 份宽 `width + 1`，其后的份宽 `width`；起点由份序累加，首尾相接。
            let lead = part.min(extra);
            let start = part * width + lead;
            let take = if part < extra { width + 1 } else { width };
            Ok((start, take))
        }

        /// 全部份的 `(起点, 长度)`，按份序。
        pub fn cover(&self, dimension: usize) -> Result<Vec<(usize, usize)>> {
            (0..self.parts)
                .map(|part| self.slice(dimension, part))
                .collect()
        }

        /// 第 `part` 份在 `dimension` 上的终点（不含）；与 [`Partition::slice`] 同一份算术。
        pub fn end(&self, dimension: usize, part: usize) -> Result<usize> {
            let (start, take) = self.slice(dimension, part)?;
            Ok(start + take)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// 切分的判据是**恰好覆盖**：首尾相接、互不重叠、并集是 `0..dimension`。
        ///
        /// 这条用例遍历余数不为零的形状：`0..parts` 任一份漏算，并集就会短一截或重叠。
        #[test]
        fn every_partition_covers_the_dimension_exactly_once() {
            for dimension in [0, 1, 2, 3, 10, 17] {
                for parts in 1..=5 {
                    let partition = Partition::new(0, parts).expect("切分");
                    let cover = partition.cover(dimension).expect("各份");
                    let mut expected = 0;
                    for (part, (start, take)) in cover.iter().enumerate() {
                        assert_eq!(*start, expected, "第 {part} 份没接上前一份");
                        assert_eq!(partition.end(dimension, part).expect("终点"), start + take);
                        expected += take;
                    }
                    assert_eq!(expected, dimension, "{dimension} 切 {parts} 份没有覆盖完");
                }
            }
        }

        #[test]
        fn the_remainder_goes_to_the_earlier_parts() {
            let partition = Partition::new(1, 3).expect("切分");
            assert_eq!(partition.axis(), 1);
            assert_eq!(partition.cover(10).expect("各份"), [(0, 4), (4, 3), (7, 3)]);
            assert_eq!(partition.cover(0).expect("空维"), [(0, 0), (0, 0), (0, 0)]);
        }

        #[test]
        fn an_index_outside_the_split_is_rejected() {
            let partition = Partition::new(0, 4).expect("切分");
            assert_eq!(partition.slice(16, 3), Ok((12, 4)));
            assert_eq!(partition.slice(16, 4).unwrap_err().code(), Code::Invalid);
            assert_eq!(Partition::new(0, 0).unwrap_err().code(), Code::Invalid);
        }
    }
}

#[cfg(test)]
#[path = "../tests/tiling_tests.rs"]
mod tests;
