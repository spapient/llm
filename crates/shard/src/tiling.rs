//! 张量并行的权重切分：[`Sharding`] 与 [`Tiling`]。

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

mod partition;

#[cfg(test)]
#[path = "../tests/tiling_tests.rs"]
mod tests;
