//! 张量并行的权重切分：[`Sharding`] 与 [`Tiling`]。

use crate::collective::Reduce;
use diagnostic::Result;

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
    ///
    /// 全 crate 只有这一处把切法翻译成轴：分片形状、随份数变化的长度都从它派生。
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
///
/// 分片是**整除**的等分：`Tiling::new` 拒绝任何摊余数的形状，因此第 `rank` 份永远是
/// `(rank · 被切维长度 / 份数, 被切维长度 / 份数)`。TP 要求各 rank 的权重形状一致，
/// 有余数就没有合法的分片，故这里不保留"前几份多拿一个"的均匀切分——那条分支在任何
/// 可达路径上都取不到。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    sharding: Sharding,
    parts: usize,
    shape: [usize; 2],
}

impl Tiling {
    /// 由切法、份数与完整权重形状构造；被切的那一维必须能被份数整除。
    ///
    /// # Errors
    ///
    /// 份数为零、形状里出现零宽，或被切维不能被份数整除。
    pub fn new(sharding: Sharding, parts: usize, out: usize, inp: usize) -> Result<Self> {
        if parts == 0 {
            fail!("shard::Tiling::new", "切分份数不能为 0");
        }
        if out == 0 || inp == 0 {
            fail!(
                "shard::Tiling::new",
                "权重形状 [{out}, {inp}] 含零宽：切出来是空分片"
            );
        }
        let shape = [out, inp];
        let split = shape[sharding.axis()];
        if !split.is_multiple_of(parts) {
            fail!(
                "shard::Tiling::new",
                "{} 的第 {} 维长度 {split} 不能被 {parts} 整除：各 rank 的权重形状会不一致",
                sharding.as_str(),
                sharding.axis()
            );
        }
        Ok(Self {
            sharding,
            parts,
            shape,
        })
    }

    /// 切法。
    pub fn sharding(&self) -> Sharding {
        self.sharding
    }

    /// 份数（通常等于 `Topology::tp()`）。
    pub fn parts(&self) -> usize {
        self.parts
    }

    /// 第 `rank` 份的权重形状 `[out, in]`：完整形状把被切维换成该份的长度。
    pub fn weight_shape(&self, rank: usize) -> Result<[usize; 2]> {
        let (_, width) = self.range(rank)?;
        let mut shape = self.shape;
        shape[self.sharding.axis()] = width;
        Ok(shape)
    }

    /// 第 `rank` 份在完整权重里的 `(起点, 长度)`，沿 [`Sharding::axis`] 那一维。
    pub fn range(&self, rank: usize) -> Result<(usize, usize)> {
        if rank >= self.parts {
            fail!(
                "shard::Tiling::range",
                "份下标 {rank} 超出 0..{}",
                self.parts
            );
        }
        let width = self.width();
        Ok((rank * width, width))
    }

    /// 被切维在每份里的长度：构造期已保证整除。
    fn width(&self) -> usize {
        self.shape[self.sharding.axis()] / self.parts
    }
}

#[cfg(test)]
#[path = "../tests/tiling_tests.rs"]
mod tests;
