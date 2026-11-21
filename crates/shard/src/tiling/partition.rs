//! 单个维度上的均匀切分：[`Partition`]。

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
#[path = "../../tests/tiling_partition_tests.rs"]
mod tests;
