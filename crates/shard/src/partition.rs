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
#[path = "../tests/partition_tests.rs"]
mod tests;
