//! 参与一次通信的进程组：[`Group`]。
//!
//! 构造时要求**升序、去重、非空**：集合通信里"组内下标"必须稳定，否则归约顺序不固定，
//! 浮点求和的最后几位会随启动顺序变化。

use diagnostic::{Code, Error, Result};

/// 参与一次通信的进程组。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    ranks: Vec<u32>,
}

impl Group {
    /// 校验并构造。
    pub fn new(ranks: impl Into<Vec<u32>>) -> Result<Self> {
        let ranks = ranks.into();
        if ranks.is_empty() {
            return Err(Error::new(Code::Invalid, "通信组不能为空").with("shard::Group::new"));
        }
        for pair in ranks.windows(2) {
            if pair[0] >= pair[1] {
                return Err(Error::new(
                    Code::Invalid,
                    format!("通信组必须升序且不重复：{} >= {}", pair[0], pair[1]),
                )
                .with("shard::Group::new"));
            }
        }
        Ok(Self { ranks })
    }

    /// 组内全部 rank。
    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    /// 组大小。
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// 组内下标（用于切片"我负责哪一段"）。
    pub fn index_of(&self, rank: u32) -> Option<usize> {
        self.ranks.iter().position(|item| *item == rank)
    }

    /// 是否包含某个 rank。
    pub fn contains(&self, rank: u32) -> bool {
        self.index_of(rank).is_some()
    }
}

#[cfg(test)]
#[path = "../../tests/collective_group_tests.rs"]
mod tests;
