//! 集合通信契约：[`Group`]、[`Reduce`] 与 [`Collective`]。
//!
//! **拥有**：`Group`、`Reduce`、`Collective`。**不拥有**：切分拓扑（归 `shard`）、通信后端
//! （NCCL / RCCL / CCL / gloo 是 `Collective` 的实现，P2 落地）。
//! **依赖**：`diagnostic`。**被依赖**：`shard`。
//!
//! 先定契约的意义在于：TP/PP/DP 三条并行路径都能写成"对某个 `Group` 做某类归约"，
//! 于是后端可以换、拓扑可以调，而上层不需要知道进程间到底怎么传。
//!
//! # 例子
//!
//! ```
//! use llm_comm::{Group, Reduce};
//!
//! let group = Group::new([0, 1]).expect("升序且非空");
//! assert_eq!(group.size(), 2);
//! assert_eq!(group.index_of(1), Some(1));
//! assert_eq!(Reduce::Sum.as_str(), "sum");
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};

/// 归约种类闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reduce {
    /// 求和（梯度同步的默认选择）。
    Sum,
    /// 取最大（张量并行里的数值稳定选项）。
    Max,
    /// 取最小。
    Min,
    /// 取乘积。
    Prod,
}

impl Reduce {
    /// 全部种类，供日志与测试遍历。
    pub const ALL: [Self; 4] = [Self::Sum, Self::Max, Self::Min, Self::Prod];

    /// 稳定的短名。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
            Self::Prod => "prod",
        }
    }
}

/// 参与一次通信的进程组。
///
/// 构造时要求**升序、去重、非空**：集合通信里"组内下标"必须稳定，否则归约顺序不固定，
/// 浮点求和的最后几位会随启动顺序变化。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    ranks: Vec<u32>,
}

impl Group {
    /// 校验并构造。
    pub fn new(ranks: impl Into<Vec<u32>>) -> Result<Self> {
        let ranks = ranks.into();
        if ranks.is_empty() {
            return Err(Error::new(Code::Invalid, "通信组不能为空").with("comm::Group::new"));
        }
        for pair in ranks.windows(2) {
            if pair[0] >= pair[1] {
                return Err(Error::new(
                    Code::Invalid,
                    format!("通信组必须升序且不重复：{} >= {}", pair[0], pair[1]),
                )
                .with("comm::Group::new"));
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

/// 通信后端要实现的东西：对组内缓冲做一次集合操作。
///
/// 只有 `all_reduce` 一个方法：它是 TP/DP 的全部需求（`all_gather` / `reduce_scatter` 都能由它
/// 与 `shard` 的切分组合出来）。后端实现不同，上层调用点不变。
pub trait Collective {
    /// 对 `buffer` 做原地归约；返回时表示"已完成"。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_must_be_sorted_and_unique() {
        let group = Group::new([0, 2, 4]).expect("升序");
        assert_eq!(group.ranks(), [0, 2, 4]);
        assert_eq!(group.size(), 3);
        assert_eq!(group.index_of(4), Some(2));
        assert!(!group.contains(1));
        assert_eq!(Group::new([]).unwrap_err().code(), Code::Invalid);
        assert_eq!(Group::new([2, 1]).unwrap_err().code(), Code::Invalid);
        assert_eq!(Group::new([1, 1]).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn reduction_kinds_are_a_closed_set() {
        let names: Vec<&str> = Reduce::ALL.iter().map(|kind| kind.as_str()).collect();
        assert_eq!(names, ["sum", "max", "min", "prod"]);
    }

    #[test]
    fn a_single_rank_group_is_legal() {
        struct Local;
        impl Collective for Local {
            fn all_reduce(
                &mut self,
                _group: &Group,
                reduce: Reduce,
                buffer: &mut [f32],
            ) -> Result<()> {
                if reduce == Reduce::Sum {
                    // 单进程组：归约是恒等。
                    buffer[0] += 0.0;
                }
                Ok(())
            }
        }
        let group = Group::new([0]).expect("单进程组");
        let mut buffer = [1.0f32];
        Local
            .all_reduce(&group, Reduce::Sum, &mut buffer)
            .expect("归约");
        assert_eq!(buffer, [1.0]);
    }
}
