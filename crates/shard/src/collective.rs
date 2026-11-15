//! 通信后端要实现的东西：[`Collective`]。
//!
//! 只有 `all_reduce` 一个方法：它是 TP/DP 的全部需求（`all_gather` / `reduce_scatter` 都能由它
//! 与切分拓扑组合出来）。后端实现不同，上层调用点不变。

use crate::group::Group;
use crate::reduce::Reduce;
use diagnostic::Result;

/// 通信后端要实现的东西：对组内缓冲做一次集合操作。
pub trait Collective {
    /// 对 `buffer` 做原地归约；返回时表示"已完成"。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()>;
}

#[cfg(test)]
#[path = "../tests/collective_tests.rs"]
mod tests;
