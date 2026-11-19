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
mod tests {
use super::*;

#[test]
fn a_single_rank_group_is_legal() {
    struct Local;
    impl Collective for Local {
        fn all_reduce(&mut self, _group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()> {
            if reduce == Reduce::Sum {
                // 故意做一次可观察的变换：恒等实现在空函数下也能通过，测不出任何东西。
                for value in buffer.iter_mut() {
                    *value *= 2.0;
                }
            }
            Ok(())
        }
    }
    let group = Group::new([0]).expect("单进程组");
    let mut buffer = [1.0f32];
    Local
        .all_reduce(&group, Reduce::Sum, &mut buffer)
        .expect("归约");
    assert_eq!(buffer, [2.0], "变换必须真的落到缓冲上");
}
}

