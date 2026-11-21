//! 通信后端要实现的东西：[`Collective`]。

use diagnostic::Result;

pub use group::Group;
pub use reduce::Reduce;
pub use remote::Remote;

/// 通信后端要实现的东西：对组内缓冲做一次集合操作。
pub trait Collective {
    /// 对 `buffer` 做原地归约；返回时表示"已完成"。
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buffer: &mut [f32]) -> Result<()>;
}

mod group;

mod reduce;

mod remote;

#[cfg(test)]
#[path = "../tests/collective_tests.rs"]
mod tests;
