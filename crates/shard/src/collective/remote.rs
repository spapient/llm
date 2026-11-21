//! 远端集合通信占位：[`Remote`]。

use super::{Collective, Group, Reduce};
use diagnostic::{Code, Error, Result};

/// 远端集合通信占位后端：与 [`crate::Local`] 同一 [`Collective`] 端口，未挂载真实通信栈。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Remote;

impl Remote {
    /// 构造占位远端后端；不绑定任何进程组、通信句柄或拓扑。
    pub fn new() -> Self {
        Self
    }
}

impl Collective for Remote {
    fn all_reduce(&mut self, _group: &Group, _reduce: Reduce, _buffer: &mut [f32]) -> Result<()> {
        Err(Error::new(
            Code::Unsupported,
            "远端集合通信未挂载：跨进程 all-reduce 在占位适配上不可用",
        )
        .with("shard::Remote::all_reduce"))
    }
}

#[cfg(test)]
#[path = "../../tests/collective_remote_tests.rs"]
mod tests;
