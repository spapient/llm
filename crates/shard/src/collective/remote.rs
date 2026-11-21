//! 远端集合通信占位：[`Remote`]。
//!
//! [`Collective`] 的进程内实现是 [`crate::Local`]；跨进程归约要走 NCCL / MPI / Gloo 等远端后端。
//! 本刀只落占位骨架：不持网络 / 通信句柄，不接 FFI，不写 `unsafe`；`all_reduce` 入口统一返回
//! [`Code::Unsupported`]。真实远端实现（组内握手、通信句柄池、拓扑绑定与心跳）随后续后端刀
//! 填入本模块；上层调用点仍是同一个 [`Collective`] 端口。

use super::{Collective, Group, Reduce};
use diagnostic::{Code, Error, Result};

/// 远端集合通信占位后端：与 [`crate::Local`] 同一 [`Collective`] 端口，未挂载真实通信栈。
///
/// 不保存进程拓扑、通信句柄或 rank 镜像；构造是无状态占位，只证明第二后端能挂到同一契约上。
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
