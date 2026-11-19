//! 远端集合通信占位：[`Remote`]。
//!
//! [`Collective`] 的进程内实现是 [`crate::Local`]；跨进程归约要走 NCCL / MPI / Gloo 等远端后端。
//! 本刀只落占位骨架：不持网络 / 通信句柄，不接 FFI，不写 `unsafe`；`all_reduce` 入口统一返回
//! [`Code::Unsupported`]。真实远端实现（组内握手、通信句柄池、拓扑绑定与心跳）随后续后端刀
//! 填入本模块；上层调用点仍是同一个 [`Collective`] 端口。

use crate::collective::Collective;
use crate::group::Group;
use crate::reduce::Reduce;
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
mod tests {
use crate::{Collective, Group, Local, Parallel, Reduce, Remote, Topology};
use diagnostic::Code;

/// 占位远端后端拒绝任何集合操作；缓冲原样保留，不产生半个归约结果。
#[test]
fn all_reduce_is_unsupported_and_leaves_the_buffer_untouched() {
    let mut remote = Remote::new();
    let group = Group::new([0, 1]).expect("双进程组");
    let mut buffer = [1.0f32, 2.0, 3.0];

    let error = remote
        .all_reduce(&group, Reduce::Sum, &mut buffer)
        .expect_err("远端未挂载应拒绝归约");
    assert_eq!(error.code(), Code::Unsupported);
    assert_eq!(buffer, [1.0, 2.0, 3.0], "拒绝时不得改动调用方缓冲");
}

/// [`Local`] 与 [`Remote`] 能放进同一个 `Box<dyn Collective>`：端口契约不随后端切换。
#[test]
fn local_and_remote_share_the_collective_port() {
    let topology = Topology::new(2, 1, 1).expect("tp 2");
    let mut local: Box<dyn Collective> = Box::new(Local::new(topology, 2).expect("宽度非零"));
    let mut remote: Box<dyn Collective> = Box::new(Remote::new());

    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    let mut buffer = [0.0f32; 2];
    local
        .all_reduce(&group, Reduce::Sum, &mut buffer)
        .expect("进程内归约");
    assert_eq!(
        remote
            .all_reduce(&group, Reduce::Sum, &mut buffer)
            .unwrap_err()
            .code(),
        Code::Unsupported
    );
}
}

