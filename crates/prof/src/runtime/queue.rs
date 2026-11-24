//! 提交记账的设备无关辅助：已入队快照、按流入队与操作数约束校验。
//!
//! [`Queued`] 只保存一次提交的节点副本与流标签，不持设备句柄或缓冲；各厂商后端复用同一份口径。

use crate::runtime::exec::{Graph, Node, Stream};
use diagnostic::{Code, Error, Result};

/// 一条已提交到本地队列、尚未随其流同步的节点快照。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Queued {
    stream: Stream,
    node: Node,
}

impl Queued {
    /// 归属流。
    pub fn stream(&self) -> Stream {
        self.stream
    }

    /// 节点快照。
    pub fn node(&self) -> &Node {
        &self.node
    }
}

/// 入队提交：把图里每个节点按流入队。各设备后端共用同一份口径。
pub fn enqueue(queue: &mut Vec<Queued>, graph: &Graph, stream: Stream) {
    for node in graph.nodes() {
        queue.push(Queued {
            stream,
            node: node.clone(),
        });
    }
}

/// 节点操作数约束：每个编号都要已绑定，且输入与输出不能同号。`unbound` 判定"未绑定"，
/// `ctx` 是执行器名（CPU / Metal 各自报自己的入口）。
pub fn check_operands(node: &Node, unbound: impl Fn(u32) -> bool, ctx: &'static str) -> Result<()> {
    for operand in node.inputs().iter().chain(node.outputs()) {
        if unbound(operand.id()) {
            return Err(Error::new(
                Code::Invalid,
                format!("节点 `{}` 的操作数 {} 未绑定", node.name(), operand.id()),
            )
            .with(ctx));
        }
    }
    let clash = node
        .outputs()
        .iter()
        .any(|output| node.inputs().iter().any(|input| input.id() == output.id()));
    if clash {
        return Err(Error::new(
            Code::Invalid,
            format!("节点 `{}` 有一个编号既是输入又是输出", node.name()),
        )
        .with(ctx));
    }
    Ok(())
}
