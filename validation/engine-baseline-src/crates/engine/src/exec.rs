//! 提交与执行：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`]、[`Operand`] 与 [`Executor`] 契约。
//!
//! 本模块只有提交契约这一件事（图、流、节点、操作数）。设备无关提交几何与核心执行缝住在
//! [`prof::runtime::exec`]；执行器扩展缝、记账与 CPU 执行体在 `crate::queue` / `crate::cpu`。
//! 三段只经公开类型相接，没有共享的私有状态。
//!
//! 节点绑定的只是不透明操作数引用（[`Operand`]：编号 + 只读/可写角色）；编号到实际缓冲的解析、
//! 缓冲生命周期与设备句柄都在后端边界侧（ADR 0025 §2/§3），本层不引用 `prof` 算子闭集。
//!
//! [`Executor`] 有两个实现：[`LocalExecutor`] 只记队列（提交、按流排水，持注入的分配器），
//! [`CpuExecutor`] 在同一套提交语义上真的执行登记过的 [`Program`]，结果写回 [`Bindings`] 里的
//! 缓冲。厂商设备后端在 `llm-prof::runtime` 岛内实现核心缝，不反向依赖本层。
//!
//! # 例子
//!
//! ```
//! use llm_engine::exec::{Graph, Launch, Node, Stream};
//!
//! let mut graph = Graph::new();
//! graph.push(Node::new("gemv_i4", Launch::new([4, 1, 1], [128, 1, 1])));
//! assert_eq!(graph.len(), 1);
//! assert_eq!(graph.nodes()[0].launch().total_threads(), 512);
//! assert_eq!(Stream::new(0).id(), 0);
//! ```

pub use crate::cpu::{Bindings, CpuExecutor, Program};
pub use crate::queue::{Buf, View, ViewMut};
pub use crate::queue::{Executor, LocalExecutor, Queued, Trace};
pub use crate::queue::{VacantExecutor, check_operands, enqueue};
pub use prof::runtime::exec::{
    Device, Graph, Launch, Node, Operand, OperandRole, ScalarSlots, Stream,
};

#[cfg(test)]
#[path = "../tests/exec.rs"]
mod tests;
