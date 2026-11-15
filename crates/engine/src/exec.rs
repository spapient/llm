//! 提交与执行：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`] 与 [`Executor`] 契约。
//!
//! **拥有**：`Stream`、`Graph`、`Node`、`Launch`、`Executor`。**不拥有**：编译 kernel
//! （`cutile::runner` 负责"源文件 → 可执行文件"，产物按 `(模板, 档案, flags)` 由 `cutile::build` 缓存；本 crate 只负责
//! "图提交到流"，每 step 都发生，见 `docs/adr/0002-single-owner-per-type.md`）、缓冲分配（`mem`）。
//! **依赖**：`diagnostic`。**被依赖**：`engine` 的其余模块（本模块并入后与它们同 crate）。
//!
//! 现在只有契约：没有真实的流、没有设备队列。缺的是实现，不是接口——后端的差异会落在
//! `Executor` 的实现里，而不是散到调度与模型代码中。
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

use diagnostic::Result;

/// 一次 kernel 的网格与线程块几何。
///
/// 三轴是设备 API 的公共形态（CUDA / Metal / HIP 都是 `(grid, block)`）；DSL 的宿主 API
/// （cutile README §4.4，P0 尚未实现）由分片比推出 `grid.x`，这里只负责承载它。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Launch {
    /// 网格大小（每轴多少个块）。
    pub grid: [u32; 3],
    /// 线程块大小（每块多少线程）。
    pub block: [u32; 3],
}

impl Launch {
    /// 由网格与线程块构造。
    pub fn new(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self { grid, block }
    }

    /// 总线程数：`grid` 与 `block` 的逐轴乘积再相乘。
    pub fn total_threads(&self) -> u64 {
        let grid = u64::from(self.grid[0])
            .saturating_mul(u64::from(self.grid[1]))
            .saturating_mul(u64::from(self.grid[2]));
        let block = u64::from(self.block[0])
            .saturating_mul(u64::from(self.block[1]))
            .saturating_mul(u64::from(self.block[2]));
        grid.saturating_mul(block)
    }
}

/// 图里的一个节点：一次 kernel 调用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    name: String,
    launch: Launch,
}

impl Node {
    /// 由算子名与几何构造。
    pub fn new(name: impl Into<String>, launch: Launch) -> Self {
        Self {
            name: name.into(),
            launch,
        }
    }

    /// 算子名（与 `op::Op::name` 一致，便于把执行与声明对上）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 几何。
    pub fn launch(&self) -> Launch {
        self.launch
    }
}

/// 一张图：按插入顺序提交的节点列表。
///
/// 不是"优化后的 IR"——`cutile` 的定位是"无 IR 的 lowering"，图只是提交单元，
/// 不做算子融合或重排。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// 空图。
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一个节点。
    pub fn push(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// 全部节点，按插入顺序。
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 节点数。
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// 一条提交队列：同一流内的节点按序执行。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream {
    id: u32,
}

impl Stream {
    /// 由编号构造。
    pub fn new(id: u32) -> Self {
        Self { id }
    }

    /// 编号。
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// 后端要实现的东西：把一张图提交到一条流。
///
/// 实现放在后端（CPU 线程池 / CUDA / Metal …）；调度与模型代码只认这个 trait，
/// 因此换后端不改上层。
pub trait Executor {
    /// 提交并发起执行；返回时表示"已入队"，不表示已完成（同步由流负责）。
    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()>;
}

#[cfg(test)]
mod tests;
