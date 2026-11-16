//! 提交与执行：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`]、[`Operand`] 与 [`Executor`] 契约。
//!
//! **拥有**：`Stream`、`Graph`、`Node`、`Launch`、`Operand`、`Executor`。**不拥有**：编译 kernel
//! （`cutile::runner` 负责"源文件 → 可执行文件"，产物按 `(模板, 档案, flags)` 由 `cutile::build` 缓存；本 crate 只负责
//! "图提交到流"，每 step 都发生，见 `docs/adr/0002-single-owner-per-type.md`）、缓冲分配（`storage`）。
//! **依赖**：`diagnostic`。**被依赖**：`engine` 的其余模块（本模块并入后与它们同 crate）。
//!
//! 节点绑定的只是不透明操作数引用（[`Operand`]：编号 + 只读/可写角色）；编号到实际缓冲的解析、
//! 缓冲生命周期与设备句柄都在后端边界侧（ADR 0025 §2/§3），本层不持缓冲字节、不引用 `op` 算子闭集。
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

/// 操作数角色：只读输入还是可写输出。
///
/// 语义对齐算子签名里的可变标记（`const` 指针为输入、可变指针为输出），但本层不依赖 `op` 声明，
/// 角色词汇在这里自定；具体缓冲的可写性由后端按算子签名复核。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandRole {
    /// 只读输入：节点消费它，不写入。
    Input,
    /// 可写输出：节点写入它。
    Output,
}

/// 一个节点操作数的不透明引用：编号 + 角色。
///
/// 引擎侧只持引用编号，不持缓冲字节、不持设备地址；编号到实际缓冲的映射、
/// 拥有/借用生命周期与设备句柄都在后端边界侧完成（ADR 0025 §2/§3）。编号的分配与去重
/// 归编排层，本层不解释编号含义。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Operand {
    /// 编排层分配的不透明引用编号。
    id: u32,
    /// 只读/可写角色。
    role: OperandRole,
}

impl Operand {
    /// 只读输入操作数。
    pub fn input(id: u32) -> Self {
        Self {
            id,
            role: OperandRole::Input,
        }
    }

    /// 可写输出操作数。
    pub fn output(id: u32) -> Self {
        Self {
            id,
            role: OperandRole::Output,
        }
    }

    /// 不透明引用编号。
    pub fn id(&self) -> u32 {
        self.id
    }

    /// 操作数角色。
    pub fn role(&self) -> OperandRole {
        self.role
    }
}

/// 图里的一个节点：一次 kernel 调用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    name: String,
    launch: Launch,
    /// 只读输入操作数，按声明顺序。
    inputs: Vec<Operand>,
    /// 可写输出操作数，按声明顺序。
    outputs: Vec<Operand>,
}

impl Node {
    /// 由算子名与几何构造，暂不绑定操作数（绑定由 [`Node::bind_input`] / [`Node::bind_output`] 追加）。
    pub fn new(name: impl Into<String>, launch: Launch) -> Self {
        Self {
            name: name.into(),
            launch,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// 追加一个只读输入操作数。
    pub fn bind_input(&mut self, id: u32) -> &mut Self {
        self.inputs.push(Operand::input(id));
        self
    }

    /// 追加一个可写输出操作数。
    pub fn bind_output(&mut self, id: u32) -> &mut Self {
        self.outputs.push(Operand::output(id));
        self
    }

    /// 算子名（与 `op::Op::name` 一致，便于把执行与声明对上）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 几何。
    pub fn launch(&self) -> Launch {
        self.launch
    }

    /// 只读输入操作数，按声明顺序。
    pub fn inputs(&self) -> &[Operand] {
        &self.inputs
    }

    /// 可写输出操作数，按声明顺序。
    pub fn outputs(&self) -> &[Operand] {
        &self.outputs
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

/// 设备标识与能力标签：`vendor` / `arch` 是登记标签，不持有设备私有句柄的具体类型
/// （ADR 0025 §1）。与 `prof` 设备档案的对应由编排层完成，本层不反向依赖档案闭集。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Device {
    vendor: String,
    arch: String,
}

impl Device {
    /// 由厂商与架构标签构造。
    pub fn new(vendor: impl Into<String>, arch: impl Into<String>) -> Self {
        Self {
            vendor: vendor.into(),
            arch: arch.into(),
        }
    }

    /// 厂商标签（`cpu` / `metal` / `cuda` …）。
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// 架构标签（`aarch64` / `neon` / `sm_90` …），只作登记，不产生热路径分发。
    pub fn arch(&self) -> &str {
        &self.arch
    }
}

/// 后端要实现的东西：把一张图提交到一条流。
///
/// 实现放在后端（CPU 线程池 / CUDA / Metal …）；调度与模型代码只认这个 trait，
/// 因此换后端不改上层。
pub trait Executor {
    /// 该执行器绑定的设备。
    fn device(&self) -> &Device;

    /// 提交并发起执行；返回时表示"已入队"，不表示已完成（同步由流负责）。
    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()>;

    /// 等待一条流上已提交的工作完成。CPU 实现可立即返回；设备后端在此做事件同步。
    fn sync(&mut self, _stream: Stream) -> Result<()> {
        Ok(())
    }
}

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

/// 同步 CPU 后端：把图节点按提交顺序记入本地队列，[`sync`](Executor::sync) 按流排水。
///
/// 这是 [`Executor`] 的第一个落地实现：单线程、不提交设备命令、不解析操作数引用——节点携带的
/// 不透明编号到缓冲的解析、算子内核的挂载与 cutile 产物复用属于后续接内核执行体的后端（见模块顶部
/// 边界）。本实现只守住契约：同一流内节点按图序入队，`sync` 后排干该流；编排层可以先接上执行入口，执行体后补。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalExecutor {
    device: Device,
    queue: Vec<Queued>,
}

impl LocalExecutor {
    /// 绑定一台 CPU 设备，构造空队列后端。
    pub fn new(device: Device) -> Self {
        Self {
            device,
            queue: Vec::new(),
        }
    }

    /// 尚未同步的提交，按入队顺序。
    pub fn pending(&self) -> &[Queued] {
        &self.queue
    }

    /// 队列里尚未同步的条数。
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// 队列是否为空。
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl Executor for LocalExecutor {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()> {
        for node in graph.nodes() {
            self.queue.push(Queued {
                stream,
                node: node.clone(),
            });
        }
        Ok(())
    }

    fn sync(&mut self, stream: Stream) -> Result<()> {
        self.queue.retain(|queued| queued.stream != stream);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/exec_tests.rs"]
mod tests;
