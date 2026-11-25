//! 设备运行时提交契约：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`]、[`Operand`] 与
//! [`Executor`] 核心缝。
//!
//! 这些类型只描述一次图提交与设备标签，不认识算子档案、设备厂商句柄或 CPU 内核表；
//! 设备后端在 [`crate::runtime`] 岛内实现本缝，上层编排只按名字与几何提交节点。

use diagnostic::{Code, Error, Result};
use std::sync::Arc;

/// 单表达式构造、访问与绑定方法统一在此声明；复杂校验和执行分支不进入宏。
macro_rules! value_methods {
    ($($type:ty { $($(#[$meta:meta])* $name:ident($($args:tt)*) $(->$ret:ty)? = $body:expr; )* })*) => {
        $(impl $type {
            $($(#[$meta])* pub fn $name($($args)*) $(->$ret)? { $body })*
        })*
    };
}

/// 定长类型化闭集标量槽（D9 / ADR 0043）：首批仅 rope 解码位置。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ScalarSlots {
    /// `rope_f32` 的解码位置；该算子尚未接入模型路径时保持 `None`。
    rope_position: Option<u32>,
}

/// 一次 kernel 的网格与线程块几何。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Launch {
    /// 网格大小（每轴多少个块）。
    pub grid: [u32; 3],
    /// 线程块大小（每块多少线程）。
    pub block: [u32; 3],
    /// 定长类型化标量槽（D9 / ADR 0043）；几何之外的语义参数走这里。
    scalars: ScalarSlots,
}

/// 操作数角色：只读输入还是可写输出。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandRole {
    /// 只读输入：节点消费它，不写入。
    Input,
    /// 可写输出：节点写入它。
    Output,
}

/// 一个节点操作数的不透明引用：编号 + 角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Operand {
    /// 编排层分配的不透明引用编号。
    id: u32,
    /// 只读/可写角色。
    role: OperandRole,
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

/// 一张图：按插入顺序提交的节点列表；克隆共享节点，首次修改时才复制。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    nodes: Arc<Vec<Node>>,
}

/// 一条提交队列：同一流内的节点按序执行。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream {
    id: u32,
}

/// 设备标识与能力标签：`vendor` / `arch` 是登记标签，不持有设备私有句柄的具体类型。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Device {
    vendor: String,
    arch: String,
}

value_methods! {
    ScalarSlots {
        /// 空槽：一个标量都没填。
        new() -> Self = Self { rope_position: None };
        /// rope 解码位置（只读；未填为 `None`）。
        rope_position(&self) -> Option<u32> = self.rope_position;
        /// 是否一个槽位都没填。
        is_empty(&self) -> bool = self.rope_position.is_none();
    }
    Operand {
        /// 只读输入操作数。
        input(id: u32) -> Self = Self { id, role: OperandRole::Input };
        /// 可写输出操作数。
        output(id: u32) -> Self = Self { id, role: OperandRole::Output };
        /// 不透明引用编号。
        id(&self) -> u32 = self.id;
        /// 操作数角色。
        role(&self) -> OperandRole = self.role;
    }
    Node {
        /// 由算子名与几何构造，暂不绑定操作数（绑定由 [`Node::bind_input`] / [`Node::bind_output`] 追加）。
        new(name: impl Into<String>, launch: Launch) -> Self = Self { name: name.into(), launch, inputs: Vec::new(), outputs: Vec::new() };
        /// 追加一个只读输入操作数。
        bind_input(&mut self, id: u32) -> &mut Self = { self.inputs.push(Operand::input(id)); self };
        /// 追加一个可写输出操作数。
        bind_output(&mut self, id: u32) -> &mut Self = { self.outputs.push(Operand::output(id)); self };
        /// 算子名（与档案算子名一致，便于把执行与声明对上）。
        name(&self) -> &str = &self.name;
        /// 几何。
        launch(&self) -> Launch = self.launch;
        /// 标量槽（只读，D9 / ADR 0043）。
        scalars(&self) -> &ScalarSlots = &self.launch.scalars;
        /// 标量槽（可写，D9 / ADR 0043）。
        scalars_mut(&mut self) -> &mut ScalarSlots = &mut self.launch.scalars;
        /// 只读输入操作数，按声明顺序。
        inputs(&self) -> &[Operand] = &self.inputs;
        /// 可写输出操作数，按声明顺序。
        outputs(&self) -> &[Operand] = &self.outputs;
    }
    Stream {
        /// 由编号构造。
        new(id: u32) -> Self = Self { id };
        /// 编号。
        id(&self) -> u32 = self.id;
    }
    Device {
        /// 由厂商与架构标签构造。
        new(vendor: impl Into<String>, arch: impl Into<String>) -> Self = Self { vendor: vendor.into(), arch: arch.into() };
        /// 厂商标签（`cpu` / `metal` / `cuda` …）。
        vendor(&self) -> &str = &self.vendor;
        /// 架构标签（`aarch64` / `neon` / `sm_90` …），只作登记，不产生热路径分发。
        arch(&self) -> &str = &self.arch;
    }
    Graph {
        /// 空图。
        new() -> Self = Self::default();
    }
}

impl ScalarSlots {
    /// 写入 rope 解码位置。
    pub fn set_rope_position(&mut self, position: u32) {
        self.rope_position = Some(position);
    }
}

impl Graph {
    /// 追加一个节点；与其它克隆共享时先复制节点表，保持写时独立。
    pub fn push(&mut self, node: Node) {
        Arc::make_mut(&mut self.nodes).push(node);
    }

    /// 全部节点，按插入顺序。
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 全部节点的可变视图：装配宿主在提交前回填逐步变化的标量槽（如 rope 位置），不改执行几何。
    pub fn nodes_mut(&mut self) -> &mut [Node] {
        Arc::make_mut(&mut self.nodes).as_mut_slice()
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

impl Launch {
    /// 由网格与线程块构造；标量槽默认全空。
    pub fn new(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            scalars: ScalarSlots::new(),
        }
    }

    /// 替换整套标量槽（构造期装配；几何之外的参数一次性给定）。
    #[must_use]
    pub fn with_scalars(mut self, scalars: ScalarSlots) -> Self {
        self.scalars = scalars;
        self
    }

    /// 标量槽（只读）。
    pub fn scalars(&self) -> &ScalarSlots {
        &self.scalars
    }

    /// 标量槽（可写）。
    pub fn scalars_mut(&mut self) -> &mut ScalarSlots {
        &mut self.scalars
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

/// 设备后端要实现的核心缝：把一张图提交到一条流，并等待该流完成。
///
/// CPU 内核表、宿主缓冲绑定等上层装配能力不在本缝；它们由引擎侧扩展 trait 补充。
pub trait Executor {
    /// 该执行器绑定的设备。
    fn device(&self) -> &Device;

    /// 提交并发起执行；返回时表示"已入队"，不表示已完成（同步由流负责）。
    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()>;

    /// 等待一条流上已提交的工作完成。
    fn sync(&mut self, _stream: Stream) -> Result<()> {
        Ok(())
    }
}

/// 一条已提交到本地队列、尚未随其流同步的节点快照；共享图只增加一次 Arc 计数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Queued {
    stream: Stream,
    graph: Graph,
    idx: usize,
}

value_methods! {
    Queued {
        /// 归属流。
        stream(&self) -> Stream = self.stream;
        /// 节点快照。
        node(&self) -> &Node = &self.graph.nodes()[self.idx];
    }
}

/// 入队提交：把图里每个节点按流入队。各设备后端共用同一份口径。
pub fn enqueue(queue: &mut Vec<Queued>, graph: &Graph, stream: Stream) {
    queue.extend((0..graph.nodes().len()).map(|idx| Queued {
        stream,
        graph: graph.clone(),
        idx,
    }));
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

/// 微内核与网格共用的面积计算：乘法溢出即报错，不回绕。
pub(crate) fn area(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{name} 元素数溢出")))
}

/// 坐标或尺寸的受检加法。
pub(crate) fn add(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{name}溢出")))
}

/// 主机 ABI 的 C `int` 转换；超范围时拒绝调用，不做截断。
pub(crate) fn c_int(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::new(Code::Invalid, format!("{name} 超出 C int 范围")))
}
