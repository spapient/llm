//! 设备运行时提交契约：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`]、[`Operand`] 与
//! [`Executor`] 核心缝。
//!
//! 这些类型只描述一次图提交与设备标签，不认识算子档案、设备厂商句柄或 CPU 内核表；
//! 设备后端在 [`crate::runtime`] 岛内实现本缝，上层编排只按名字与几何提交节点。

use diagnostic::Result;

/// 定长类型化闭集标量槽（D9 / ADR 0043）：首批仅 rope 解码位置。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ScalarSlots {
    /// `rope_f32` 的解码位置；该算子尚未接入模型路径时保持 `None`。
    rope_position: Option<u32>,
}

impl ScalarSlots {
    /// 空槽：一个标量都没填。
    pub const fn new() -> Self {
        Self {
            rope_position: None,
        }
    }

    /// 写入 rope 解码位置。
    pub fn set_rope_position(&mut self, position: u32) {
        self.rope_position = Some(position);
    }

    /// rope 解码位置（只读；未填为 `None`）。
    pub fn rope_position(&self) -> Option<u32> {
        self.rope_position
    }

    /// 是否一个槽位都没填。
    pub fn is_empty(&self) -> bool {
        self.rope_position.is_none()
    }
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

    /// 算子名（与档案算子名一致，便于把执行与声明对上）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 几何。
    pub fn launch(&self) -> Launch {
        self.launch
    }

    /// 标量槽（只读，D9 / ADR 0043）。
    pub fn scalars(&self) -> &ScalarSlots {
        &self.launch.scalars
    }

    /// 标量槽（可写，D9 / ADR 0043）。
    pub fn scalars_mut(&mut self) -> &mut ScalarSlots {
        &mut self.launch.scalars
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

    /// 全部节点的可变视图：装配宿主在提交前回填逐步变化的标量槽（如 rope 位置），不改执行几何。
    pub fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
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

/// 设备标识与能力标签：`vendor` / `arch` 是登记标签，不持有设备私有句柄的具体类型。
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
