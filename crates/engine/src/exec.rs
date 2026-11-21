//! 提交与执行：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`]、[`Operand`] 与 [`Executor`] 契约。
//!
//! 本模块按主题分成四个文件：[`submit`]（提交契约）、[`queue`]（执行器缝与记账）、
//! [`memory`]（缓冲视图）、[`cpu`]（CPU 执行体）。四段之间只经公开类型相接，没有共享的私有状态。
//! **拥有**：`Stream`、`Graph`、`Node`、`Launch`、`Operand`、`Executor`。**不拥有**：编译 kernel
//! （`cutile::runner` 负责"源文件 → 可执行文件"，产物按 `(模板, 档案, flags)` 由 `cutile::build` 缓存；本 crate 只负责
//! "图提交到流"，每 step 都发生）、缓冲分配（`storage`）。
//! **依赖**：`diagnostic`、`tensor`、`storage`（仅 [`Allocator`] trait 对象；`Buffer`/`Allocator`
//! 类型定义归 `storage`）。**被依赖**：`engine` 的其余模块（本模块并入后与它们同 crate）。
//!
//! 节点绑定的只是不透明操作数引用（[`Operand`]：编号 + 只读/可写角色）；编号到实际缓冲的解析、
//! 缓冲生命周期与设备句柄都在后端边界侧（ADR 0025 §2/§3），本层不引用 `op` 算子闭集。
//! 设备后端的执行器只持有分配器 [`Allocator`] trait 对象（与设备同生命周期），不持有
//! [`storage::Buffer`] 字节；缓冲的分配、解析与写回由后端执行体经分配器完成。CPU 执行器
//! 是例外、而且必须是例外：它的缓冲就是 [`Bindings`] 里的 `Vec<f32>`——CPU 上没有独立的
//! “后端缓冲对象”，硬造一个只会多一层拷贝。
//!
//! [`Executor`] 有两个实现：[`LocalExecutor`] 只记队列（提交、按流排水，持注入的分配器），
//! [`CpuExecutor`] 在同一套提交语义上真的执行登记过的 [`Program`]，结果写回 [`Bindings`] 里的
//! 缓冲。设备后端（CUDA / Metal）仍缺实现，但差异只落在 `Executor` 的实现里，不会散到调度与
//! 模型代码中。
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

mod cpu;
mod queue;

pub use cpu::{Bindings, CpuExecutor, Program};
pub use queue::{Buffer, View, ViewMut};
pub use queue::{Executor, LocalExecutor, Queued, Trace};
pub(crate) use queue::{check_operands, enqueue};

use diagnostic::{Code, Error, Result};
use std::fmt;
use storage::Allocator;

#[cfg(test)]
#[path = "../tests/exec.rs"]
mod tests;

/// 定长类型化闭集标量槽（D9 / ADR 0043）：首批仅 rope 解码位置。
///
/// 槽位集合冻结在 ADR 0043，新增槽位走 ADR 修订；不用字符串键名值表，也不用 `HashMap` 标量
/// 上下文，热路径没有查表。位置这类语义参数不编进 1 元素 f32 缓冲——缓冲切片通道仍由
/// [`Program::run`] 的 `inputs` / `outputs` 承担。
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
///
/// 三轴是设备 API 的公共形态（CUDA / Metal / HIP 都是 `(grid, block)`）；DSL 的宿主 API
/// （cutile README §4.4，P0 尚未实现）由分片比推出 `grid.x`，这里只负责承载它。
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
