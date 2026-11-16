//! 提交与执行：[`Stream`]、[`Graph`]、[`Node`]、[`Launch`]、[`Operand`] 与 [`Executor`] 契约。
//!
//! **拥有**：`Stream`、`Graph`、`Node`、`Launch`、`Operand`、`Executor`。**不拥有**：编译 kernel
//! （`cutile::runner` 负责"源文件 → 可执行文件"，产物按 `(模板, 档案, flags)` 由 `cutile::build` 缓存；本 crate 只负责
//! "图提交到流"，每 step 都发生，见 `docs/adr/0002-single-owner-per-type.md`）、缓冲分配（`storage`）。
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

use diagnostic::{Code, Error, Result};
use std::fmt;
use storage::Allocator;

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

    /// 等待一条流上已提交的工作完成。
    ///
    /// 这是“提交不等于完成”的兑现处：CPU 实现会把队列里的节点真的执行完再返回，设备后端
    /// 在此做事件同步。跳过它就去读结果，看到的是未初始化的缓冲。
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
///
/// 构造时注入 [`Allocator`] trait 对象；执行器只持有分配器，不持有 [`storage::Buffer`] 字节，
/// 分配与写回由后端执行体经分配器完成。分配器是有状态后端句柄，因此本类型不派生克隆与相等。
pub struct LocalExecutor {
    device: Device,
    allocator: Box<dyn Allocator>,
    queue: Vec<Queued>,
}

impl fmt::Debug for LocalExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalExecutor")
            .field("device", &self.device)
            .field("pending", &self.queue.len())
            .finish()
    }
}

impl LocalExecutor {
    /// 绑定一台 CPU 设备与其缓冲分配器，构造空队列后端。
    pub fn new(device: Device, allocator: Box<dyn Allocator>) -> Self {
        Self {
            device,
            allocator,
            queue: Vec::new(),
        }
    }

    /// 绑定的缓冲分配器（trait 对象；只持分配器，不持缓冲字节）。
    pub fn allocator(&self) -> &dyn Allocator {
        self.allocator.as_ref()
    }

    /// 可变借用分配器：分配与归还经此入口，缓冲字节不升到本层。
    pub fn allocator_mut(&mut self) -> &mut dyn Allocator {
        self.allocator.as_mut()
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

/// 一段可执行的内核体：按节点名字登记，在绑定好的缓冲上就地计算。
///
/// 本层不编译 kernel（那是 `cutile` 的 `runner` 与 `build`），也不解释操作数编号——
/// `inputs` / `outputs` 是执行器按节点角色拆好的切片，与 [`Node::inputs`] /
/// [`Node::outputs`] 一一对应、顺序一致。因此"哪个编号是张量"这件事到不了本层：
/// 执行器只接线，内核只算数。
pub trait Program {
    /// 与 [`Node::name`] 相同的内核名：注册表按它查，重名即注册失败。
    fn name(&self) -> &str;

    /// 运行一次：`inputs` 只读、`outputs` 可写，两组编号互不重叠。
    fn run(&self, launch: Launch, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()>;
}

/// 执行器持有的操作数表：编号 ↔ 平坦的 `f32` 缓冲。
///
/// 编号由编排层分配（ADR 0025 §2：本层只认不透明引用）。表按编号排序，因此诊断输出与
/// 遍历顺序与插入顺序无关——同一张图的两次执行打印出来是同一份。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bindings {
    values: std::collections::BTreeMap<u32, Vec<f32>>,
}

impl Bindings {
    /// 空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 绑定一个编号；返回被替换掉的旧缓冲。
    pub fn bind(&mut self, id: u32, values: Vec<f32>) -> Option<Vec<f32>> {
        self.values.insert(id, values)
    }

    /// 某个编号的当前内容。
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        self.values.get(&id).map(Vec::as_slice)
    }

    /// 已绑定的编号数。
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 是否一个编号都没绑。
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// 执行器从 [`Bindings`] 里借出的一组缓冲：编号与内容成对，用完原样放回。
///
/// 借出/放回是"同时拿到只读输入与可写输出"的唯一安全办法：直接从同一张表里取
/// `&[f32]` 与 `&mut [f32]`，借用检查器无法知道编号不重叠（编号是运行期的值）。
#[derive(Default)]
struct Taken {
    ids: Vec<u32>,
    values: Vec<Vec<f32>>,
}

impl Taken {
    /// 按节点角色取切片；`Taken` 被 mut 借用时给出可写视图。
    fn views(&self) -> Vec<&[f32]> {
        self.values.iter().map(Vec::as_slice).collect()
    }
}

/// CPU 执行器：把图提交到本地队列，并在 [`sync`](Executor::sync) 时按图序**真的**跑内核。
///
/// 与 [`LocalExecutor`] 的区别在于排水那一步：`LocalExecutor` 只把队列里的条目丢掉，
/// [`CpuExecutor`] 会为每个节点找到登记过的 [`Program`]、把操作数从 [`Bindings`] 里借出来、
/// 执行、再放回。于是同一个 [`Executor`] 契约下，CPU 后端是"提交即入队、同步才计算"，
/// 而不是 `submit` 就同步算完——那会让上层无法按流写出正确的依赖顺序。
///
/// `sync` 只排**指定流**：别的流的条目原样留着（跨流先后不属于本层，见 [`Executor::sync`]）。
/// 一条条目只有在执行成功之后才出队，所以失败时它仍然在队列里，"还没跑完"这件事查得到。
///
/// 执行顺序严格按 [`Graph::nodes`]：本层不做重排、不做融合（`cutile` 的"无 IR"前提）。
pub struct CpuExecutor {
    device: Device,
    programs: Vec<Box<dyn Program>>,
    bindings: Bindings,
    queue: Vec<Queued>,
}

impl fmt::Debug for CpuExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuExecutor")
            .field("device", &self.device)
            .field("programs", &self.programs.len())
            .field("bindings", &self.bindings.len())
            .field("pending", &self.queue.len())
            .finish()
    }
}

impl CpuExecutor {
    /// 绑定一台 CPU 设备；内核表、操作数表与队列都是空的。
    pub fn new(device: Device) -> Self {
        Self {
            device,
            programs: Vec::new(),
            bindings: Bindings::new(),
            queue: Vec::new(),
        }
    }

    /// 尚未同步的提交，按入队顺序（跨流交错时保持提交顺序）。
    pub fn pending(&self) -> &[Queued] {
        &self.queue
    }

    /// 尚未同步的提交条数。
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// 队列是否为空。
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// 登记一段内核体；同名已登记时返回 `Invalid`，而不是让后登记的那份静默顶替。
    pub fn register(&mut self, program: Box<dyn Program>) -> Result<()> {
        if self.position(program.name()).is_ok() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "内核 `{}` 已登记：重名会让「哪一份在跑」变成一个运行期问题",
                    program.name()
                ),
            )
            .with("engine::CpuExecutor::register"));
        }
        self.programs.push(program);
        Ok(())
    }

    /// 绑定一个操作数编号；返回被替换掉的旧缓冲。
    pub fn bind(&mut self, id: u32, values: Vec<f32>) -> Option<Vec<f32>> {
        self.bindings.bind(id, values)
    }

    /// 当前操作数表。
    pub fn bindings(&self) -> &Bindings {
        &self.bindings
    }

    /// 某个编号的当前内容。
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        self.bindings.get(id)
    }

    /// 执行一个节点：查内核、借出操作数、跑、放回。
    ///
    /// # Errors
    ///
    /// 内核未登记、操作数未绑定、同一节点里一个编号既是输入又是输出、或内核本身报错。
    pub fn run(&mut self, node: &Node) -> Result<()> {
        let index = self.position(node.name())?;
        self.check_operands(node)?;
        let mut inputs = self.take(node.inputs());
        let mut outputs = self.take(node.outputs());
        let mut writable: Vec<&mut [f32]> =
            outputs.values.iter_mut().map(Vec::as_mut_slice).collect();
        let result = self.programs[index].run(node.launch(), &inputs.views(), &mut writable);
        drop(writable);
        // 缓冲先放回、再报错：执行失败不该把操作数留在一个"查不到"的状态里。
        self.restore(std::mem::take(&mut inputs));
        self.restore(std::mem::take(&mut outputs));
        result.map_err(|error| error.with("engine::CpuExecutor::run"))
    }

    /// 按名字找内核下标。
    fn position(&self, name: &str) -> Result<usize> {
        self.programs
            .iter()
            .position(|program| program.name() == name)
            .ok_or_else(|| {
                Error::new(Code::Invalid, format!("内核 `{name}` 未登记"))
                    .with("engine::CpuExecutor::run")
            })
    }

    /// 节点自己的操作数约束：每个编号都要已绑定，且输入与输出不能是同一个编号。
    fn check_operands(&self, node: &Node) -> Result<()> {
        for operand in node.inputs().iter().chain(node.outputs()) {
            if self.bindings.get(operand.id()).is_none() {
                return Err(Error::new(
                    Code::Invalid,
                    format!("节点 `{}` 的操作数 {} 未绑定", node.name(), operand.id()),
                )
                .with("engine::CpuExecutor::run"));
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
            .with("engine::CpuExecutor::run"));
        }
        Ok(())
    }

    /// 把一组缓冲从操作数表里借出来；缺一个就整体放回（不留半个状态）。
    fn take(&mut self, operands: &[Operand]) -> Taken {
        let mut ids = Vec::with_capacity(operands.len());
        let mut values = Vec::with_capacity(operands.len());
        for operand in operands {
            let Some(value) = self.bindings.values.remove(&operand.id()) else {
                self.restore(Taken { ids, values });
                return Taken {
                    ids: Vec::new(),
                    values: Vec::new(),
                };
            };
            ids.push(operand.id());
            values.push(value);
        }
        Taken { ids, values }
    }

    /// 把借出的缓冲按编号放回。
    fn restore(&mut self, taken: Taken) {
        for (id, value) in taken.ids.into_iter().zip(taken.values) {
            self.bindings.values.insert(id, value);
        }
    }
}

impl Executor for CpuExecutor {
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

    /// 排这一条流上已提交的节点：按入队顺序执行，成功一条才出队一条。
    ///
    /// 别的流不动：跨流先后不属于本层（[`Executor::sync`] 只承诺"这条流上已提交的工作完成"）。
    /// 失败时出错的那条留在队列里，调用方能看到"它还没跑完"，而不是被悄悄丢掉。
    fn sync(&mut self, stream: Stream) -> Result<()> {
        let mut index = 0;
        while index < self.queue.len() {
            if self.queue[index].stream() != stream {
                index += 1;
                continue;
            }
            let node = self.queue[index].node().clone();
            self.run(&node)?;
            self.queue.remove(index);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/exec_tests.rs"]
mod tests;
