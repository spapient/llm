//! 执行器缝与提交记账：`Executor` 契约、`Trace` 装饰器、本地执行器。

use crate::exec::{Device, Graph, Node, Program, Stream};
use crate::invalid;
use diagnostic::{Code, Error, Result};
use std::fmt;
use storage::Allocator;

/// 后端要实现的东西：把一张图提交到一条流，并按编号登记内核与宿主缓冲。
pub trait Executor {
    /// 该执行器绑定的设备。
    fn device(&self) -> &Device;

    /// 提交并发起执行；返回时表示"已入队"，不表示已完成（同步由流负责）。
    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()>;

    /// 等待一条流上已提交的工作完成。
    fn sync(&mut self, _stream: Stream) -> Result<()> {
        Ok(())
    }

    /// 登记一段内核体；执行时按 [`Program::name`] 查找。
    fn register_program(&mut self, _program: Box<dyn Program>) -> Result<()> {
        Err(Error::new(
            Code::Unsupported,
            "该执行器没有内核表".to_string(),
        ))
    }

    /// 绑定一段配置操作数（浮点或打包字节，lane 标签由 [`Buf`] 自带）；返回被替换掉的旧块。
    fn bind(&mut self, _id: u32, _values: Buf) -> Option<Buf> {
        None
    }

    /// 按编号取回一段配置操作数；未绑定为 `None`，lane 形态在缓冲内自证。
    fn fetch(&self, _id: u32) -> Option<&Buf> {
        None
    }
}

/// 执行基座的装饰器：包住任意 [`Executor`]，记录每次提交的节点名，其余方法原样转发。
#[derive(Debug)]
pub struct Trace<E> {
    inner: E,
    submitted: Vec<Vec<String>>,
}

impl<E> Trace<E> {
    /// 包住一个执行器；记录从空开始。
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            submitted: Vec::new(),
        }
    }

    /// 每次 [`Executor::submit`] 记录到的节点名（按提交序，含空图）。
    pub fn submitted(&self) -> &[Vec<String>] {
        &self.submitted
    }

    /// 被装饰的执行器（只读）。
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// 被装饰的执行器（可写）；配置内核表等一次性装配走这里。
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }
}

impl<E: Executor> Executor for Trace<E> {
    fn device(&self) -> &Device {
        self.inner.device()
    }

    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()> {
        self.submitted.push(
            graph
                .nodes()
                .iter()
                .map(|node| node.name().to_owned())
                .collect(),
        );
        self.inner.submit(graph, stream)
    }

    fn sync(&mut self, stream: Stream) -> Result<()> {
        self.inner.sync(stream)
    }

    fn register_program(&mut self, program: Box<dyn Program>) -> Result<()> {
        self.inner.register_program(program)
    }

    fn bind(&mut self, id: u32, values: Buf) -> Option<Buf> {
        self.inner.bind(id, values)
    }

    fn fetch(&self, id: u32) -> Option<&Buf> {
        self.inner.fetch(id)
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

/// 入队提交：把图里每个节点按流入队。Local、Cpu 与 Metal 共用同一份口径。
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
            return Err(invalid(
                ctx,
                format!("节点 `{}` 的操作数 {} 未绑定", node.name(), operand.id()),
            ));
        }
    }
    let clash = node
        .outputs()
        .iter()
        .any(|output| node.inputs().iter().any(|input| input.id() == output.id()));
    if clash {
        return Err(invalid(
            ctx,
            format!("节点 `{}` 有一个编号既是输入又是输出", node.name()),
        ));
    }
    Ok(())
}

impl Executor for LocalExecutor {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()> {
        enqueue(&mut self.queue, graph, stream);
        Ok(())
    }

    fn sync(&mut self, stream: Stream) -> Result<()> {
        self.queue.retain(|queued| queued.stream != stream);
        Ok(())
    }
}

/// 一段已绑定缓冲的存储形态：32 位浮点，或按位打包的字节（i4/i8 lane）。
/// 公共定义在 [`storage::LaneBuf`]，本别名保留执行缝既有命名。
pub type Buf = storage::LaneBuf;

/// 尚未接入运行时的设备后端位：只登记设备标签，不持有设备句柄、内核表或宿主缓冲。
///
/// CUDA / Ascend 等厂商运行时执行器在真机管线落地前，先在同一条 [`Executor`] 缝上落位：
/// 除 [`device()`](Executor::device) 给出标签外，提交、登记内核、绑定与取回操作数一律回
/// [`Code::Unsupported`] 并带站点——缺证据不静默成功，也不另造一套运行时 trait（ADR 0051 §4）。
/// 真机管线接入后，把该位升级为持有句柄的具体执行器，缝的其余消费方不改。
pub struct VacantExecutor {
    device: Device,
}

impl fmt::Debug for VacantExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VacantExecutor")
            .field("device", &self.device)
            .finish()
    }
}

impl VacantExecutor {
    /// 由厂商与架构标签构造一个未接入运行时的设备后端位。
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// 这一位登记的厂商标签（`cuda` / `ascend` …），只作台账标记，不产生热路径分派。
    pub fn vendor(&self) -> &str {
        self.device.vendor()
    }

    /// 未接入运行时：提交一幅图被拒，错误类别 [`Code::Unsupported`]。
    fn rejected(&self, action: &'static str) -> Error {
        Error::new(
            Code::Unsupported,
            format!(
                "{} 后端尚未接入运行时执行器，{}",
                self.device.vendor(),
                action
            ),
        )
        .with("engine::VacantExecutor")
    }
}

impl Executor for VacantExecutor {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, _graph: &Graph, _stream: Stream) -> Result<()> {
        Err(self.rejected("不能提交图"))
    }

    fn register_program(&mut self, _program: Box<dyn Program>) -> Result<()> {
        Err(self.rejected("不能登记内核"))
    }

    fn bind(&mut self, _id: u32, _values: Buf) -> Option<Buf> {
        None
    }

    fn fetch(&self, _id: u32) -> Option<&Buf> {
        None
    }
}

/// 只读视图：一次 [`Program::run`] 的输入操作数。公共定义在 [`storage::LaneView`]。
pub type View<'a> = storage::LaneView<'a>;

/// 可写视图：一次 [`Program::run`] 的输出操作数。公共定义在 [`storage::LaneViewMut`]。
pub type ViewMut<'a> = storage::LaneViewMut<'a>;
