//! 执行器缝与提交记账：`Executor` 契约、`Trace` 装饰器、本地执行器。

use super::*;

/// 后端要实现的东西：把一张图提交到一条流，并按编号登记内核与宿主缓冲。
///
/// 实现放在后端（CPU 线程池 / CUDA / Metal …）；调度与模型代码只认这个 trait，
/// 因此换后端不改上层。它是本仓的 **Backend 基座**（ADR 0051）：Burn 的 `Backend` 在
/// llm 里由本 trait 承担，装饰器（[`Trace`] 等）以同样的方式实现它并转发。
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

    /// 登记一段内核体；执行时按 [`Program::name`] 查找。
    ///
    /// 默认拒绝：只记账的队列执行器没有内核表，静默接受会让"提交成功、同步才发现没实现"
    /// 变成一种运行期惊喜。
    fn register_program(&mut self, _program: Box<dyn Program>) -> Result<()> {
        Err(Error::new(
            Code::Unsupported,
            "该执行器没有内核表".to_string(),
        ))
    }

    /// 绑定一块 `f32` 操作数；返回被替换掉的旧块。
    ///
    /// 默认返回 `None`：没有宿主缓冲的执行器不拥有这个编号，调用方按缺失处理。
    fn bind_f32(&mut self, _id: u32, _values: Vec<f32>) -> Option<Vec<f32>> {
        None
    }

    /// 绑定一块打包字节操作数；返回被替换掉的旧块。
    fn bind_bytes(&mut self, _id: u32, _values: Vec<u8>) -> Option<Vec<u8>> {
        None
    }

    /// 按编号取 `f32` 操作数；缺失或形态不符为 `None`。
    fn fetch_f32(&self, _id: u32) -> Option<&[f32]> {
        None
    }

    /// 按编号取打包字节操作数；缺失或形态不符为 `None`。
    fn fetch_bytes(&self, _id: u32) -> Option<&[u8]> {
        None
    }
}

/// 执行基座的装饰器：包住任意 [`Executor`]，记录每次提交的节点名，其余方法原样转发。
///
/// 这是"装饰器叠在基座上"的第一个实例（ADR 0051）：上层只认 `E: Executor`，因此换成
/// `Trace<CpuExecutor>` 不需要改模型代码；审计拿到的是**真实提交序**，不是装配侧另记的一份。
/// 本装饰器不改变执行语义、不做重排或融合（ADR 0017 无 IR 冻结）。
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

    fn bind_f32(&mut self, id: u32, values: Vec<f32>) -> Option<Vec<f32>> {
        self.inner.bind_f32(id, values)
    }

    fn bind_bytes(&mut self, id: u32, values: Vec<u8>) -> Option<Vec<u8>> {
        self.inner.bind_bytes(id, values)
    }

    fn fetch_f32(&self, id: u32) -> Option<&[f32]> {
        self.inner.fetch_f32(id)
    }

    fn fetch_bytes(&self, id: u32) -> Option<&[u8]> {
        self.inner.fetch_bytes(id)
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
/// `context` 是执行器名（CPU / Metal 各自报自己的入口）。
pub fn check_operands(
    node: &Node,
    unbound: impl Fn(u32) -> bool,
    context: &'static str,
) -> Result<()> {
    for operand in node.inputs().iter().chain(node.outputs()) {
        if unbound(operand.id()) {
            return Err(Error::new(
                Code::Invalid,
                format!("节点 `{}` 的操作数 {} 未绑定", node.name(), operand.id()),
            )
            .with(context));
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
        .with(context));
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
///
/// 形态只承载元素宽度这一事实；形状与条目名描述留在装配侧（G4），本层不解释编号。
#[derive(Debug, Clone, PartialEq)]
pub enum Buffer {
    /// 32 位浮点。
    F32(Vec<f32>),
    /// 按位打包的整型：i8 逐元素一字节，i4 每字节两个 nibble；字节序与内核一致。
    U8(Vec<u8>),
}

impl Buffer {
    /// 当成 `f32` 读；不是浮点缓冲时为 `None`。
    pub fn f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }

    /// 当成打包字节读；不是字节缓冲时为 `None`。
    pub fn u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(values) => Some(values),
            Self::F32(_) => None,
        }
    }

    /// 元素个数：f32 元素数或打包字节数。
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U8(values) => values.len(),
        }
    }

    /// 是否没有元素。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 只读视图。
    pub(super) fn view(&self) -> View<'_> {
        match self {
            Self::F32(values) => View::F32(values),
            Self::U8(values) => View::U8(values),
        }
    }

    /// 可写视图。
    pub(super) fn view_mut(&mut self) -> ViewMut<'_> {
        match self {
            Self::F32(values) => ViewMut::F32(values),
            Self::U8(values) => ViewMut::U8(values),
        }
    }
}

/// 只读视图：一次 [`Program::run`] 的输入操作数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum View<'a> {
    /// 32 位浮点。
    F32(&'a [f32]),
    /// 按位打包的整型。
    U8(&'a [u8]),
}

impl<'a> View<'a> {
    /// 当成 `f32` 读；不是浮点缓冲时为 `None`。
    pub fn f32(&self) -> Option<&'a [f32]> {
        match self {
            Self::F32(values) => Some(*values),
            Self::U8(_) => None,
        }
    }

    /// 当成打包字节读；不是字节缓冲时为 `None`。
    pub fn u8(&self) -> Option<&'a [u8]> {
        match self {
            Self::U8(values) => Some(*values),
            Self::F32(_) => None,
        }
    }
}

/// 可写视图：一次 [`Program::run`] 的输出操作数。
#[derive(Debug, PartialEq)]
pub enum ViewMut<'a> {
    /// 32 位浮点。
    F32(&'a mut [f32]),
    /// 按位打包的整型。
    U8(&'a mut [u8]),
}

impl<'a> ViewMut<'a> {
    /// 当成 `f32` 写；不是浮点缓冲时为 `None`。
    pub fn f32_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }

    /// 当成打包字节写；不是字节缓冲时为 `None`。
    pub fn u8_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::U8(values) => Some(values),
            Self::F32(_) => None,
        }
    }
}
