//! CPU 执行体：内核注册、参数绑定，以及解释一节点的执行器。

use super::*;

/// 一段可执行的内核体：按节点名字登记，在绑定好的缓冲上就地计算。
///
/// 本层不编译 kernel（那是 `cutile` 的 runner 与 `build`），也不解释操作数编号——
/// `inputs` / `outputs` 是执行器按节点角色拆好的视图，与 [`Node::inputs`] /
/// [`Node::outputs`] 一一对应、顺序一致。因此"哪个编号是张量"这件事到不了本层：
/// 执行器只接线，内核只算数。
pub trait Program {
    /// 与 [`Node::name`] 相同的内核名：注册表按它查，重名即注册失败。
    fn name(&self) -> &str;

    /// 运行一次：`inputs` 只读、`outputs` 可写，两组编号互不重叠。
    fn run(&self, launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()>;
}

/// 执行器持有的操作数表：编号 ↔ 平坦的 `f32` 缓冲。
///
/// 编号由编排层分配（ADR 0025 §2：本层只认不透明引用）。表按编号排序，因此诊断输出与
/// 遍历顺序与插入顺序无关——同一张图的两次执行打印出来是同一份。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bindings {
    values: std::collections::BTreeMap<u32, Buffer>,
}

impl Bindings {
    /// 空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 绑定一个 `f32` 编号；返回被替换掉的旧浮点块。
    pub fn bind_f32(&mut self, id: u32, values: Vec<f32>) -> Option<Vec<f32>> {
        match self.values.insert(id, Buffer::F32(values)) {
            Some(Buffer::F32(old)) => Some(old),
            Some(Buffer::U8(_)) | None => None,
        }
    }

    /// 绑定一个打包字节编号；返回被替换掉的旧字节块。
    pub fn bind_u8(&mut self, id: u32, values: Vec<u8>) -> Option<Vec<u8>> {
        match self.values.insert(id, Buffer::U8(values)) {
            Some(Buffer::U8(old)) => Some(old),
            Some(Buffer::F32(_)) | None => None,
        }
    }

    /// 某个编号的 `f32` 内容；字节缓冲返回 `None`。
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        self.values.get(&id).and_then(Buffer::f32)
    }

    /// 某个编号的打包字节内容；浮点缓冲返回 `None`。
    pub fn get_u8(&self, id: u32) -> Option<&[u8]> {
        self.values.get(&id).and_then(Buffer::u8)
    }

    /// 某个编号绑定的缓冲（只读）。
    pub fn buffer(&self, id: u32) -> Option<&Buffer> {
        self.values.get(&id)
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
    values: Vec<Buffer>,
}

impl Taken {
    /// 按节点角色取只读视图。
    fn views(&self) -> Vec<View<'_>> {
        self.values.iter().map(Buffer::view).collect()
    }

    /// 按节点角色取可写视图。
    fn views_mut(&mut self) -> Vec<ViewMut<'_>> {
        self.values.iter_mut().map(Buffer::view_mut).collect()
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

    /// 绑定一个 `f32` 操作数编号；返回被替换掉的旧浮点块。
    pub fn bind(&mut self, id: u32, values: Vec<f32>) -> Option<Vec<f32>> {
        self.bindings.bind_f32(id, values)
    }

    /// 绑定一个打包字节操作数编号；返回被替换掉的旧字节块。
    pub fn bind_u8(&mut self, id: u32, values: Vec<u8>) -> Option<Vec<u8>> {
        self.bindings.bind_u8(id, values)
    }

    /// 当前操作数表。
    pub fn bindings(&self) -> &Bindings {
        &self.bindings
    }

    /// 某个编号的 `f32` 内容；字节缓冲返回 `None`。
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        self.bindings.get(id)
    }

    /// 某个编号的打包字节内容；浮点缓冲返回 `None`。
    pub fn get_u8(&self, id: u32) -> Option<&[u8]> {
        self.bindings.get_u8(id)
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
        let input_views = inputs.views();
        let mut output_views = outputs.views_mut();
        let result = self.programs[index].run(node.launch(), &input_views, &mut output_views);
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
        check_operands(
            node,
            |id| self.bindings.buffer(id).is_none(),
            "engine::CpuExecutor::run",
        )
    }

    /// 把一组缓冲从操作数表里借出来；缺一个就整体放回（不留半个状态）。
    ///
    /// 同一编号在本节点重复作为只读输入（如逐元素二元两路借同一块残差流）：第二次取到的是
    /// 已借出那份的克隆，输入视图互不持有可写引用；放回时按编号写回，不改动执行几何。
    fn take(&mut self, operands: &[Operand]) -> Taken {
        let mut ids = Vec::with_capacity(operands.len());
        let mut values = Vec::with_capacity(operands.len());
        for operand in operands {
            match self.bindings.values.remove(&operand.id()) {
                Some(value) => {
                    ids.push(operand.id());
                    values.push(value);
                }
                None => match ids.iter().position(|&id| id == operand.id()) {
                    Some(taken_at) => values.push(values[taken_at].clone()),
                    None => {
                        self.restore(Taken { ids, values });
                        return Taken {
                            ids: Vec::new(),
                            values: Vec::new(),
                        };
                    }
                },
            }
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
        enqueue(&mut self.queue, graph, stream);
        Ok(())
    }

    fn register_program(&mut self, program: Box<dyn Program>) -> Result<()> {
        CpuExecutor::register(self, program)
    }

    fn bind_f32(&mut self, id: u32, values: Vec<f32>) -> Option<Vec<f32>> {
        CpuExecutor::bind(self, id, values)
    }

    fn bind_bytes(&mut self, id: u32, values: Vec<u8>) -> Option<Vec<u8>> {
        CpuExecutor::bind_u8(self, id, values)
    }

    fn fetch_f32(&self, id: u32) -> Option<&[f32]> {
        CpuExecutor::get(self, id)
    }

    fn fetch_bytes(&self, id: u32) -> Option<&[u8]> {
        CpuExecutor::get_u8(self, id)
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
