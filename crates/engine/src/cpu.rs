//! CPU 执行体：内核注册、参数绑定，以及解释一节点的执行器。

use crate::exec::{Device, Graph, Launch, Node, Operand, Stream};
use crate::invalid;
use crate::queue::{Buf, Executor, Queued, View, ViewMut, check_operands, enqueue};
use diagnostic::Result;
use std::fmt;

/// 一段可执行的内核体：按节点名字登记，在绑定好的缓冲上就地计算。
pub trait Program {
    /// 与 [`Node::name`] 相同的内核名：注册表按它查，重名即注册失败。
    fn name(&self) -> &str;

    /// 运行一次：`inputs` 只读、`outputs` 可写，两组编号互不重叠。
    fn run(&self, launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()>;
}

/// 执行器持有的操作数表：编号 ↔ 平坦的 `f32` 缓冲。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bindings {
    values: std::collections::BTreeMap<u32, Buf>,
}

impl Bindings {
    /// 空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 绑定一段配置缓冲；同 lane 旧块原样返回，lane 不符或原先为空返回 `None`。
    pub fn bind(&mut self, id: u32, values: Buf) -> Option<Buf> {
        match self.values.entry(id) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(values);
                None
            }
            std::collections::btree_map::Entry::Occupied(mut slot)
                if slot.get().lane() == values.lane() =>
            {
                Some(slot.insert(values))
            }
            std::collections::btree_map::Entry::Occupied(_) => None,
        }
    }

    /// 某个编号的 `f32` 内容；字节缓冲返回 `None`。
    pub fn get(&self, id: u32) -> Option<&[f32]> {
        self.values.get(&id).and_then(Buf::f32)
    }

    /// 某个编号的打包字节内容；浮点缓冲返回 `None`。
    pub fn get_u8(&self, id: u32) -> Option<&[u8]> {
        self.values.get(&id).and_then(Buf::u8)
    }

    /// 某个编号绑定的缓冲（只读）。
    pub fn buf(&self, id: u32) -> Option<&Buf> {
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
#[derive(Default)]
struct Taken {
    ids: Vec<u32>,
    values: Vec<Buf>,
}

impl Taken {
    /// 按节点角色取只读视图。
    fn views(&self) -> Vec<View<'_>> {
        self.values.iter().map(Buf::view).collect()
    }

    /// 按节点角色取可写视图。
    fn views_mut(&mut self) -> Vec<ViewMut<'_>> {
        self.values.iter_mut().map(Buf::view_mut).collect()
    }
}

/// CPU 执行器：把图提交到本地队列，并在 [`sync`](Executor::sync) 时按图序**真的**跑内核。
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
            return Err(invalid(
                "engine::CpuExecutor::register",
                format!(
                    "内核 `{}` 已登记：重名会让「哪一份在跑」变成一个运行期问题",
                    program.name()
                ),
            ));
        }
        self.programs.push(program);
        Ok(())
    }

    /// 绑定一段配置操作数；返回被替换掉的旧块（lane 不符返回 `None`）。
    pub fn bind(&mut self, id: u32, values: Buf) -> Option<Buf> {
        self.bindings.bind(id, values)
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
        let idx = self.position(node.name())?;
        self.check_operands(node)?;
        let mut inputs = self.take(node.inputs());
        let mut outputs = self.take(node.outputs());
        let input_views = inputs.views();
        let mut output_views = outputs.views_mut();
        let result = self.programs[idx].run(node.launch(), &input_views, &mut output_views);
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
            .ok_or_else(|| invalid("engine::CpuExecutor::run", format!("内核 `{name}` 未登记")))
    }

    /// 节点自己的操作数约束：每个编号都要已绑定，且输入与输出不能是同一个编号。
    fn check_operands(&self, node: &Node) -> Result<()> {
        check_operands(
            node,
            |id| self.bindings.buf(id).is_none(),
            "engine::CpuExecutor::run",
        )
    }

    /// 把一组缓冲从操作数表里借出来；缺一个就整体放回（不留半个状态）。
    fn take(&mut self, operands: &[Operand]) -> Taken {
        let mut ids = Vec::with_capacity(operands.len());
        let mut values = Vec::with_capacity(operands.len());
        for operand in operands {
            match self.bindings.values.remove(&operand.id()) {
                Some(value) => {
                    ids.push(operand.id());
                    values.push(value);
                }
                None => if let Some(taken_at) = ids.iter().position(|&id| id == operand.id()) { values.push(values[taken_at].clone()) } else {
                    self.restore(Taken { ids, values });
                    return Taken {
                        ids: Vec::new(),
                        values: Vec::new(),
                    };
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

    fn bind(&mut self, id: u32, values: Buf) -> Option<Buf> {
        CpuExecutor::bind(self, id, values)
    }

    fn fetch(&self, id: u32) -> Option<&Buf> {
        self.bindings.buf(id)
    }

    /// 排这一条流上已提交的节点：按入队顺序执行，成功一条才出队一条。
    fn sync(&mut self, stream: Stream) -> Result<()> {
        let mut idx = 0;
        while idx < self.queue.len() {
            if self.queue[idx].stream() != stream {
                idx += 1;
                continue;
            }
            let node = self.queue[idx].node().clone();
            self.run(&node)?;
            self.queue.remove(idx);
        }
        Ok(())
    }
}
