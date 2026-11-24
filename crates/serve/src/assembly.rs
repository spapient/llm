//! 装配管线：把 `nn` 层栈与驻留参数表接成 engine [`Graph`]，再由 [`CpuModel`](crate::model::CpuModel) 宿主执行。

use diagnostic::{Code, Error, Result};
use engine::exec::{Buf, Executor, Graph, Launch, Node, Stream};
use nn::{Contract, Stack};

pub(crate) use crate::block::BlockBinding;
pub(crate) use crate::descriptors::BindingDesc;
pub(crate) use crate::params::{Block, Params};
pub(crate) use crate::plan::{LayerKind, LayerPlan};

use crate::kernels::SITE;

/// 编号空间分区（与 [`engine::exec::Bindings`](../engine/exec) 的 u32 编号一一对应）：
const INPUT_SLOT: u32 = 0;

/// 激活输出编号起始基址；输入占位占 [`INPUT_SLOT`]。
const ACTIVATION_BASE: u32 = 1;

/// 权重编号起始基址；激活段 `1..WEIGHT_BASE` 留给逐层输出。
const WEIGHT_BASE: u32 = 10_000;

/// 装配结果：图、输入/输出编号、权重编号 ↔ 条目名、各激活缓冲长度、编号 ↔ 形状描述表。
#[derive(Debug)]
pub(crate) struct Assembly {
    pub(crate) graph: Graph,
    pub(crate) input: u32,
    pub(crate) output: u32,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    /// 首层族别：决定一步前向的输入是 token 索引还是隐向量。
    pub(crate) input_kind: LayerKind,
    pub(crate) weights: Vec<(u32, String)>,
    pub(crate) activations: Vec<(u32, usize)>,
    /// 编号 ↔ 形状/条目名描述表（D10 / ADR 0044）。
    pub(crate) descriptors: Vec<BindingDesc>,
    /// 注意力块接线记录：运行时按它装载历史 K/V、取回当前步 K/V 写回页行。
    pub(crate) blocks: Vec<BlockBinding>,
    /// rope 节点输出槽编号：宿主在每步提交前把当前位置写进这些节点的标量槽。
    pub(crate) rope: Vec<u32>,
    /// 批位置表缓冲编号（ADR 0043 §5）：全模型共享，批形态每步覆写；无 rope 块时 `None`。
    pub(crate) rope_pos_table: Option<u32>,
    /// attention ragged 段长表缓冲编号（ADR 0046 §8）：全模型共享，批形态每步覆写；无块时 `None`。
    pub(crate) hist_len_table: Option<u32>,
    /// 条件记忆层接线记录：宿主每步按它覆写槽表与门控缓冲，再加回主链。
    pub(crate) memories: Vec<MemoryBinding>,
}

/// 一条条件记忆层的运行时绑定：该层专有的槽表与门控缓冲编号，加装配期定下的四元几何。
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryBinding {
    /// 槽表缓冲编号：每行一个槽号，每步覆写。
    pub(crate) slots: u32,
    /// 门控缓冲编号：逐元素 `γ`，不入表的行为 0，每步覆写。
    pub(crate) gate: u32,
    /// 装配期定下的阶 / 种子 / 槽数 / 门控。
    pub(crate) spec: crate::plan::MemorySpec,
    /// 每行宽度（门控缓冲长度 `= 行数 × width`）。
    pub(crate) width: usize,
}

impl Assembly {
    /// 把当前位置写进全部 rope 节点的标量槽；提交前由宿主按当前步回填，图结构不改。
    /// 批形态逐序列位置改绑位置表输入缓冲（ADR 0043 §5），不再回填标量槽。
    pub(crate) fn set_rope_positions(&mut self, position: u32) {
        for &id in &self.rope {
            let node = self.graph.nodes_mut().iter_mut().find(|node| {
                node.outputs()
                    .first()
                    .is_some_and(|operand| operand.id() == id)
            });
            if let Some(node) = node {
                node.scalars_mut().set_rope_position(position);
            }
        }
    }
}

impl Assembly {
    /// 描述表（只读）：编号 ↔ 条目名与声明形状。仅装配自检与测试读取。
    #[cfg(test)]
    pub(crate) fn descriptors(&self) -> &[BindingDesc] {
        &self.descriptors
    }

    /// 装配期自洽：描述表编号与权重/激活台账一一对应，形状积（权重）或单轴长度（激活）相符。
    /// 描述表是装配侧事实（D10 / ADR 0044）；引擎不读这张表，也不解释编号含义。
    fn check_descriptors(&self, params: &Params) -> Result<()> {
        crate::descriptors::check_all(&self.descriptors, params, &self.activations)
    }
}

/// 编号分配器：激活段与权重段各自顺序取下一个编号，取代按层索引推导。
pub(crate) struct Slots {
    activation: u32,
    weight: u32,
}

impl Slots {
    fn new() -> Self {
        Self {
            activation: ACTIVATION_BASE,
            weight: WEIGHT_BASE,
        }
    }

    /// 取走一段编号的下一号；溢出归内部错误。`what` 用于诊断区分激活/权重段。
    fn next(slot: &mut u32, what: &str) -> Result<u32> {
        let id = *slot;
        *slot = id.checked_add(1).ok_or_else(|| {
            Error::new(Code::Internal, format!("{what}编号超出 u32 编号空间")).with(SITE)
        })?;
        Ok(id)
    }

    /// 取下一个激活编号；溢出归内部错误。
    pub(crate) fn activation(&mut self) -> Result<u32> {
        Self::next(&mut self.activation, "激活")
    }

    /// 取下一个权重编号；溢出归内部错误。
    pub(crate) fn weight(&mut self) -> Result<u32> {
        Self::next(&mut self.weight, "权重")
    }
}

/// 图接线收口：图、权重/激活台账与编号分配器一起传递，避免逐函数串四五个可变引用。
pub(crate) struct NodeSink<'a> {
    pub(crate) graph: &'a mut Graph,
    pub(crate) weights: &'a mut Vec<(u32, String)>,
    pub(crate) activations: &'a mut Vec<(u32, usize)>,
    pub(crate) descriptors: &'a mut Vec<BindingDesc>,
    /// 注意力块接线记录：运行时按它装载历史 K/V、取回当前步 K/V 写回页行。
    pub(crate) blocks: &'a mut Vec<BlockBinding>,
    pub(crate) slots: &'a mut Slots,
    /// rope 节点输出槽编号清单：位置标量槽回填按它走。
    pub(crate) rope: &'a mut Vec<u32>,
    /// 批位置表编号：遇到 rope 块时懒分配，全模型共享一张。
    pub(crate) rope_pos_table: &'a mut Option<u32>,
    /// attention 段长表编号：遇到块时懒分配，全模型共享一张。
    pub(crate) hist_len_table: &'a mut Option<u32>,
    /// 条件记忆层接线记录：逐层追加，宿主每步按它回填槽表与门控。
    pub(crate) memories: &'a mut Vec<MemoryBinding>,
}

impl<'a> NodeSink<'a> {
    /// 一层接进图：输入 0 = 主链上一层输出；逐元素二元的旁路输入 1（若有）；投影/归一化/查表
    /// 再加驻留权重编号，一元点wise/二元/位置编码只有各自的输入路数。
    pub(crate) fn push_node(
        &mut self,
        prev: u32,
        plan: &LayerPlan,
        bypass: Option<u32>,
    ) -> Result<u32> {
        let output_id = self.slots.activation()?;
        // 投影/查表第二维登记输入宽（行批切段用）；等长族第二维占位 0。
        let width_in = matches!(plan.kind, LayerKind::Projection | LayerKind::Embedding)
            .then_some(plan.in_features)
            .unwrap_or(0);
        let mut node = Node::new(
            plan.operator.clone(),
            Launch::new([1, 1, 1], [plan.out_features as u32, width_in as u32, 1]),
        );
        node.bind_input(prev);
        if let Some(bypass) = bypass {
            node.bind_input(bypass);
        }
        self.bind_entry(&mut node, plan)?;
        node.bind_output(output_id);
        self.graph.push(node);
        self.register_activation(output_id, plan.out_features);
        if plan.kind == LayerKind::Rope {
            self.rope.push(output_id);
        }
        Ok(output_id)
    }

    /// 发一个定输入节点：分配输出编号、建节点、按序绑输入、绑输出、入图、登记激活长度。
    /// 描述表 / rope 槽由调用方在返回后按需补登记；块内 rope / 注意力 / 残差共用这一份，
    /// 带权重绑定的层走 [`Self::push_node`] + [`Self::bind_weight`]。
    pub(crate) fn emit(
        &mut self,
        operator: &str,
        launch: Launch,
        inputs: &[u32],
        out_len: usize,
    ) -> Result<u32> {
        let output = self.slots.activation()?;
        let mut node = Node::new(operator.to_string(), launch);
        for &input in inputs {
            node.bind_input(input);
        }
        node.bind_output(output);
        self.graph.push(node);
        self.activations.push((output, out_len));
        Ok(output)
    }

    /// 登记一个等长激活：台账长度 + 描述表单轴形状，一处定义。
    fn register_activation(&mut self, id: u32, width: usize) {
        self.activations.push((id, width));
        self.descriptors.push(BindingDesc {
            id,
            entry: None,
            shape: vec![width],
        });
    }

    /// 发一个逐元素双输入节点（两路等长），返回输出编号；描述表按单轴宽度登记。
    fn emit_pair(&mut self, operator: &str, lhs: u32, rhs: u32, width: usize) -> Result<u32> {
        let output = self.emit(
            operator,
            Launch::new([1, 1, 1], [width as u32, 0, 1]),
            &[lhs, rhs],
            width,
        )?;
        self.descriptors.push(BindingDesc {
            id: output,
            entry: None,
            shape: vec![width],
        });
        Ok(output)
    }

    /// 条件记忆层接三节点：查表 `[槽表, 表] → [width]`、逐元素门控缩放、加回主链。
    pub(crate) fn push_memory(
        &mut self,
        prev: u32,
        plan: &LayerPlan,
        spec: crate::plan::MemorySpec,
    ) -> Result<u32> {
        let entry = Self::memory_entry(plan)?;
        let tables = self.alloc_memory_tables()?;
        let output = self.memory_chain(prev, plan, entry, tables)?;
        self.memories.push(MemoryBinding {
            slots: tables.0,
            gate: tables.1,
            spec,
            width: plan.out_features,
        });
        Ok(output)
    }

    /// 条件记忆层的驻留表条目名：装配期已校验必有。
    fn memory_entry(plan: &LayerPlan) -> Result<&str> {
        plan.entry
            .as_deref()
            .ok_or_else(|| Error::new(Code::Internal, "条件记忆层必有驻留表条目").with(SITE))
    }

    /// 该层专有的两张批表：槽表与门控缓冲，宿主每步覆写。
    fn alloc_memory_tables(&mut self) -> Result<(u32, u32)> {
        Ok((self.alloc_table()?, self.alloc_table()?))
    }

    /// 三节点接线：查表 → 门控缩放 → 加回主链，返回末节点编号。
    fn memory_chain(
        &mut self,
        prev: u32,
        plan: &LayerPlan,
        entry: &str,
        tables: (u32, u32),
    ) -> Result<u32> {
        let (table_slots, gate) = tables;
        let width = plan.out_features;
        let gathered = self.emit_gather(plan, entry, table_slots)?;
        let scaled = self.emit_pair(nn::MemoryBlock::SCALE, gathered, gate, width)?;
        self.emit_pair(nn::MemoryBlock::RESIDUAL, prev, scaled, width)
    }

    /// 查表节点：槽表 + 驻留表 → `[width]`，建节点、绑权重、登记激活一处完成。
    fn emit_gather(&mut self, plan: &LayerPlan, entry: &str, slots: u32) -> Result<u32> {
        let width = plan.out_features;
        let gathered = self.slots.activation()?;
        let mut node = Node::new(
            nn::MemoryBlock::EMBED.to_string(),
            Launch::new([1, 1, 1], [width as u32, 1, 1]),
        );
        node.bind_input(slots);
        self.bind_weight(&mut node, entry, &plan.weight_shape, None, &[])?;
        node.bind_output(gathered);
        self.graph.push(node);
        self.register_activation(gathered, width);
        Ok(gathered)
    }

    /// 驻留权重 + 可选标度的输入与台账：打包字节 / 标度各占一个编号，都挂进节点输入
    /// （`[x, packed, scales]`），台账按 `(编号, 条目名)` 登记、描述表按形状登记。投影与
    /// 块内 q/k/v/o 共用这一份绑定，不各写一遍。
    fn bind_weight(
        &mut self,
        node: &mut Node,
        entry: &str,
        weight_shape: &[usize],
        scale_entry: Option<&str>,
        scale_shape: &[usize],
    ) -> Result<()> {
        let weight_id = self.slots.weight()?;
        node.bind_input(weight_id);
        self.weights.push((weight_id, entry.to_string()));
        self.descriptors.push(BindingDesc {
            id: weight_id,
            entry: Some(entry.to_string()),
            shape: weight_shape.to_vec(),
        });
        if let Some(scale_entry) = scale_entry {
            let scale_id = self.slots.weight()?;
            node.bind_input(scale_id);
            self.weights.push((scale_id, scale_entry.to_string()));
            self.descriptors.push(BindingDesc {
                id: scale_id,
                entry: Some(scale_entry.to_string()),
                shape: scale_shape.to_vec(),
            });
        }
        Ok(())
    }

    /// 投影/归一化/查表层的驻留权重输入与台账；一元点wise/二元/位置编码没有条目分支。
    fn bind_entry(&mut self, node: &mut Node, plan: &LayerPlan) -> Result<()> {
        let Some(entry) = &plan.entry else {
            return Ok(());
        };
        self.bind_weight(
            node,
            entry,
            &plan.weight_shape,
            plan.scale_entry.as_deref(),
            &plan.scales_shape,
        )
    }

    /// 取号登记一张运行时覆写的批表缓冲：长度 0 占位，台账与描述表对齐历史块同口径。
    pub(crate) fn alloc_table(&mut self) -> Result<u32> {
        let id = self.slots.activation()?;
        self.activations.push((id, 0));
        self.descriptors.push(BindingDesc {
            id,
            entry: None,
            shape: vec![0],
        });
        Ok(id)
    }

    /// 懒分配全模型共享的 rope 批位置表编号：已取号复用，否则登记新号。
    pub(crate) fn ensure_rope_pos_table(&mut self) -> Result<u32> {
        match *self.rope_pos_table {
            Some(id) => Ok(id),
            None => {
                let id = self.alloc_table()?;
                *self.rope_pos_table = Some(id);
                Ok(id)
            }
        }
    }

    /// 懒分配全模型共享的 attention 段长表编号：已取号复用，否则登记新号。
    pub(crate) fn ensure_hist_len_table(&mut self) -> Result<u32> {
        match *self.hist_len_table {
            Some(id) => Ok(id),
            None => {
                let id = self.alloc_table()?;
                *self.hist_len_table = Some(id);
                Ok(id)
            }
        }
    }

    /// 一次取齐两张共享批表编号：rope 位置表与 attention 段长表。
    pub(crate) fn ensure_tables(&mut self) -> Result<(u32, u32)> {
        let pos = self.ensure_rope_pos_table()?;
        let len = self.ensure_hist_len_table()?;
        Ok((pos, len))
    }
}

/// 逐层串联：返回首层输入宽度、末层输出宽度与末层激活编号；注意力块拆成子节点接进图。
fn chain(
    layers: &[Box<dyn Contract>],
    params: &Params,
    sink: &mut NodeSink<'_>,
) -> Result<(usize, usize, u32, LayerKind)> {
    let mut prev = INPUT_SLOT;
    let mut prev_out: Option<usize> = None;
    let mut input_len = 0;
    let mut last_out = 0;
    let mut input_kind = LayerKind::Projection;
    // 旁路持有槽：逐元素二元可借用的残差/门控激活；非二元层输出刷新它，二元层不覆盖。
    let mut held: Option<(u32, usize)> = None;

    for layer in layers.iter() {
        let (next, in_width, out_width, kind) =
            chain_step(sink, prev, prev_out, held, layer.as_ref(), params)?;
        if prev_out.is_none() {
            input_len = in_width;
            input_kind = kind;
        }
        if kind != LayerKind::Binary {
            held = Some((next, out_width));
        }
        prev = next;
        prev_out = Some(out_width);
        last_out = out_width;
    }
    Ok((input_len, last_out, prev, input_kind))
}

/// 接一层：注意力块拆子节点，其余层出单节点；返回输出编号、入宽、出宽与族别。
fn chain_step(
    sink: &mut NodeSink<'_>,
    prev: u32,
    prev_out: Option<usize>,
    held: Option<(u32, usize)>,
    layer: &dyn Contract,
    params: &Params,
) -> Result<(u32, usize, usize, LayerKind)> {
    match layer.attention_block() {
        Some(block) => crate::block::advance_block(sink, prev, prev_out, block, params),
        None => advance_layer(sink, prev, prev_out, held, layer, params),
    }
}

/// 普通族别层：出计划后接单节点；返回输出编号、入宽、出宽与族别。
fn advance_layer(
    sink: &mut NodeSink<'_>,
    prev: u32,
    prev_out: Option<usize>,
    held: Option<(u32, usize)>,
    layer: &dyn Contract,
    params: &Params,
) -> Result<(u32, usize, usize, LayerKind)> {
    let plan = crate::plan::layer_plan(layer, prev_out, params)?;
    let next = attach(sink, prev, &plan, held, layer.name())?;
    Ok((next, plan.in_features, plan.out_features, plan.kind))
}

/// 按计划接线：记忆层走三节点旁路，其余层出单节点。
fn attach(
    sink: &mut NodeSink<'_>,
    prev: u32,
    plan: &LayerPlan,
    held: Option<(u32, usize)>,
    name: &str,
) -> Result<u32> {
    let Some(spec) = plan.memory else {
        return push_single(sink, prev, plan, held, name);
    };
    // 记忆层是三节点旁路：不走单节点通路，也不占用与二元层共享的 held 衔接。
    sink.push_memory(prev, plan, spec)
}

/// 单节点通路：二元层先取旁路第二输入槽，再出节点。
fn push_single(
    sink: &mut NodeSink<'_>,
    prev: u32,
    plan: &LayerPlan,
    held: Option<(u32, usize)>,
    name: &str,
) -> Result<u32> {
    let bypass = plan
        .kind
        .needs_bypass()
        .then(|| bypass_input(name, plan.out_features, held))
        .transpose()?;
    sink.push_node(prev, plan, bypass)
}

/// 取二元层的旁路第二输入槽；无持有激活或宽度不符时返回 [`Code::Invalid`]。
fn bypass_input(name: &str, width: usize, held: Option<(u32, usize)>) -> Result<u32> {
    let Some((slot, held_width)) = held else {
        return Err(Error::new(
            Code::Invalid,
            format!("层 `{name}` 是逐元素二元层，但前面没有可借用的旁路激活"),
        )
        .with("serve::assembly::bypass_input"));
    };
    if held_width != width {
        return Err(Error::new(
            Code::Invalid,
            format!("层 `{name}` 旁路宽度 {held_width} 与主链宽度 {width} 不符"),
        )
        .with("serve::assembly::bypass_input"));
    }
    Ok(slot)
}

/// 把层栈装配成图：逐层接线，上一层输出即下一层输入。
pub(crate) fn assemble(stack: &Stack, params: &Params) -> Result<Assembly> {
    let layers = stack.layers();
    if layers.is_empty() {
        return Err(Error::new(Code::Invalid, "模型栈为空：没有可装配的层").with(SITE));
    }

    let mut graph = Graph::new();
    let mut activations = Vec::new();
    let mut weights: Vec<(u32, String)> = Vec::new();
    let mut descriptors = Vec::new();
    let mut blocks = Vec::new();
    let mut rope = Vec::new();
    let mut rope_pos_table = None;
    let mut hist_len_table = None;
    let mut memories = Vec::new();
    let mut slots = Slots::new();
    let mut sink = NodeSink {
        graph: &mut graph,
        weights: &mut weights,
        activations: &mut activations,
        descriptors: &mut descriptors,
        blocks: &mut blocks,
        rope: &mut rope,
        slots: &mut slots,
        rope_pos_table: &mut rope_pos_table,
        hist_len_table: &mut hist_len_table,
        memories: &mut memories,
    };
    let (input_len, output_len, output, input_kind) = chain(layers, params, &mut sink)?;

    let assembly = Assembly {
        graph,
        input: INPUT_SLOT,
        output,
        input_len,
        output_len,
        input_kind,
        weights,
        activations,
        descriptors,
        blocks,
        rope,
        rope_pos_table,
        hist_len_table,
        memories,
    };
    assembly.check_descriptors(params)?;
    Ok(assembly)
}

/// 把驻留权重整块绑进执行器编号表：稠密块绑 f32，量化打包块绑 u8。
pub(crate) fn bind_weights<E: Executor>(
    executor: &mut E,
    assembly: &Assembly,
    params: &Params,
) -> Result<()> {
    for (id, entry) in &assembly.weights {
        let block = params.get(entry).ok_or_else(|| {
            Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE)
        })?;
        let buf = match block {
            Block::Dense(values) => Buf::F32(values.to_vec()),
            Block::Packed(bytes) => Buf::U8(bytes.to_vec()),
        };
        executor.bind(*id, buf);
    }
    Ok(())
}

/// 预绑输入与各逐层激活的零缓冲：执行时只覆写，不在图运行期分配。
pub(crate) fn bind_activations<E: Executor>(executor: &mut E, assembly: &Assembly) {
    executor.bind(
        assembly.input,
        Buf::F32(vec![0.0; assembly.input_len]),
    );
    for (id, len) in &assembly.activations {
        executor.bind(*id, Buf::F32(vec![0.0; *len]));
    }
}

/// 提交图、同步执行并取回末层激活缓冲。
pub(crate) fn submit_and_fetch<E: Executor>(
    executor: &mut E,
    assembly: &Assembly,
) -> Result<Vec<f32>> {
    executor.submit(&assembly.graph, Stream::new(0))?;
    executor.sync(Stream::new(0))?;
    let output = executor
        .fetch(assembly.output)
        .and_then(Buf::f32)
        .ok_or_else(|| Error::new(Code::Internal, "输出缓冲已预绑，取回失败").with(SITE))?;
    Ok(output.to_vec())
}
