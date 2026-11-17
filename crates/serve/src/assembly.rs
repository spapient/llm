//! 装配管线：把 `nn` 层栈与驻留参数表接成 engine [`Graph`]，再由 [`CpuModel`](crate::model_host::CpuModel) 宿主执行。
//!
//! 内核适配表留在 [`kernels`](crate::kernels)；层计划与族别分流在 [`plan`](crate::plan)，
//! 注意力块子节点接线在 [`block`](crate::block)，本模块只收口图、台账与编号分配。前向取数与
//! K/V 读写在 [`model_host`](crate::model_host)，不认识算子内核算式。

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Executor, Graph, Launch, Node, Stream};
use nn::{Contract, Linear, Stack};

pub(crate) use crate::block::BlockBinding;
pub(crate) use crate::descriptors::BindingDesc;
pub(crate) use crate::params::Params;
pub(crate) use crate::plan::{LayerKind, LayerPlan};

use crate::kernels::SITE;

/// 编号空间分区（与 [`engine::exec::Bindings`](../engine/exec) 的 u32 编号一一对应）：
///
/// - [`INPUT_SLOT`]：首层输入（提示词）占位编号，由 [`chain`](chain) 的 `prev` 初始绑定；
/// - [`ACTIVATION_BASE`] .. 权重基址之前：逐层激活输出编号；
/// - [`WEIGHT_BASE`] 及之后：驻留权重整块编号。
///
/// 两段基址分开登记，避免逐层复用的激活编号与驻留权重编号撞上。
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
}

impl Assembly {
    /// 把当前位置写进全部 rope 节点的标量槽；提交前由宿主按当前步回填，图结构不改。
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

    /// 取下一个激活编号；溢出归内部错误。
    pub(crate) fn activation(&mut self) -> Result<u32> {
        let id = self.activation;
        self.activation = id
            .checked_add(1)
            .ok_or_else(|| Error::new(Code::Internal, "激活编号超出 u32 编号空间").with(SITE))?;
        Ok(id)
    }

    /// 取下一个权重编号；溢出归内部错误。
    pub(crate) fn weight(&mut self) -> Result<u32> {
        let id = self.weight;
        self.weight = id
            .checked_add(1)
            .ok_or_else(|| Error::new(Code::Internal, "权重编号超出 u32 编号空间").with(SITE))?;
        Ok(id)
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
}

impl<'a> NodeSink<'a> {
    /// 一层接进图：输入 0 = 主链上一层输出；逐元素二元的旁路输入 1（若有）；投影/归一化/查表
    /// 再加驻留权重编号，一元点wise/二元/位置编码只有各自的输入路数。
    fn push_node(&mut self, prev: u32, plan: &LayerPlan, bypass: Option<u32>) -> Result<u32> {
        let output_id = self.slots.activation()?;
        let mut node = Node::new(
            plan.operator.clone(),
            Launch::new([1, 1, 1], [plan.out_features as u32, 1, 1]),
        );
        node.bind_input(prev);
        if let Some(bypass) = bypass {
            node.bind_input(bypass);
        }
        if let Some(entry) = &plan.entry {
            let weight_id = self.slots.weight()?;
            node.bind_input(weight_id);
            self.weights.push((weight_id, entry.clone()));
            self.descriptors.push(BindingDesc {
                id: weight_id,
                entry: Some(entry.clone()),
                shape: plan.weight_shape.clone(),
            });
        }
        node.bind_output(output_id);
        self.graph.push(node);
        self.activations.push((output_id, plan.out_features));
        self.descriptors.push(BindingDesc {
            id: output_id,
            entry: None,
            shape: vec![plan.out_features],
        });
        if plan.kind == LayerKind::Rope {
            self.rope.push(output_id);
        }
        Ok(output_id)
    }

    /// 单个投影子节点：输入向量 + 驻留权重整块 `[out, in]` → 输出向量；块内 q/k/v/o 复用。
    pub(crate) fn push_projection(
        &mut self,
        input: u32,
        linear: &Linear,
        entry: &str,
        operator: &str,
    ) -> Result<u32> {
        let output_id = self.slots.activation()?;
        let weight_id = self.slots.weight()?;
        let mut node = Node::new(
            operator.to_string(),
            Launch::new([1, 1, 1], [linear.out_features() as u32, 1, 1]),
        );
        node.bind_input(input);
        node.bind_input(weight_id);
        node.bind_output(output_id);
        self.graph.push(node);
        self.weights.push((weight_id, entry.to_string()));
        self.descriptors.push(BindingDesc {
            id: weight_id,
            entry: Some(entry.to_string()),
            shape: vec![linear.out_features(), linear.in_features()],
        });
        self.activations.push((output_id, linear.out_features()));
        self.descriptors.push(BindingDesc {
            id: output_id,
            entry: None,
            shape: vec![linear.out_features()],
        });
        Ok(output_id)
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
    let bypass = plan
        .kind
        .needs_bypass()
        .then(|| bypass_input(layer.name(), plan.out_features, held))
        .transpose()?;
    let next = sink.push_node(prev, &plan, bypass)?;
    Ok((next, plan.in_features, plan.out_features, plan.kind))
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
///
/// 族别（投影 / 查表 / 一元点wise / 归一化 / 逐元素二元 / 位置编码 / 注意力块）决定操作数路数与
/// 权重形态；形状、宽度、权重缺失都在装配期报出，不带进执行期。量化族、归约按 [`Code::Unsupported`]
/// 拒绝，不静默跳过。
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
    let mut slots = Slots::new();
    let mut sink = NodeSink {
        graph: &mut graph,
        weights: &mut weights,
        activations: &mut activations,
        descriptors: &mut descriptors,
        blocks: &mut blocks,
        rope: &mut rope,
        slots: &mut slots,
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
    };
    assembly.check_descriptors(params)?;
    Ok(assembly)
}

/// 把驻留权重整块绑进执行器编号表。
pub(crate) fn bind_weights(
    executor: &mut CpuExecutor,
    assembly: &Assembly,
    params: &Params,
) -> Result<()> {
    for (id, entry) in &assembly.weights {
        let block = params.get(entry).ok_or_else(|| {
            Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE)
        })?;
        executor.bind(*id, block.to_vec());
    }
    Ok(())
}

/// 预绑输入与各逐层激活的零缓冲：执行时只覆写，不在图运行期分配。
pub(crate) fn bind_activations(executor: &mut CpuExecutor, assembly: &Assembly) {
    executor.bind(assembly.input, vec![0.0; assembly.input_len]);
    for (id, len) in &assembly.activations {
        executor.bind(*id, vec![0.0; *len]);
    }
}

/// 提交图、同步执行并取回末层激活缓冲。
pub(crate) fn submit_and_fetch(
    executor: &mut CpuExecutor,
    assembly: &Assembly,
) -> Result<Vec<f32>> {
    executor.submit(&assembly.graph, Stream::new(0))?;
    executor.sync(Stream::new(0))?;
    let output = executor
        .get(assembly.output)
        .ok_or_else(|| Error::new(Code::Internal, "输出缓冲已预绑，取回失败").with(SITE))?;
    Ok(output.to_vec())
}
