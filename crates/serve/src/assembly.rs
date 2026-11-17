//! 装配管线：把 `nn` 层栈与驻留参数表接成 engine [`Graph`]，再由 [`CpuModel`](crate::model_host::CpuModel) 宿主执行。
//!
//! 内核适配表留在 [`kernels`](crate::kernels)；本模块只负责层族别分流、宽度链与权重形状校验、图接线，
//! 前向取数与 K/V 读写在 [`model_host`](crate::model_host)，不认识算子内核算式。

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Executor, Graph, Launch, Node, Stream};
use nn::{
    AttentionAlgo, AttentionBlock, Contract, Embedding, Linear, Shape, Stack, WEIGHT_SUFFIX,
    WeightFormat,
};

pub(crate) use crate::descriptors::BindingDesc;
pub(crate) use crate::params::Params;

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

/// 一个注意力块接线记录：当前步 K/V 取回拼页行；历史 K/V 缓冲每步重绑；旁路即块输入编号。
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockBinding {
    pub(crate) k: u32,
    pub(crate) v: u32,
    pub(crate) hist_k: u32,
    pub(crate) hist_v: u32,
    /// K/V 宽度（单头口径 = 头宽），页行按它切 K/V 半行。
    pub(crate) kv_width: usize,
}

/// 编号分配器：激活段与权重段各自顺序取下一个编号，取代按层索引推导。
struct Slots {
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
    fn activation(&mut self) -> Result<u32> {
        let id = self.activation;
        self.activation = id
            .checked_add(1)
            .ok_or_else(|| Error::new(Code::Internal, "激活编号超出 u32 编号空间").with(SITE))?;
        Ok(id)
    }

    /// 取下一个权重编号；溢出归内部错误。
    fn weight(&mut self) -> Result<u32> {
        let id = self.weight;
        self.weight = id
            .checked_add(1)
            .ok_or_else(|| Error::new(Code::Internal, "权重编号超出 u32 编号空间").with(SITE))?;
        Ok(id)
    }
}

/// 装配族：决定本层操作数怎么接线（输入路数、是否带驻留权重及权重形状）。
///
/// 族别按算子 arity 与权重形态归类，是接线方式的机械分类，不是后端厂商分支。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    /// 投影：输入向量 + 驻留权重整块 `[out, in]` → 输出向量。
    Projection,
    /// 一元点wise：单路输入 → 等长输出，无驻留权重。
    Unary,
    /// 归一化：输入向量 + 驻留 gamma 向量 `[width]` → 等长输出。
    Norm,
    /// 查表：索引序列 + 驻留权重表 `[vocab, dim]` → 行向量块。
    Embedding,
}

impl LayerKind {
    /// 链上本层与上一层输出对齐的宽度：投影/查表对齐输入宽，等长族（一元/归一化）对齐输出宽。
    fn chain_width(self, plan: &LayerPlan) -> usize {
        match self {
            Self::Projection | Self::Embedding => plan.in_features,
            Self::Unary | Self::Norm => plan.out_features,
        }
    }
}

/// 算子族别描述：一个算子名集合归到一个装配族。
///
/// 类型视图族（投影走 `Linear`、查表走 `Embedding`）不在这张表里——它们由类型化视图优先分流；
/// 这张表只收按算子名归类的等长族。新增族别在此加一行，plan 构造与链轴规则随 [`LayerKind`] 走。
struct OperatorFamily {
    kind: LayerKind,
    operators: &'static [&'static str],
}

/// 等长族算子归属表：编译期定长切片，检索即闭集查找。
const OPERATOR_FAMILIES: &[OperatorFamily] = &[
    OperatorFamily {
        kind: LayerKind::Unary,
        operators: &["gelu_f32", "scale_f32", "silu_f32", "softmax_f32"],
    },
    OperatorFamily {
        kind: LayerKind::Norm,
        operators: &["rmsnorm_f32"],
    },
];

/// 按算子名查装配族；未归属的算子返回 `None`（由分流层报不支持）。
fn family_of(operator: &str) -> Option<LayerKind> {
    OPERATOR_FAMILIES
        .iter()
        .find(|family| family.operators.contains(&operator))
        .map(|family| family.kind)
}

/// 一层装配所需的族别、算子名、权重条目名、几何宽度、驻留权重期望长度与声明形状。
struct LayerPlan {
    kind: LayerKind,
    operator: String,
    /// 驻留权重条目名；一元点wise无权重，为 `None`。
    entry: Option<String>,
    in_features: usize,
    out_features: usize,
    /// 驻留权重元素数；一元点wise无权重，为 `None`。
    weight_len: Option<usize>,
    /// 驻留权重声明形状；无权重为空表。
    weight_shape: Vec<usize>,
}

/// 投影层必须是 F32 稠密格式，且能推出已注册算子名；其余在此拒绝。
fn plan_operator(name: &str, linear: &Linear) -> Result<&'static str> {
    if linear.format() != WeightFormat::F32 {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "层 `{name}` 是 {}，P1 只接 F32 稠密投影：{}",
                linear.format(),
                linear.format().unpriced_reason()
            ),
        )
        .with(SITE));
    }
    linear.format().operator().ok_or_else(|| {
        Error::new(Code::Unsupported, format!("层 `{name}` 没有可装配的内核")).with(SITE)
    })
}

/// 取层绑定的算子名；空算子列表表示编码还没有内核，装配期拒绝。
fn first_operator(layer: &dyn Contract) -> Result<&str> {
    layer
        .spca()
        .ops()
        .first()
        .map(String::as_str)
        .ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                format!("层 `{}` 没有可装配的内核", layer.name()),
            )
            .with(SITE)
        })
}

/// 单轴层几何：输出形状必须是一条轴 `[width]`；标量（归约）与多轴形状不在隐藏主链装配。
fn axis_width(shape: &Shape, name: &str) -> Result<usize> {
    match shape.dims() {
        [width] => Ok(*width),
        other => Err(Error::new(
            Code::Invalid,
            format!("层 `{name}` 形状 `{other:?}` 不是单轴宽度（归约与多轴层留后续通路）"),
        )
        .with(SITE)),
    }
}

/// 层间宽度链：首层不限；投影输入对上一层输出，等长族（一元/归一化）输出对上一层输出。
fn check_chain(prev_out: Option<usize>, plan: &LayerPlan, name: &str) -> Result<()> {
    let Some(previous) = prev_out else {
        return Ok(());
    };
    let required = plan.kind.chain_width(plan);
    if previous != required {
        return Err(Error::new(
            Code::Invalid,
            format!("层 `{name}` 宽度 {required} 与上一层输出宽度 {previous} 不符"),
        )
        .with(SITE));
    }
    Ok(())
}

/// 核对驻留权重条目存在且长度符合族别；一元点wise无权重，直接通过。
///
/// 期望长度由族别在出计划时填好（投影 `in × out`、归一化 gamma `width`、查表 `vocab × dim`）。
fn check_weight(plan: &LayerPlan, params: &Params) -> Result<()> {
    let Some(expected) = plan.weight_len else {
        return Ok(());
    };
    let Some(entry) = &plan.entry else {
        return Ok(());
    };
    let Some(block) = params.get(entry) else {
        return Err(Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE));
    };
    if block.len() != expected {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "条目 `{entry}` 有 {} 个元素，而需要 {expected}",
                block.len()
            ),
        )
        .with(SITE));
    }
    Ok(())
}

/// 一层的装配校验：宽度链衔接 + 权重条目齐备且形状相符。
fn check_plan(
    prev_out: Option<usize>,
    plan: &LayerPlan,
    name: &str,
    params: &Params,
) -> Result<()> {
    check_chain(prev_out, plan, name)?;
    check_weight(plan, params)
}

/// 填一份层计划：族别几何由调用方给出，条目名、权重长度与声明形状随权重形态保持一致。
fn build_plan(
    kind: LayerKind,
    operator: String,
    entry: Option<String>,
    in_features: usize,
    out_features: usize,
    weight_len: Option<usize>,
    weight_shape: Vec<usize>,
) -> LayerPlan {
    LayerPlan {
        kind,
        operator,
        entry,
        in_features,
        out_features,
        weight_len,
        weight_shape,
    }
}

/// 驻留权重条目名：层名 + 权重后缀。
fn entry_name(layer: &dyn Contract) -> String {
    format!("{}{WEIGHT_SUFFIX}", layer.name())
}

/// 投影层装配计划：身份、F32 格式、算子名与 `[out, in]` 权重整块。
fn projection_plan(
    layer: &dyn Contract,
    linear: &Linear,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = plan_operator(layer.name(), linear)?;
    let plan = build_plan(
        LayerKind::Projection,
        operator.to_string(),
        Some(entry_name(layer)),
        linear.in_features(),
        linear.out_features(),
        Some(linear.in_features() * linear.out_features()),
        vec![linear.out_features(), linear.in_features()],
    );
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 等长族计划（一元点wise / 归一化）：宽度取自输出形状；归一化带 gamma `[width]`，一元点wise无权重。
fn width_plan(
    kind: LayerKind,
    layer: &dyn Contract,
    operator: &str,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let width = axis_width(layer.output(), layer.name())?;
    // 等长族里只有归一化带 gamma `[width]`；一元点wise无权重。
    let weight_len = (kind == LayerKind::Norm).then_some(width);
    let weight_shape = (kind == LayerKind::Norm)
        .then_some(vec![width])
        .unwrap_or_default();
    let plan = build_plan(
        kind,
        operator.to_string(),
        weight_len.map(|_| entry_name(layer)),
        width,
        width,
        weight_len,
        weight_shape,
    );
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 查表计划：索引序列 + 驻留权重表 `[vocab, dim]`；当前向量主链只接单 token decode 步。
fn embedding_plan(
    layer: &dyn Contract,
    emb: &Embedding,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = first_operator(layer)?;
    let tokens = emb.tokens();
    if tokens != 1 {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "层 `{}` 是 {tokens}-token 查表层；当前向量主链只接单 token decode 步，多 token 矩阵通路随连续批刀落地",
                layer.name()
            ),
        )
        .with(SITE));
    }
    let (vocab, dim) = (emb.vocab(), emb.dim());
    let plan = build_plan(
        LayerKind::Embedding,
        operator.to_string(),
        Some(entry_name(layer)),
        tokens,
        dim,
        Some(vocab * dim),
        vec![vocab, dim],
    );
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 一层的装配分流：类型视图（投影/查表）优先；都不是再按算子族别归类。二元残差、归约、
/// 位置编码、注意力在本通路不接线（残差旁路与 KV 通路随注意力刀落地）。
fn layer_plan(layer: &dyn Contract, prev_out: Option<usize>, params: &Params) -> Result<LayerPlan> {
    if let Some(plan) = view_plan(layer, prev_out, params)? {
        return Ok(plan);
    }
    operator_plan(layer, prev_out, params)
}

/// 类型视图层：能取到 `Linear` / `Embedding` 类型化视图的层出计划；都不是返回 `Ok(None)`。
fn view_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<Option<LayerPlan>> {
    if let Some(linear) = layer.linear() {
        let plan = projection_plan(layer, linear, prev_out, params)?;
        return Ok(Some(plan));
    }
    if let Some(emb) = layer.embedding() {
        let plan = embedding_plan(layer, emb, prev_out, params)?;
        return Ok(Some(plan));
    }
    Ok(None)
}

/// 算子族别层：按 [`OPERATOR_FAMILIES`] 归类等长族；其余族别报 [`Code::Unsupported`]。
fn operator_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = first_operator(layer)?;
    let Some(kind) = family_of(operator) else {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "层 `{}` 的算子 `{operator}` 不在当前装配族别（投影/查表/一元点wise/归一化）",
                layer.name()
            ),
        )
        .with(SITE));
    };
    width_plan(kind, layer, operator, prev_out, params)
}

/// 图接线收口：图、权重/激活台账与编号分配器一起传递，避免逐函数串四五个可变引用。
struct NodeSink<'a> {
    graph: &'a mut Graph,
    weights: &'a mut Vec<(u32, String)>,
    activations: &'a mut Vec<(u32, usize)>,
    descriptors: &'a mut Vec<BindingDesc>,
    slots: &'a mut Slots,
}

impl<'a> NodeSink<'a> {
    /// 一层接进图：绑定上一层输出；投影/归一化再加驻留权重编号，一元点wise只有单路输入。
    fn push_node(&mut self, prev: u32, plan: &LayerPlan) -> Result<u32> {
        let output_id = self.slots.activation()?;
        let mut node = Node::new(
            plan.operator.clone(),
            Launch::new([1, 1, 1], [plan.out_features as u32, 1, 1]),
        );
        node.bind_input(prev);
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
        Ok(output_id)
    }

    /// 单个投影子节点：输入向量 + 驻留权重整块 `[out, in]` → 输出向量；块内 q/k/v/o 复用。
    fn push_projection(
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

/// 块内投影条目名：`{块名}.{level}.weight`。
fn block_entry(block: &AttentionBlock, level: &str) -> String {
    format!("{}.{level}{WEIGHT_SUFFIX}", block.name())
}

/// decode 算子名映射：块内算法沿用 nn 绑定的 prefill 族别，装配侧切成对应 decode 变体。
fn decode_operator(algo: AttentionAlgo) -> &'static str {
    match algo {
        AttentionAlgo::Naive => "decode_attention_f32",
        AttentionAlgo::Flash => "decode_flash_attention_f32",
    }
}

/// 注意力块装配校验：单头几何、宽度链衔接、四个投影条目齐备且长度相符。
fn check_block(block: &AttentionBlock, prev_out: Option<usize>, params: &Params) -> Result<()> {
    check_head(block, prev_out)?;
    check_block_weights(block, params)
}

/// 单头几何与宽度链衔接：`kv_width == head_dim`；块输入对齐上一层输出宽度。
fn check_head(block: &AttentionBlock, prev_out: Option<usize>) -> Result<()> {
    let hidden = block.hidden();
    let kv = block.kv_width();
    let head = block.head_dim();
    if kv != head {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "注意力块 `{}` 是多头展平（kv_width {kv} ≠ head_dim {head}）；单头通路本刀落地，多头切片留后续",
                block.name()
            ),
        )
        .with(SITE));
    }
    if let Some(previous) = prev_out
        && previous != hidden
    {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "注意力块 `{}` 输入宽度 {hidden} 与上一层输出宽度 {previous} 不符",
                block.name()
            ),
        )
        .with(SITE));
    }
    Ok(())
}

/// 四个投影条目：条目齐备、元素数与 `in × out` 相符；算子格式校验统一在
/// [`block_operators`] 内走一次 [`plan_operator`]，这里不再重复拆格式。
fn check_block_weights(block: &AttentionBlock, params: &Params) -> Result<()> {
    let hidden = block.hidden();
    let kv = block.kv_width();
    let subs: [(&str, usize, usize); 4] = [
        ("q", hidden, kv),
        ("k", hidden, kv),
        ("v", hidden, kv),
        ("o", kv, hidden),
    ];
    for (level, in_width, out_width) in subs {
        let entry = block_entry(block, level);
        let Some(weight) = params.get(&entry) else {
            return Err(
                Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE),
            );
        };
        let expected = in_width * out_width;
        if weight.len() != expected {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "条目 `{entry}` 有 {} 个元素，而需要 {expected}",
                    weight.len()
                ),
            )
            .with(SITE));
        }
    }
    Ok(())
}

/// 块内四个投影的算子名：格式校验在 `plan_operator` 内一次完成。
struct ProjectionOps {
    q: &'static str,
    k: &'static str,
    v: &'static str,
    o: &'static str,
}

fn block_operators(block: &AttentionBlock) -> Result<ProjectionOps> {
    Ok(ProjectionOps {
        q: plan_operator(&format!("{}.q", block.name()), block.q())?,
        k: plan_operator(&format!("{}.k", block.name()), block.k())?,
        v: plan_operator(&format!("{}.v", block.name()), block.v())?,
        o: plan_operator(&format!("{}.o", block.name()), block.out())?,
    })
}

/// q/k/v 三路投影共享块输入扇出；返回三个投影输出编号。
fn wire_qkv(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    ops: &ProjectionOps,
) -> Result<(u32, u32, u32)> {
    let q = sink.push_projection(prev, block.q(), &block_entry(block, "q"), ops.q)?;
    let k = sink.push_projection(prev, block.k(), &block_entry(block, "k"), ops.k)?;
    let v = sink.push_projection(prev, block.v(), &block_entry(block, "v"), ops.v)?;
    Ok((q, k, v))
}

/// decode 注意力节点：q + 历史 K + 历史 V → 注意力输出；返回 `(hist_k, hist_v, attn)` 编号。
fn wire_attention(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    q_slot: u32,
) -> Result<(u32, u32, u32)> {
    let kv = block.kv_width();
    let hist_k = sink.slots.activation()?;
    let hist_v = sink.slots.activation()?;
    sink.activations.push((hist_k, 0));
    sink.activations.push((hist_v, 0));
    sink.descriptors.push(BindingDesc {
        id: hist_k,
        entry: None,
        shape: vec![0],
    });
    sink.descriptors.push(BindingDesc {
        id: hist_v,
        entry: None,
        shape: vec![0],
    });
    let attn_slot = sink.slots.activation()?;
    let mut attention = Node::new(
        decode_operator(block.algo()).to_string(),
        Launch::new([1, 1, 1], [kv as u32, 1, 1]),
    );
    attention.bind_input(q_slot);
    attention.bind_input(hist_k);
    attention.bind_input(hist_v);
    attention.bind_output(attn_slot);
    sink.graph.push(attention);
    sink.activations.push((attn_slot, kv));
    sink.descriptors.push(BindingDesc {
        id: attn_slot,
        entry: None,
        shape: vec![kv],
    });
    Ok((hist_k, hist_v, attn_slot))
}

/// o 投影回 hidden，再接残差旁路（o 输出 + 块输入）；返回块输出编号。
fn wire_output(
    sink: &mut NodeSink<'_>,
    block: &AttentionBlock,
    prev: u32,
    attn_slot: u32,
    ops: &ProjectionOps,
) -> Result<u32> {
    let hidden = block.hidden();
    let pre_residual =
        sink.push_projection(attn_slot, block.out(), &block_entry(block, "o"), ops.o)?;
    let out_slot = sink.slots.activation()?;
    let mut residual = Node::new(
        "residual_add_f32".to_string(),
        Launch::new([1, 1, 1], [hidden as u32, 1, 1]),
    );
    residual.bind_input(pre_residual);
    residual.bind_input(prev);
    residual.bind_output(out_slot);
    sink.graph.push(residual);
    sink.activations.push((out_slot, hidden));
    Ok(out_slot)
}

/// 把注意力块拆成子节点接进图；返回块输出激活编号。
fn push_block(
    sink: &mut NodeSink<'_>,
    blocks: &mut Vec<BlockBinding>,
    prev: u32,
    block: &AttentionBlock,
) -> Result<u32> {
    let ops = block_operators(block)?;
    let (q_slot, k_slot, v_slot) = wire_qkv(sink, block, prev, &ops)?;
    let (hist_k, hist_v, attn_slot) = wire_attention(sink, block, q_slot)?;
    let out_slot = wire_output(sink, block, prev, attn_slot, &ops)?;
    blocks.push(BlockBinding {
        k: k_slot,
        v: v_slot,
        hist_k,
        hist_v,
        kv_width: block.kv_width(),
    });
    Ok(out_slot)
}

/// 逐层串联：返回首层输入宽度、末层输出宽度与末层激活编号；注意力块拆成子节点接进图。
fn chain(
    layers: &[Box<dyn Contract>],
    params: &Params,
    graph: &mut Graph,
    weights: &mut Vec<(u32, String)>,
    activations: &mut Vec<(u32, usize)>,
    descriptors: &mut Vec<BindingDesc>,
    blocks: &mut Vec<BlockBinding>,
) -> Result<(usize, usize, u32, LayerKind)> {
    let mut slots = Slots::new();
    let mut sink = NodeSink {
        graph,
        weights,
        activations,
        descriptors,
        slots: &mut slots,
    };
    let mut prev = INPUT_SLOT;
    let mut prev_out: Option<usize> = None;
    let mut input_len = 0;
    let mut last_out = 0;
    let mut input_kind = LayerKind::Projection;

    for layer in layers.iter() {
        let (next, in_width, out_width, kind) =
            chain_step(&mut sink, blocks, prev, prev_out, layer.as_ref(), params)?;
        if prev_out.is_none() {
            input_len = in_width;
            input_kind = kind;
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
    blocks: &mut Vec<BlockBinding>,
    prev: u32,
    prev_out: Option<usize>,
    layer: &dyn Contract,
    params: &Params,
) -> Result<(u32, usize, usize, LayerKind)> {
    match layer.attention_block() {
        Some(block) => advance_block(sink, blocks, prev, prev_out, block, params),
        None => advance_layer(sink, prev, prev_out, layer, params),
    }
}

/// 注意力块层：几何/权重校验后拆成 q/k/v/o + decode 注意力 + 残差子节点。
fn advance_block(
    sink: &mut NodeSink<'_>,
    blocks: &mut Vec<BlockBinding>,
    prev: u32,
    prev_out: Option<usize>,
    block: &AttentionBlock,
    params: &Params,
) -> Result<(u32, usize, usize, LayerKind)> {
    check_block(block, prev_out, params)?;
    let next = push_block(sink, blocks, prev, block)?;
    let hidden = block.hidden();
    Ok((next, hidden, hidden, LayerKind::Projection))
}

/// 普通族别层：出计划后接单节点；返回输出编号、入宽、出宽与族别。
fn advance_layer(
    sink: &mut NodeSink<'_>,
    prev: u32,
    prev_out: Option<usize>,
    layer: &dyn Contract,
    params: &Params,
) -> Result<(u32, usize, usize, LayerKind)> {
    let plan = layer_plan(layer, prev_out, params)?;
    let next = sink.push_node(prev, &plan)?;
    Ok((next, plan.in_features, plan.out_features, plan.kind))
}

/// 把层栈装配成图：逐层接线，上一层输出即下一层输入。
///
/// 族别（投影 / 查表 / 一元点wise / 归一化 / 注意力块）决定操作数路数与权重形态；形状、宽度、权重缺失都在装配期报出，
/// 不带进执行期。量化族、standalone 二元残差、归约、位置编码按 [`Code::Unsupported`] 拒绝，不静默跳过。
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
    let (input_len, output_len, output, input_kind) = chain(
        layers,
        params,
        &mut graph,
        &mut weights,
        &mut activations,
        &mut descriptors,
        &mut blocks,
    )?;

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
