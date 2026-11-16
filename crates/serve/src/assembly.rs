//! 装配管线：把 `nn` 层栈与驻留参数表接成 engine [`Graph`]，再由 [`CpuModel`] 宿主执行。
//!
//! 内核适配表留在 [`kernels`](crate::kernels)；本模块只负责层族别分流、宽度链与权重形状校验、图接线
//! 与前向取数，不认识算子内核算式。

use std::collections::BTreeMap;

use diagnostic::{Code, Error, Result};
use engine::Sequence;
use engine::exec::{CpuExecutor, Device, Executor, Graph, Launch, Node, Stream};
use nn::{Contract, Embedding, Linear, Shape, Stack, WEIGHT_SUFFIX, WeightFormat};
use weights::{Entry as WeightEntry, Header, Loader, Wire, decode_f32};

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

/// 驻留权重表：条目名 → f32 整块。装载期一次性解码进来，生成循环内不再读文件。
#[derive(Debug, Clone, Default)]
pub struct Params {
    entries: BTreeMap<String, Vec<f32>>,
}

impl Params {
    /// 空参数表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一条权重整块；同名条目覆盖旧值（装配期唯一性由 [`Stack`] 层名检查守住）。
    pub fn insert(&mut self, entry: impl Into<String>, weights: Vec<f32>) -> &mut Self {
        self.entries.insert(entry.into(), weights);
        self
    }

    /// 按条目名取权重整块。
    pub fn get(&self, entry: &str) -> Option<&[f32]> {
        self.entries.get(entry).map(Vec::as_slice)
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否一条权重都没有。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 按层栈与权重头部，经读取后端把 F32 条目解码进驻留表；非 F32 条目不填块，
    /// 留给装配期按格式拒绝。
    pub fn from_stack(stack: &Stack, header: &Header, loader: &impl Loader) -> Result<Self> {
        let mut params = Self::new();
        for layer in stack.layers() {
            let entry = Self::entry_for(layer.as_ref(), header)?;
            let Some(block) = Self::load_f32(entry, loader)? else {
                continue;
            };
            params.insert(entry.name().to_string(), block);
        }
        Ok(params)
    }

    /// 取层名对应的权重条目；头部缺条目报 [`Code::Missing`]。
    fn entry_for<'a>(layer: &dyn Contract, header: &'a Header) -> Result<&'a WeightEntry> {
        let name = format!("{}{WEIGHT_SUFFIX}", layer.name());
        header
            .get(&name)
            .ok_or_else(|| Error::new(Code::Missing, format!("权重头部缺条目 `{name}`")).with(SITE))
    }

    /// 解码一条 F32 条目；其余编码返回 `None`，不在装载期解包。
    fn load_f32(entry: &WeightEntry, loader: &impl Loader) -> Result<Option<Vec<f32>>> {
        if entry.dtype() != Wire::F32 {
            return Ok(None);
        }
        let bytes = loader.load(entry)?;
        decode_f32(&bytes).map(Some)
    }
}

/// 装配结果：图、输入/输出编号、权重编号 ↔ 条目名、各激活缓冲长度。
#[derive(Debug)]
struct Assembly {
    graph: Graph,
    input: u32,
    output: u32,
    input_len: usize,
    output_len: usize,
    /// 首层族别：决定一步前向的输入是 token 索引还是隐向量。
    input_kind: LayerKind,
    weights: Vec<(u32, String)>,
    activations: Vec<(u32, usize)>,
}

/// 装配族：决定本层操作数怎么接线（输入路数、是否带驻留权重及权重形状）。
///
/// 族别按算子 arity 与权重形态归类，是接线方式的机械分类，不是后端厂商分支。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
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

/// 一层装配所需的族别、算子名、权重条目名、几何宽度与驻留权重期望长度。
struct LayerPlan {
    kind: LayerKind,
    operator: String,
    /// 驻留权重条目名；一元点wise无权重，为 `None`。
    entry: Option<String>,
    in_features: usize,
    out_features: usize,
    /// 驻留权重元素数；一元点wise无权重，为 `None`。
    weight_len: Option<usize>,
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

/// 填一份层计划：族别几何由调用方给出，条目名与权重长度随权重形态保持一致。
fn build_plan(
    kind: LayerKind,
    operator: String,
    entry: Option<String>,
    in_features: usize,
    out_features: usize,
    weight_len: Option<usize>,
) -> LayerPlan {
    LayerPlan {
        kind,
        operator,
        entry,
        in_features,
        out_features,
        weight_len,
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
    let plan = build_plan(
        kind,
        operator.to_string(),
        weight_len.map(|_| entry_name(layer)),
        width,
        width,
        weight_len,
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

/// 层序号换算成编号：`base + index`；层序号不可能溢出 u32，溢出归内部错误。
fn slot_id(index: usize, base: u32) -> Result<u32> {
    u32::try_from(index)
        .map(|offset| base + offset)
        .map_err(|_| Error::new(Code::Internal, "层编号超出 u32 编号空间").with(SITE))
}

/// 把一层接进图：绑定上一层输出；投影/归一化再加驻留权重编号，一元点wise只有单路输入。
/// 返回本层输出激活编号。
fn push_node(
    graph: &mut Graph,
    weights: &mut Vec<(u32, String)>,
    activations: &mut Vec<(u32, usize)>,
    prev: u32,
    index: usize,
    plan: &LayerPlan,
) -> Result<u32> {
    let output_id = slot_id(index, ACTIVATION_BASE)?;
    let mut node = Node::new(
        plan.operator.clone(),
        Launch::new([1, 1, 1], [plan.out_features as u32, 1, 1]),
    );
    node.bind_input(prev);
    if let Some(entry) = &plan.entry {
        let weight_id = slot_id(index, WEIGHT_BASE)?;
        node.bind_input(weight_id);
        weights.push((weight_id, entry.clone()));
    }
    node.bind_output(output_id);
    graph.push(node);

    activations.push((output_id, plan.out_features));
    Ok(output_id)
}

/// 逐层串联：返回首层输入宽度、末层输出宽度与末层激活编号。
fn chain(
    layers: &[Box<dyn Contract>],
    params: &Params,
    graph: &mut Graph,
    weights: &mut Vec<(u32, String)>,
    activations: &mut Vec<(u32, usize)>,
) -> Result<(usize, usize, u32, LayerKind)> {
    let mut prev = INPUT_SLOT;
    let mut prev_out: Option<usize> = None;
    let mut input_len = 0;
    let mut last_out = 0;
    let mut input_kind = LayerKind::Projection;

    for (index, layer) in layers.iter().enumerate() {
        let plan = layer_plan(layer.as_ref(), prev_out, params)?;
        if prev_out.is_none() {
            input_len = plan.in_features;
            input_kind = plan.kind;
        }
        prev = push_node(graph, weights, activations, prev, index, &plan)?;
        prev_out = Some(plan.out_features);
        last_out = plan.out_features;
    }
    Ok((input_len, last_out, prev, input_kind))
}

/// 把层栈装配成图：逐层接线，上一层输出即下一层输入。
///
/// 族别（投影 / 查表 / 一元点wise / 归一化）决定操作数路数与权重形态；形状、宽度、权重缺失都在装配期报出，
/// 不带进执行期。量化族、二元残差、归约、位置编码、注意力按 [`Code::Unsupported`] 拒绝，不静默跳过。
fn assemble(stack: &Stack, params: &Params) -> Result<Assembly> {
    let layers = stack.layers();
    if layers.is_empty() {
        return Err(Error::new(Code::Invalid, "模型栈为空：没有可装配的层").with(SITE));
    }

    let mut graph = Graph::new();
    let mut activations = Vec::new();
    let mut weights: Vec<(u32, String)> = Vec::new();
    let (input_len, output_len, output, input_kind) =
        chain(layers, params, &mut graph, &mut weights, &mut activations)?;

    Ok(Assembly {
        graph,
        input: INPUT_SLOT,
        output,
        input_len,
        output_len,
        input_kind,
        weights,
        activations,
    })
}

/// 把驻留权重整块绑进执行器编号表。
fn bind_weights(executor: &mut CpuExecutor, assembly: &Assembly, params: &Params) -> Result<()> {
    for (id, entry) in &assembly.weights {
        let block = params.get(entry).ok_or_else(|| {
            Error::new(Code::Missing, format!("参数表缺权重条目 `{entry}`")).with(SITE)
        })?;
        executor.bind(*id, block.to_vec());
    }
    Ok(())
}

/// 预绑输入与各逐层激活的零缓冲：执行时只覆写，不在图运行期分配。
fn bind_activations(executor: &mut CpuExecutor, assembly: &Assembly) {
    executor.bind(assembly.input, vec![0.0; assembly.input_len]);
    for (id, len) in &assembly.activations {
        executor.bind(*id, vec![0.0; *len]);
    }
}

/// 提交图、同步执行并取回末层激活缓冲。
fn submit_and_fetch(executor: &mut CpuExecutor, assembly: &Assembly) -> Result<Vec<f32>> {
    executor.submit(&assembly.graph, Stream::new(0))?;
    executor.sync(Stream::new(0))?;
    let output = executor
        .get(assembly.output)
        .ok_or_else(|| Error::new(Code::Internal, "输出缓冲已预绑，取回失败").with(SITE))?;
    Ok(output.to_vec())
}

/// CPU 模型宿主：驻留权重 + 装配好的图 + 登记了内核的执行器。
///
/// 前向是张量级入口（[`CpuModel::forward`]）；接回 token 序列的 [`Predict`](crate::Predict)
/// 实现等嵌入通路落地后再补，本结构不另存 KV 账。
#[derive(Debug)]
pub struct CpuModel {
    executor: CpuExecutor,
    assembly: Assembly,
}

impl CpuModel {
    /// 由层栈与驻留参数表装配；形状、宽度、权重缺失都在构造期报出。
    pub fn new(stack: &Stack, params: &Params) -> Result<Self> {
        let mut executor = CpuExecutor::new(Device::new("cpu", "generic"));
        crate::kernels::register_cpu(&mut executor)?;
        let assembly = assemble(stack, params)?;
        bind_weights(&mut executor, &assembly, params)?;
        bind_activations(&mut executor, &assembly);

        Ok(Self { executor, assembly })
    }

    /// 一步张量级前向；输入长度必须等于首层输入宽度，返回末层激活。
    ///
    /// 权重在构造期驻留一次；每步只覆写输入与逐层激活缓冲，不重新分配。
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.assembly.input_len {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "输入 {} 维，模型首层要 {} 维",
                    input.len(),
                    self.assembly.input_len
                ),
            )
            .with(SITE));
        }
        self.executor.bind(self.assembly.input, input.to_vec());
        submit_and_fetch(&mut self.executor, &self.assembly)
    }

    /// 模型输出宽度（末层输出）。
    pub fn output_len(&self) -> usize {
        self.assembly.output_len
    }

    /// 按首层族别把序列当前位置 token 折成一步前向的输入缓冲。
    ///
    /// 查表层首层收一个索引（decode 一步取上下文末位 token）；投影/点wise/归一化首层需要的是
    /// 隐向量，而序列只持有 token id，给不出向量，当前通路拒绝，向量主链随连续批刀落地。
    fn predict_input(&self, sequence: &Sequence) -> Result<Vec<f32>> {
        match self.assembly.input_kind {
            LayerKind::Embedding => {
                let token = sequence.context().last().copied().ok_or_else(|| {
                    Error::new(Code::Invalid, "序列为空：没有可前向的 token").with(SITE)
                })?;
                Ok(vec![token as f32])
            }
            LayerKind::Projection | LayerKind::Unary | LayerKind::Norm => Err(Error::new(
                Code::Unsupported,
                "首层不是查表层：序列只持有 token id，无法提供隐向量输入；向量主链通路随连续批落地",
            )
            .with(SITE)),
        }
    }
}

impl crate::Predict for CpuModel {
    fn logits(&mut self, sequence: &Sequence) -> Result<Vec<f32>> {
        let input = self.predict_input(sequence)?;
        self.forward(&input)
    }
}
