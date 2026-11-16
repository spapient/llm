//! CPU 推理内核桥：把 `op` 的 f32 标量参考适配成 engine 可登记的 [`Program`]。
//!
//! 落点裁决见 [`backend-runtime-design.md`](../../../docs/backend-runtime-design.md) §4：
//! 端点契约、路由与线格式保持模型无关；CPU 模型实现作为注入 [`Predict`](crate::Predict)
//! 的提供者收在本模块内，真设备后端仍按 ADR 0025 收在独立边界 crate，不回流这里。
//!
//! # 职责
//!
//! - 编译期锁定的算子表：表内每个名字都必须在 `op::REGISTRY` 内（`const` 子集断言）；
//!   表未覆盖的算子不登记，执行时由 [`CpuExecutor`](engine::exec::CpuExecutor) 按名拒绝，
//!   不留静默空入口。
//! - 适配函数只做拆切片 + 调 `op::<op>::forward`，算术不重写。模式复刻 `train::KERNELS`
//!   （`crates/train/src/kernel.rs`），但不带 train 的 `View`/`Lane` 适配层：engine 缓冲
//!   形态就是 `Vec<f32>` 切片。
//!
//! # 不拥有
//!
//! 执行/调度/KV 簿记（`engine`）、算子声明与参考算术（`op`）、层形状语义（`nn`）、
//! 权重字节读取（`weights`）。本模块不出现设备厂商或后端类型的特判分支。
//!
//! # P1 范围
//!
//! 只接纯切片签名的 f32 算子；`rope_f32` 的位置标量、`attention_*` 的头几何随 P2 KV
//! 工作一起落地（设计文档 §5.1）。u8 lane 量化族延后 P3（§5.3）。

use std::collections::BTreeMap;

use diagnostic::{Code, Error, Result};
use engine::exec::{CpuExecutor, Device, Executor, Graph, Launch, Node, Program, Stream};
use nn::{Contract, Embedding, Linear, Shape, Stack, WEIGHT_SUFFIX, WeightFormat};
use op::REGISTRY;
use weights::{Entry as WeightEntry, Header, Loader, Wire, decode_f32};

const SITE: &str = "serve::model";

/// 激活编号与权重编号的隔离基址：权重整块从这个编号起登记，避免与逐层复用的激活编号撞上。
const WEIGHT_BASE: u32 = 10_000;

/// 适配入口：拆好切片后调用算子参考实现。
type KernelFn = fn(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()>;

/// 表内一行：算子名（与 `op::REGISTRY` 同名）+ 缓冲级函数指针。
struct Entry {
    op: &'static str,
    run: KernelFn,
}

/// 一元 `[f32] → [f32]` 适配：套用算子的标量实现。
macro_rules! unary {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
            let (Some(input), Some(output)) = (inputs.first(), outputs.first_mut()) else {
                return Err(Error::new(
                    Code::Invalid,
                    concat!(stringify!($run), " 需要一进一出"),
                )
                .with(SITE));
            };
            op::$run::forward(input, output)
        }
    )*};
}

/// 二元 `[f32, f32] → [f32]` 适配。
macro_rules! binary {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
            let (Some(left), Some(right), Some(output)) =
                (inputs.first(), inputs.get(1), outputs.first_mut())
            else {
                return Err(Error::new(
                    Code::Invalid,
                    concat!(stringify!($run), " 需要两进一出"),
                )
                .with(SITE));
            };
            op::$run::forward(left, right, output)
        }
    )*};
}

/// 二路适配：`[激活/索引, 驻留权重整块] → [out]`；第一路是输入向量或索引序列，第二路是驻留权重。
macro_rules! gemv {
    ($($run:ident),* $(,)?) => {$(
        fn $run(inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
            let (Some(x), Some(weights), Some(output)) =
                (inputs.first(), inputs.get(1), outputs.first_mut())
            else {
                return Err(Error::new(
                    Code::Invalid,
                    concat!(stringify!($run), " 需要两进一出"),
                )
                .with(SITE));
            };
            op::$run::forward(x, weights, output)
        }
    )*};
}

unary!(
    gelu_f32,
    reduce_max_f32,
    reduce_sum_f32,
    scale_f32,
    silu_f32,
    softmax_f32
);
binary!(add_f32, mul_f32, residual_add_f32, rmsnorm_f32);
gemv!(embed_f32, gemv_f32);

/// CPU 内核表：按 `op::REGISTRY` 顺序排列的 f32 纯切片子集；长度由表本身数出。
const KERNELS: &[Entry] = &[
    Entry {
        op: "add_f32",
        run: add_f32,
    },
    Entry {
        op: "embed_f32",
        run: embed_f32,
    },
    Entry {
        op: "gelu_f32",
        run: gelu_f32,
    },
    Entry {
        op: "gemv_f32",
        run: gemv_f32,
    },
    Entry {
        op: "mul_f32",
        run: mul_f32,
    },
    Entry {
        op: "reduce_max_f32",
        run: reduce_max_f32,
    },
    Entry {
        op: "reduce_sum_f32",
        run: reduce_sum_f32,
    },
    Entry {
        op: "residual_add_f32",
        run: residual_add_f32,
    },
    Entry {
        op: "rmsnorm_f32",
        run: rmsnorm_f32,
    },
    Entry {
        op: "scale_f32",
        run: scale_f32,
    },
    Entry {
        op: "silu_f32",
        run: silu_f32,
    },
    Entry {
        op: "softmax_f32",
        run: softmax_f32,
    },
];

const _: () = all_in_registry(KERNELS, &REGISTRY);

/// 编译期断言：表内每个算子名都能在 `op::REGISTRY` 里找到（子集即可，不必等长）。
const fn all_in_registry(entries: &[Entry], registry: &[op::Registration]) {
    let mut index = 0;
    while index < entries.len() {
        assert!(
            name_located(entries[index].op, registry),
            "serve::model 内核表含注册表之外的算子"
        );
        index += 1;
    }
}

/// `const` 上下文里没有 `str` 比较，逐字节比名字。
const fn name_located(name: &str, registry: &[op::Registration]) -> bool {
    let mut index = 0;
    while index < registry.len() {
        if same_str(name, registry[index].name) {
            return true;
        }
        index += 1;
    }
    false
}

const fn same_str(left: &str, right: &str) -> bool {
    let (left, right) = (left.as_bytes(), right.as_bytes());
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// 表条目对 engine [`Program`] 契约的适配：名字随表走，执行入口落到适配函数。
struct Kernel {
    entry: &'static Entry,
}

impl Program for Kernel {
    fn name(&self) -> &str {
        self.entry.op
    }

    fn run(&self, _launch: Launch, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        (self.entry.run)(inputs, outputs)
    }
}

/// 把表内全部 CPU 内核登记进执行器；重名由 [`CpuExecutor::register`] 拒绝。
///
/// 装配侧在构造 CPU 模型宿主时调用一次；之后执行器只收图与操作数编号，不再改内核表。
pub fn register_cpu(executor: &mut CpuExecutor) -> Result<()> {
    for entry in KERNELS {
        executor.register(Box::new(Kernel { entry }))?;
    }
    Ok(())
}

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

/// 一元点wise族算子：一进一出、无权重，输入输出等长。
const UNARY_OPS: &[&str] = &["gelu_f32", "scale_f32", "silu_f32", "softmax_f32"];

/// 归一化族算子：输入 + gamma 向量 → 等长输出。
const NORM_OPS: &[&str] = &["rmsnorm_f32"];

/// 算子名是否在一张常量表内（闭集检索，表为编译期定长切片）。
fn in_table(operator: &str, table: &[&str]) -> bool {
    table.contains(&operator)
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
    let required = match plan.kind {
        LayerKind::Projection | LayerKind::Embedding => plan.in_features,
        LayerKind::Unary | LayerKind::Norm => plan.out_features,
    };
    if previous != required {
        return Err(Error::new(
            Code::Invalid,
            format!("层 `{name}` 宽度 {required} 与上一层输出宽度 {previous} 不符"),
        )
        .with(SITE));
    }
    Ok(())
}

/// 驻留权重期望长度：族别在出计划时填好（投影 `in × out`、归一化 gamma `width`、查表 `vocab × dim`）；
/// 一元点wise无权重返回 `None`。
fn weight_len(plan: &LayerPlan) -> Option<usize> {
    plan.weight_len
}

/// 核对驻留权重条目存在且长度符合族别；一元点wise无权重，直接通过。
fn check_weight(plan: &LayerPlan, params: &Params) -> Result<()> {
    let Some(expected) = weight_len(plan) else {
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

/// 投影层装配计划：身份、F32 格式、算子名与 `[out, in]` 权重整块。
fn projection_plan(
    layer: &dyn Contract,
    linear: &Linear,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = plan_operator(layer.name(), linear)?;
    let plan = LayerPlan {
        kind: LayerKind::Projection,
        operator: operator.to_string(),
        entry: Some(format!("{}{WEIGHT_SUFFIX}", layer.name())),
        in_features: linear.in_features(),
        out_features: linear.out_features(),
        weight_len: Some(linear.in_features() * linear.out_features()),
    };
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 一元点wise计划：等长、无权重；宽度取自输出形状。
fn unary_plan(layer: &dyn Contract, operator: &str, prev_out: Option<usize>) -> Result<LayerPlan> {
    let width = axis_width(layer.output(), layer.name())?;
    let plan = LayerPlan {
        kind: LayerKind::Unary,
        operator: operator.to_string(),
        entry: None,
        in_features: width,
        out_features: width,
        weight_len: None,
    };
    check_chain(prev_out, &plan, layer.name())?;
    Ok(plan)
}

/// 归一化计划：等长、驻留 gamma `[width]`；宽度取自输出形状。
fn norm_plan(
    layer: &dyn Contract,
    operator: &str,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let width = axis_width(layer.output(), layer.name())?;
    let plan = LayerPlan {
        kind: LayerKind::Norm,
        operator: operator.to_string(),
        entry: Some(format!("{}{WEIGHT_SUFFIX}", layer.name())),
        in_features: width,
        out_features: width,
        weight_len: Some(width),
    };
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
    let plan = LayerPlan {
        kind: LayerKind::Embedding,
        operator: operator.to_string(),
        entry: Some(format!("{}{WEIGHT_SUFFIX}", layer.name())),
        in_features: tokens,
        out_features: dim,
        weight_len: Some(vocab * dim),
    };
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

/// 算子族别层：按常量表归类一元点wise / 归一化；其余族别报 [`Code::Unsupported`]。
fn operator_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = first_operator(layer)?;
    if in_table(operator, UNARY_OPS) {
        return unary_plan(layer, operator, prev_out);
    }
    if in_table(operator, NORM_OPS) {
        return norm_plan(layer, operator, prev_out, params);
    }
    Err(Error::new(
        Code::Unsupported,
        format!(
            "层 `{}` 的算子 `{operator}` 不在当前装配族别（投影/查表/一元点wise/归一化）",
            layer.name()
        ),
    )
    .with(SITE))
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
    let output_id = slot_id(index, 1)?;
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
) -> Result<(usize, usize, u32)> {
    let mut prev = 0;
    let mut prev_out: Option<usize> = None;
    let mut input_len = 0;
    let mut last_out = 0;

    for (index, layer) in layers.iter().enumerate() {
        let plan = layer_plan(layer.as_ref(), prev_out, params)?;
        if prev_out.is_none() {
            input_len = plan.in_features;
        }
        prev = push_node(graph, weights, activations, prev, index, &plan)?;
        prev_out = Some(plan.out_features);
        last_out = plan.out_features;
    }
    Ok((input_len, last_out, prev))
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
    let (input_len, output_len, output) =
        chain(layers, params, &mut graph, &mut weights, &mut activations)?;

    Ok(Assembly {
        graph,
        input: 0,
        output,
        input_len,
        output_len,
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
        register_cpu(&mut executor)?;
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
}

#[cfg(test)]
#[path = "../tests/model_tests.rs"]
mod tests;
