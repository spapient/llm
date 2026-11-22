//! 层计划与族别分流：类型视图（投影/查表）优先，再按算子族别闭集归类等长族。

use diagnostic::{Code, Error, Result};
use nn::{Contract, Embedding, Linear, MemoryBlock, Shape, WEIGHT_SUFFIX, WeightFormat};
use nn::quant::{I4, I8};

use crate::kernels::SITE;
use crate::params::Params;

/// 装配族：决定本层操作数怎么接线（输入路数、是否带驻留权重及权重形状）。
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
    /// 逐元素二元：主链输入 + 旁路激活 → 等长输出，无驻留权重。
    Binary,
    /// 位置编码：单路输入 → 等长输出；位置走标量槽（D9 / ADR 0043），无驻留权重。
    Rope,
    /// 条件记忆：主链输入 + 驻留表 `[buckets, width]` + 逐行槽号 → 等长输出（旁路加回）。
    Memory,
}

impl LayerKind {
    /// 链上本层与上一层输出对齐的宽度：投影/查表对齐输入宽，等长族（一元/归一化/二元/位置编码）对齐输出宽。
    fn chain_width(self, plan: &LayerPlan) -> usize {
        match self {
            Self::Projection | Self::Embedding => plan.in_features,
            Self::Unary | Self::Norm | Self::Binary | Self::Rope | Self::Memory => {
                plan.out_features
            }
        }
    }

    /// 本族是否需要旁路第二输入（逐元素二元的残差/门控路）。
    pub(crate) fn needs_bypass(self) -> bool {
        matches!(self, Self::Binary)
    }
}

/// 算子族别描述：一个算子名集合归到一个装配族。
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
    OperatorFamily {
        kind: LayerKind::Binary,
        operators: &["add_f32", "mul_f32", "residual_add_f32"],
    },
    OperatorFamily {
        kind: LayerKind::Rope,
        operators: &["rope_f32"],
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
#[derive(Debug, Clone)]
pub(crate) struct LayerPlan {
    pub(crate) kind: LayerKind,
    pub(crate) operator: String,
    /// 驻留权重条目名；一元点wise无权重，为 `None`。
    pub(crate) entry: Option<String>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    /// 驻留权重声明形状；无权重为空表。量化投影这里是打包字节形状 `[out, in/cpb]`；
    /// 元素数由形状元素积派生，不另存一份长度。
    pub(crate) weight_shape: Vec<usize>,
    /// 量化标度声明形状 `[out, in/group]`；非量化投影为空表，条数由形状元素积派生。
    pub(crate) scales_shape: Vec<usize>,
    /// 量化标度条目名；非量化投影为 `None`。
    pub(crate) scale_entry: Option<String>,
    /// 条件记忆的旁路几何；只有记忆族有，其余族为 `None`。
    pub(crate) memory: Option<MemorySpec>,
}

/// 条件记忆的旁路几何：行宽已进 [`LayerPlan`]，这里只留“查哪一行、加多少”的四个数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MemorySpec {
    /// N-gram 阶 `N`。
    pub(crate) order: usize,
    /// 槽哈希种子。
    pub(crate) seed: u64,
    /// 表槽数 `M`（也是驻留表行数）。
    pub(crate) buckets: usize,
    /// 门控强度 `γ`：历史不足 `order` 时宿主写 0，加回零向量。
    pub(crate) gate: f32,
}

/// 投影层必须能推出已注册算子名（F32 稠密或分组匹配内核的 int4/int8）；其余在此拒绝。
/// 块内投影复用同一份判据，格式到算子名的映射统一走 [`WeightFormat::operator`]，不另立一份表。
pub(crate) fn plan_operator(name: &str, linear: &Linear) -> Result<&'static str> {
    linear.format().operator().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!(
                "层 `{name}` 是 {}，没有可装配的内核：{}",
                linear.format(),
                linear.format().unpriced_reason()
            ),
        )
        .with(SITE)
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
pub(crate) fn check_weight(plan: &LayerPlan, params: &Params) -> Result<()> {
    let Some(entry) = &plan.entry else {
        return Ok(());
    };
    if plan.weight_shape.is_empty() {
        return Err(weight_shape_missing());
    }
    // 形状积走 `tensor::product` 的饱和乘法：回绕会乘出一个偏小的数骗过下面的长度核对。
    let weight_len = tensor::product(&plan.weight_shape);
    let scales_len = (!plan.scales_shape.is_empty()).then(|| tensor::product(&plan.scales_shape));
    check_weight_entry(
        params,
        entry,
        weight_len,
        plan.scale_entry.as_deref(),
        scales_len,
    )
}

/// 核对一条驻留权重 + 可选标度：稠密按 f32 元素数、量化按打包字节数 + 标度条数。
pub(crate) fn check_weight_entry(
    params: &Params,
    entry: &str,
    weight_len: usize,
    scale_entry: Option<&str>,
    scales_len: Option<usize>,
) -> Result<()> {
    check_entry(params, "权重", entry, weight_len)?;
    match (scale_entry, scales_len) {
        (Some(scale_entry), Some(scales_len)) => {
            check_entry(params, "标度", scale_entry, scales_len)
        }
        (None, None) => Ok(()),
        _ => Err(Error::new(Code::Missing, format!("量化条目 `{entry}` 缺标度条目")).with(SITE)),
    }
}

/// 条目存在且长度与期望一致：`Block::len` 按块自身形态取稠密元素数或打包字节数。
pub(crate) fn check_entry(
    params: &Params,
    label: &str,
    entry: &str,
    expected: usize,
) -> Result<()> {
    let Some(block) = params.get(entry) else {
        return Err(Error::new(Code::Missing, format!("参数表缺{label}条目 `{entry}`")).with(SITE));
    };
    if block.len() != expected {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "{label}条目 `{entry}` 有 {} 个元素/字节，而需要 {expected}",
                block.len()
            ),
        )
        .with(SITE));
    }
    Ok(())
}

/// 有条目必有权重形状的不变式违约。
pub(crate) fn weight_shape_missing() -> Error {
    Error::new(Code::Internal, "有条目必有权重形状").with(SITE)
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

/// 填一份层计划：族别几何由调用方给出，条目名、权重形状随权重形态保持一致。
fn build_plan(
    kind: LayerKind,
    operator: String,
    entry: Option<String>,
    in_features: usize,
    out_features: usize,
    weight_shape: Vec<usize>,
) -> LayerPlan {
    LayerPlan {
        kind,
        operator,
        entry,
        in_features,
        out_features,
        weight_shape,
        scales_shape: Vec::new(),
        scale_entry: None,
        memory: None,
    }
}

/// 驻留权重条目名：层名 + 权重后缀。
fn entry_name(layer: &dyn Contract) -> String {
    format!("{}{WEIGHT_SUFFIX}", layer.name())
}

/// 线性投影计划核心：身份、格式、算子名与权重/标度形状，不做宽度链校验。普通投影与块内
/// q/k/v/o 共用——块投影的宽度链由块几何单独校验，不走逐层链。
pub(crate) fn linear_plan(
    name: &str,
    entry: &str,
    linear: &Linear,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = plan_operator(name, linear)?;
    validate_quantized_shape(linear.format(), linear.in_features())?;
    let (weight_shape, scales_shape) =
        projection_shapes(linear.format(), linear.in_features(), linear.out_features());
    let mut plan = build_plan(
        LayerKind::Projection,
        operator.to_string(),
        Some(entry.to_string()),
        linear.in_features(),
        linear.out_features(),
        weight_shape,
    );
    plan.scales_shape = scales_shape;
    plan.scale_entry = params.scale_entry(entry).map(str::to_owned);
    Ok(plan)
}

/// 投影层装配计划：身份、格式、算子名与权重整块。量化格式给打包字节形状 + 标度形状。
fn projection_plan(
    layer: &dyn Contract,
    linear: &Linear,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let entry = entry_name(layer);
    let plan = linear_plan(layer.name(), &entry, linear, params)?;
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 投影的驻留权重与标度形状：稠密 `[out, in]` 整块 f32；量化给打包字节
/// `[out, in/cpb]` 加标度 `[out, in/group]`。每字节码数读 [`nn::quant::Row`] 配置表，
/// 与 `op` 的 GEMV/GEMM/反量化同一事实源（`I4.cpb=2`、`I8.cpb=1`）；分组来自格式声明。
pub(crate) fn projection_shapes(
    format: WeightFormat,
    in_features: usize,
    out_features: usize,
) -> (Vec<usize>, Vec<usize>) {
    match format {
        WeightFormat::Int4 { group } => (
            vec![out_features, in_features / I4.cpb],
            vec![out_features, in_features / group],
        ),
        WeightFormat::Int8 { group } => (
            vec![out_features, in_features / I8.cpb],
            vec![out_features, in_features / group],
        ),
        _ => (vec![out_features, in_features], Vec::new()),
    }
}

/// 量化投影的整除校验：输入宽度必须是分组宽度（与每字节码数）的整数倍。
pub(crate) fn validate_quantized_shape(format: WeightFormat, in_features: usize) -> Result<()> {
    let (cpb, group) = match format {
        WeightFormat::Int4 { group } => (I4.cpb, group),
        WeightFormat::Int8 { group } => (I8.cpb, group),
        _ => return Ok(()),
    };
    if !in_features.is_multiple_of(group) || !in_features.is_multiple_of(cpb) {
        return Err(Error::new(
            Code::Invalid,
            format!("量化投影输入宽度 {in_features} 不是分组 {group} 的整数倍"),
        )
        .with(SITE));
    }
    Ok(())
}

/// 等长族计划（一元点wise / 归一化 / 逐元素二元 / 位置编码）：宽度取自输出形状；归一化带 gamma
/// `[width]`，其余族无权重。
fn width_plan(
    kind: LayerKind,
    layer: &dyn Contract,
    operator: &str,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let width = axis_width(layer.output(), layer.name())?;
    // 等长族里只有归一化带 gamma `[width]`；一元点wise/二元/位置编码无权重。
    let weighted = kind == LayerKind::Norm;
    let weight_shape = weighted.then_some(vec![width]).unwrap_or_default();
    let plan = build_plan(
        kind,
        operator.to_string(),
        weighted.then(|| entry_name(layer)),
        width,
        width,
        weight_shape,
    );
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 查表计划：索引序列 + 驻留权重表 `[vocab, dim]` → 行向量块 `[tokens, dim]` 展平。
fn embedding_plan(
    layer: &dyn Contract,
    emb: &Embedding,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let operator = first_operator(layer)?;
    let (vocab, dim, tokens) = (emb.vocab(), emb.dim(), emb.tokens());
    let plan = build_plan(
        LayerKind::Embedding,
        operator.to_string(),
        Some(entry_name(layer)),
        tokens,
        tokens * dim,
        vec![vocab, dim],
    );
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 条件记忆计划：主链宽度不变，驻留条目是表 `[buckets, width]`，输出与输入等宽。
fn memory_plan(
    layer: &dyn Contract,
    memory: &MemoryBlock,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    if prev_out.is_none() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "层 `{}` 是条件记忆层，不能作首层：没有主链隐向量可加回",
                layer.name()
            ),
        )
        .with(SITE));
    }
    let width = memory.width();
    let mut plan = build_plan(
        LayerKind::Memory,
        first_operator(layer)?.to_string(),
        Some(entry_name(layer)),
        width,
        width,
        vec![memory.buckets(), width],
    );
    plan.memory = Some(MemorySpec {
        order: memory.order(),
        seed: memory.seed(),
        buckets: memory.buckets(),
        gate: memory.gate(),
    });
    check_plan(prev_out, &plan, layer.name(), params)?;
    Ok(plan)
}

/// 一层的装配分流：类型视图（投影/查表/条件记忆）优先；都不是再按算子族别归类。归约、注意力在本通路
/// 不接线（KV 通路随注意力刀落地）。
pub(crate) fn layer_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
    let Some(plan) = view_plan(layer, prev_out, params)? else {
        return operator_plan(layer, prev_out, params);
    };
    Ok(plan)
}

/// 类型视图层：能取到 `Linear` / `Embedding` / `MemoryBlock` 类型化视图的层出计划；都不是返回 `Ok(None)`。
fn view_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<Option<LayerPlan>> {
    match layer.linear() {
        Some(linear) => projection_plan(layer, linear, prev_out, params).map(Some),
        None => lookup_plan(layer, prev_out, params),
    }
}

/// 查表与条件记忆：两者都只挂一个驻留条目，按「查表优先」次序回答（同一层不会同时声明两视图）。
fn lookup_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<Option<LayerPlan>> {
    if let Some(emb) = layer.embedding() {
        let plan = embedding_plan(layer, emb, prev_out, params)?;
        return Ok(Some(plan));
    }
    let Some(memory) = layer.memory() else {
        return Ok(None);
    };
    memory_plan(layer, memory, prev_out, params).map(Some)
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
                "层 `{}` 的算子 `{operator}` 不在当前装配族别（投影/查表/条件记忆/一元点wise/归一化/二元/位置编码）",
                layer.name()
            ),
        )
        .with(SITE));
    };
    width_plan(kind, layer, operator, prev_out, params)
}
