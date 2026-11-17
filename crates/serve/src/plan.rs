//! 层计划与族别分流：类型视图（投影/查表）优先，再按算子族别闭集归类等长族。
//!
//! 宽度链衔接、驻留权重条目齐备与形状相符都在装配期校验；本模块不出图接线，
//! 图节点收口在 [`crate::assembly`](crate::assembly)，注意力块子节点在 [`crate::block`](crate::block)。

use diagnostic::{Code, Error, Result};
use nn::{Contract, Embedding, Linear, Shape, WEIGHT_SUFFIX, WeightFormat};

use crate::kernels::SITE;
use crate::params::Params;

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
#[derive(Debug, Clone)]
pub(crate) struct LayerPlan {
    pub(crate) kind: LayerKind,
    pub(crate) operator: String,
    /// 驻留权重条目名；一元点wise无权重，为 `None`。
    pub(crate) entry: Option<String>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    /// 驻留权重元素数；一元点wise无权重，为 `None`。
    pub(crate) weight_len: Option<usize>,
    /// 驻留权重声明形状；无权重为空表。
    pub(crate) weight_shape: Vec<usize>,
}

/// 投影层必须是 F32 稠密格式，且能推出已注册算子名；其余在此拒绝。块内投影复用同一份判据。
pub(crate) fn plan_operator(name: &str, linear: &Linear) -> Result<&'static str> {
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
pub(crate) fn layer_plan(
    layer: &dyn Contract,
    prev_out: Option<usize>,
    params: &Params,
) -> Result<LayerPlan> {
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


