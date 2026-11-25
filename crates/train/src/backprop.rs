//! 反向数值执行器：按 [`Schedule`] 的逆拓扑序把 loss 梯度回传到参数终点。

use autograd::{Node, Schedule, Tape, Var};
use diagnostic::{Code, Error, Result};

use crate::geometry::Geometry;
use crate::pool::Bufs;
use crate::{Kernel, same_str};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::backprop";

/// 按已规划的 [`Schedule`] 回传梯度：起点 loss 梯度为全 `1`，逐节点分配到图上输入。
///
/// 梯度累加进 `bufs` 的梯度槽；图外（需求为无梯度）输入不收账。
///
/// # Errors
///
/// 前向值/上游梯度不在场、账本顺序不符，或遇到尚未接反向公式的算子时返回对应 [`Code`]。
pub(crate) fn backprop<K: Kernel>(
    bufs: &mut Bufs<K>,
    tape: &Tape,
    schedule: &Schedule,
) -> Result<()> {
    // 暂存从缓冲区里借出：`Ctx` 持只读快照、`apply_formula` 要 `&mut bufs`，
    // 同一结构无法同时借，故把暂存放在独立结构里逐节点复用——稳态不再向分配器要内存。
    let mut scratch = std::mem::take(&mut bufs.scratch);
    seed_loss(bufs, schedule.loss(), &mut scratch)?;
    let mut account = schedule.account();
    for step in schedule.steps() {
        reverse_step(bufs, tape, step, &mut scratch)?;
        account.settle(step).map_err(|error| error.with(SITE))?;
    }
    bufs.scratch = scratch;
    Ok(())
}

/// 单节点反向：取上游梯度、装配公式上下文，查表执行并把逐输入贡献喂回梯度槽。
fn reverse_step<K: Kernel>(
    bufs: &mut Bufs<K>,
    tape: &Tape,
    step: &Node,
    scratch: &mut Scratch,
) -> Result<()> {
    copy_into(&mut scratch.incoming, incoming(bufs, step)?);
    collect_inputs(bufs, step, &mut scratch.inputs)?;
    copy_into(&mut scratch.output, values(bufs, step.output())?);
    let ctx = Ctx {
        incoming: &scratch.incoming,
        inputs: &scratch.inputs,
        output: &scratch.output,
        geometry: bufs.geometry(step.output()),
    };
    apply_formula(bufs, tape, step, ctx)
}

/// 反向逐节点复用的暂存：容量只增不减，热路径上不再逐节点向分配器要内存。
///
/// 三者都是本节点前向快照的只读副本，故与 `Bufs` 分开持有：一份借不冲突的读视图，
/// 一份可写缓冲区，40 条公式因此仍然只吃 `Ctx`、不碰 `Bufs`。
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    incoming: Vec<f32>,
    inputs: Vec<Vec<f32>>,
    output: Vec<f32>,
    ones: Vec<f32>,
}

impl Scratch {
    /// 暂存的实际占用（**容量**之和）：容量只增不减，故容量不变即"没有重新分配"。
    #[cfg(test)]
    pub(crate) fn footprint(&self) -> usize {
        self.incoming.capacity()
            + self.output.capacity()
            + self.ones.capacity()
            + self.inputs.capacity()
            + self.inputs.iter().map(Vec::capacity).sum::<usize>()
    }
}

/// 整块覆盖写：保留已有容量，只改变长度与内容。
fn copy_into(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.extend_from_slice(src);
}

/// 把各输入的前向值整块拷出只读快照；任一前向值不在场即回缺失。
fn collect_inputs<K: Kernel>(bufs: &Bufs<K>, step: &Node, out: &mut Vec<Vec<f32>>) -> Result<()> {
    let inputs = step.inputs();
    out.resize_with(inputs.len(), Vec::new);
    for (slot, &input) in out.iter_mut().zip(inputs) {
        copy_into(slot, values(bufs, input)?);
    }
    Ok(())
}

/// 按算子名查公式目录执行，把返回的逐输入贡献喂回对应梯度槽；未登记公式即拒绝。
fn apply_formula<K: Kernel>(
    bufs: &mut Bufs<K>,
    tape: &Tape,
    step: &Node,
    ctx: Ctx<'_>,
) -> Result<()> {
    let formula = lookup(step.op())?;
    let contributions = formula(ctx)?;
    for (&input, contrib) in step.inputs().iter().zip(contributions) {
        feed(bufs, tape, input, contrib)?;
    }
    Ok(())
}

/// 记一份梯度贡献；图外输入没有梯度缓冲，直接丢弃。
fn feed<K: Kernel>(bufs: &mut Bufs<K>, tape: &Tape, var: Var, contrib: Vec<f32>) -> Result<()> {
    let receives = tape
        .requirement(var)
        .is_some_and(|requirement| !requirement.is_none());
    if receives {
        bufs.accumulate_gradient(var, &contrib)
            .map_err(|error| error.with(SITE))?;
    }
    Ok(())
}

/// loss 起点梯度：每个前向元素都从上游 `1` 出发。
fn seed_loss<K: Kernel>(bufs: &mut Bufs<K>, loss: Var, scratch: &mut Scratch) -> Result<()> {
    let width = values(bufs, loss)?.len();
    scratch.ones.clear();
    scratch.ones.resize(width, 1.0);
    bufs.load_gradient(loss, &scratch.ones)
        .map_err(|error| error.with(SITE))?;
    Ok(())
}

/// 读变量在场的前向 f32 值；缺失是重放/装配口径问题。
fn values<K: Kernel>(bufs: &Bufs<K>, var: Var) -> Result<&[f32]> {
    bufs.value(var).and_then(|view| view.f32()).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("变量 {} 的前向值不在场，无法反传", var.id()),
        )
        .with(SITE)
    })
}

/// 读节点输出已就位的上游梯度；缺失说明拓扑序上有前驱没回传到。
fn incoming<'a, K: Kernel>(bufs: &'a Bufs<K>, step: &Node) -> Result<&'a [f32]> {
    bufs.gradient(step.output()).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("节点 {} 的上游梯度未就位", step.output().id()),
        )
        .with(SITE)
    })
}

fn unsupported(op: &str) -> Error {
    Error::new(Code::Unsupported, format!("算子 `{op}` 还没有解析反向公式")).with(SITE)
}

/// 公式只读上下文：上游梯度、各输入前向值、输出前向值（均为本节点前向快照）。
struct Ctx<'a> {
    incoming: &'a [f32],
    inputs: &'a [Vec<f32>],
    output: &'a [f32],
    /// 本节点的实例几何；`None` 即该节点按声明档实例执行。
    geometry: Option<Geometry>,
}

impl Ctx<'_> {
    /// 本节点的实例几何：显式登记过就用它，否则取该算子的声明档实例（对拍基准）。
    fn instance(&self, op: &str) -> Result<Geometry> {
        match self.geometry {
            Some(geometry) => Ok(geometry),
            None => Geometry::declared(op),
        }
    }
}

/// 反向公式：按上下文算出与输入等长的逐元素贡献，顺序与节点输入对齐。
type Formula = fn(Ctx<'_>) -> Result<Vec<Vec<f32>>>;

/// 一条公式目录项：算子名与其解析反向公式。
struct Entry {
    op: &'static str,
    formula: Formula,
}

/// 已接反向公式的可微算子目录；其余算子（量化权重 / 反量化）在 [`apply_formula`] 处回
/// [`Code::Unsupported`]。与 [`nn::registry::REGISTRY`] 的 `differentiable` 标记一一对应，由下方 `const`
/// 断言在编译期守住。
/// 展开公式目录：`op => formula` 一行一条，数组长度由下方 [`FORMULAS`] 的声明给定。
macro_rules! formulas {
    ($($op:literal => $formula:expr),* $(,)?) => {
        [$(Entry { op: $op, formula: $formula }),*]
    };
}

/// 实例几何目录项：非捕获闭包直接 coerce 为公式函数指针。
///
/// 定义必须先于 [`FORMULAS`]（`macro_rules!` 按文本顺序解析）。
macro_rules! instance_backward {
    ($op:literal) => {
        |ctx: Ctx<'_>| -> Result<Vec<Vec<f32>>> {
            ctx.instance($op)?.backward(ctx.inputs, ctx.incoming)
        }
    };
}

const FORMULAS: [Entry; 40] = formulas![
    "adaptive_pool_f32" => instance_backward!("adaptive_pool_f32"),
    "add_f32" => passthrough,
    "residual_add_f32" => passthrough,
    "mul_f32" => mul,
    "scale_f32" => |ctx| unary(ctx, |g, _| g * nn::scale_f32::ALPHA),
    "sub_f32" => sub,
    "div_f32" => div,
    "exp_f32" => |ctx| output_unary(ctx, |g, y| g * y),
    "log_f32" => |ctx| unary(ctx, |g, x| g / x),
    "min_f32" => min,
    "max_f32" => max,
    "max_pool1d_f32" => instance_backward!("max_pool1d_f32"),
    "max_pool2d_f32" => instance_backward!("max_pool2d_f32"),
    "silu_f32" => |ctx| unary(ctx, |g, v| {
        let sigmoid = 1.0 / (1.0 + (-v).exp());
        g * (sigmoid + v * sigmoid * (1.0 - sigmoid))
    }),
    "gelu_f32" => |ctx| unary(ctx, |g, x| g * gelu_grad(x)),
    "hard_sigmoid_f32" => |ctx| unary(ctx, |g, v| {
        let v = f64::from(v);
        let inside = (v / 6.0 + 0.5) > 0.0 && (v / 6.0 + 0.5) < 1.0;
        g * if inside { 1.0 / 6.0 } else { 0.0 }
    }),
    "instance_norm_f32" => instance_backward!("instance_norm_f32"),
    "interpolate_f32" => instance_backward!("interpolate_f32"),
    "layer_norm_f32" => instance_backward!("layer_norm_f32"),
    "leaky_relu_f32" => |ctx| unary(ctx, |g, v| {
        g * if v >= 0.0 { 1.0 } else { nn::leaky_relu_f32::SLOPE }
    }),
    "relu_f32" => |ctx| unary(ctx, |g, v| g * if v > 0.0 { 1.0 } else { 0.0 }),
    "sigmoid_f32" => |ctx| output_unary(ctx, |g, y| g * y * (1.0 - y)),
    "tanh_f32" => |ctx| output_unary(ctx, |g, y| g * (1.0 - y * y)),
    "reduce_sum_f32" => reduce_sum,
    "reduce_max_f32" => reduce_max,
    "softmax_f32" => softmax,
    "gemv_f32" => gemv,
    "group_norm_f32" => instance_backward!("group_norm_f32"),
    "gemm_f32" => gemm,
    "embed_f32" => embed,
    "rmsnorm_f32" => rmsnorm,
    "rope_f32" => rope,
    "attention_f32" => attention,
    "latent_attention_f32" => latent_attention,
    "avg_pool2d_f32" => instance_backward!("avg_pool2d_f32"),
    "batch_norm_f32" => instance_backward!("batch_norm_f32"),
    "conv2d_f32" => instance_backward!("conv2d_f32"),
    "flash_attention_f32" => attention,
    "decode_attention_f32" => decode_attention,
    "decode_flash_attention_f32" => decode_attention,
];

/// 编译期闭合：公式目录与 [`nn::registry::REGISTRY`] 的 `differentiable` 标记一一对应。
const _: () = formulas_cover(&FORMULAS, &nn::registry::REGISTRY);

/// 双向覆盖且总数相等即构成一一对应，因此也排除重复公式。
const fn formulas_cover(formulas: &[Entry], registry: &[prof::catalogue::Registration]) {
    let mut i = 0;
    let mut expected = 0;
    while i < registry.len() {
        if registry[i].differentiable {
            expected += 1;
            assert!(
                has_formula(formulas, registry[i].name),
                "可微算子缺少反向公式"
            );
        }
        i += 1;
    }
    assert!(expected == formulas.len(), "反向公式数与可微算子数不符");
    let mut j = 0;
    while j < formulas.len() {
        assert!(
            differentiable(registry, formulas[j].op),
            "公式指向不可微或未注册算子"
        );
        j += 1;
    }
}

/// 公式目录里是否已有某算子名。
const fn has_formula(formulas: &[Entry], name: &str) -> bool {
    let mut i = 0;
    while i < formulas.len() {
        if same_str(formulas[i].op, name) {
            return true;
        }
        i += 1;
    }
    false
}

/// 算子名是否登记为可微；未注册返回 `false`（不存在"可微但未注册"的算子）。
const fn differentiable(registry: &[prof::catalogue::Registration], name: &str) -> bool {
    let mut i = 0;
    while i < registry.len() {
        if same_str(registry[i].name, name) {
            return registry[i].differentiable;
        }
        i += 1;
    }
    false
}

/// 按算子名查公式；目录未登记即回 [`Code::Unsupported`]。
fn lookup(op: &str) -> Result<Formula> {
    FORMULAS
        .iter()
        .find(|entry| entry.op == op)
        .map(|entry| entry.formula)
        .ok_or_else(|| unsupported(op))
}

// ---- 反向公式表：40 条解析公式与共享数值助手 ----

/// y = a + b：两条路径都原样回传上游梯度。
fn passthrough(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![ctx.incoming.to_vec(); ctx.inputs.len()])
}

/// y = a·b：dA = in·b，dB = in·a。
fn mul(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let da = ctx.incoming.iter().zip(b).map(|(&g, w)| g * w).collect();
    let db = ctx.incoming.iter().zip(a).map(|(&g, w)| g * w).collect();
    Ok(vec![da, db])
}

/// y = a−b：dA = in，dB = −in。
fn sub(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![
        ctx.incoming.to_vec(),
        ctx.incoming.iter().map(|&g| -g).collect(),
    ])
}

/// y = a/b：dA = in/b，dB = −in·a/b²。
fn div(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let (a, b) = (&ctx.inputs[0], &ctx.inputs[1]);
    let da = ctx.incoming.iter().zip(b).map(|(&g, &w)| g / w).collect();
    let db = ctx
        .incoming
        .iter()
        .zip(a)
        .zip(b)
        .map(|((&g, &x), &w)| -g * x / (w * w))
        .collect();
    Ok(vec![da, db])
}

/// y = min(a,b)：梯度给较小者；相等给 `a`（避免双计）。
fn min(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    extremum(ctx, |x, y| x <= y, |x, y| y < x)
}

/// y = max(a,b)：梯度给较大者；相等给 `a`（避免双计）。
fn max(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    extremum(ctx, |x, y| x >= y, |x, y| y > x)
}

/// `min`/`max` 共用选点反向，等号只归 `a`。
fn extremum(
    ctx: Ctx<'_>,
    a_wins: impl Fn(f32, f32) -> bool,
    b_wins: impl Fn(f32, f32) -> bool,
) -> Result<Vec<Vec<f32>>> {
    let (a, b) = (&ctx.inputs[0], &ctx.inputs[1]);
    let pick = |wins: &dyn Fn(f32, f32) -> bool| {
        ctx.incoming
            .iter()
            .zip(a)
            .zip(b)
            .map(|((&g, &x), &y)| if wins(x, y) { g } else { 0.0 })
            .collect()
    };
    Ok(vec![pick(&a_wins), pick(&b_wins)])
}

/// 一元公式目录的公共迭代器；闭包保留每项自己的浮点结合顺序。
fn unary(ctx: Ctx<'_>, contribution: impl Fn(f32, f32) -> f32) -> Result<Vec<Vec<f32>>> {
    Ok(vec![
        ctx.incoming
            .iter()
            .zip(&ctx.inputs[0])
            .map(|(&g, &x)| contribution(g, x))
            .collect(),
    ])
}

/// 输出可复用的一元公式（exp/sigmoid/tanh），避免重算前向函数。
fn output_unary(ctx: Ctx<'_>, contribution: impl Fn(f32, f32) -> f32) -> Result<Vec<Vec<f32>>> {
    Ok(vec![
        ctx.incoming
            .iter()
            .zip(ctx.output)
            .map(|(&g, &y)| contribution(g, y))
            .collect(),
    ])
}

/// y = Σa：上游标量梯度广播回每个元素，dA[i] = in[0]。
fn reduce_sum(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = vec![ctx.incoming[0]; ctx.inputs[0].len()];
    Ok(vec![da])
}

/// y = max(a)：只有最大值元素收到 `in[0]`，其余梯度为 0。
fn reduce_max(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let peak = a.iter().copied().fold(f32::MIN, f32::max);
    let da = a
        .iter()
        .map(|&v| if v == peak { ctx.incoming[0] } else { 0.0 })
        .collect();
    Ok(vec![da])
}

/// p = softmax(x)：dA[i] = p[i]·(in[i] − Σ_j in[j]·p[j])。
fn softmax(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let probability = ctx.output;
    let dot: f32 = ctx
        .incoming
        .iter()
        .zip(probability)
        .map(|(g, &p)| g * p)
        .sum();
    let da = ctx
        .incoming
        .iter()
        .zip(probability)
        .map(|(&g, &p)| p * (g - dot))
        .collect();
    Ok(vec![da])
}

/// `y = W·x`（`[n] = Σ_k W[n,k]·x[k]`）：返回 `[dW, dx]`。
fn gemv(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let weight = &ctx.inputs[0][..];
    let input = &ctx.inputs[1][..];
    let (rows, cols) = matrix_shape(weight, input.len())?;
    let dweight = gemv_weight(weight, input, ctx.incoming, rows, cols);
    let dinput = gemv_input(weight, ctx.incoming, rows, cols);
    Ok(vec![dweight, dinput])
}

/// `dW[n,k] = in[n]·x[k]`：外层行号决定被广播的上游分量，内层整块复制输入行。
fn gemv_weight(
    weight: &[f32],
    input: &[f32],
    incoming: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut dweight = vec![0.0; weight.len()];
    for n in 0..rows {
        let row = &mut dweight[n * cols..(n + 1) * cols];
        for (slot, &x_value) in row.iter_mut().zip(input) {
            *slot = incoming[n] * x_value;
        }
    }
    dweight
}

/// `dx[k] = Σ_n in[n]·W[n,k]`：按列累加，W 按行主序展开。
fn gemv_input(weight: &[f32], incoming: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dinput = vec![0.0; cols];
    for n in 0..rows {
        for k in 0..cols {
            dinput[k] += incoming[n] * weight[n * cols + k];
        }
    }
    dinput
}

/// `Y = A·B`：返回 `[dA, dB]`，收缩轴固定 `nn::gemm::K`。
fn gemm(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0][..];
    let b = &ctx.inputs[1][..];
    let grad = &ctx.incoming;
    let k = nn::gemm::K;
    let m = a.len() / k;
    let n = grad.len() / m;
    let mut da = vec![0.0; a.len()];
    let mut db = vec![0.0; b.len()];
    for m_row in 0..m {
        for k_dim in 0..k {
            for n_col in 0..n {
                let g = grad[m_row * n + n_col];
                da[m_row * k + k_dim] += g * b[k_dim * n + n_col];
                db[k_dim * n + n_col] += a[m_row * k + k_dim] * g;
            }
        }
    }
    Ok(vec![da, db])
}

/// `y = table[idx]`（行 gather）：索引不连续、无雅可比；权重表按被访问行累加上游行。
fn embed(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let idx = &ctx.inputs[0][..];
    let table = &ctx.inputs[1][..];
    let dim = ctx.incoming.len() / idx.len();
    let mut dtable = vec![0.0; table.len()];
    for (token, &entry) in idx.iter().enumerate() {
        let row = entry as usize * dim;
        let grad_row = &ctx.incoming[token * dim..][..dim];
        accumulate_into(&mut dtable, row, grad_row)?;
    }
    // 索引输入无雅可比：贡献置空，只回传表梯度。
    Ok(vec![vec![0.0; idx.len()], dtable])
}

/// `y = γ·x / rms(x)`：`dx[j] = g[j]·γ[j]/r − x[j]·Σ_i g[i]·γ[i]·x[i] / (n·r³)`，
/// `dγ[i] = g[i]·x[i]/r`。
fn rmsnorm(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let (dx, dgamma) = rmsnorm_grad(&ctx.inputs[0], &ctx.inputs[1], ctx.incoming)?;
    Ok(vec![dx, dgamma])
}

/// RMSNorm 反向本体：均方根与加权和先各算一遍，逐元素公式用迭代器表达，不展开第二套循环。
fn rmsnorm_grad(x: &[f32], gamma: &[f32], grad: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = x.len();
    if n == 0 {
        return Err(Error::new(Code::Invalid, "rmsnorm 反向输入不能为空").with(SITE));
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (mean_sq + nn::rmsnorm_f32::EPS).sqrt();
    let weighted = grad
        .iter()
        .zip(gamma)
        .zip(x)
        .map(|((&g, &w), &v)| g * w * v)
        .sum::<f32>();
    let dx = grad
        .iter()
        .zip(gamma)
        .zip(x)
        .map(|((&g, &w), &v)| g * w / rms - v * weighted / (n as f32 * rms * rms * rms))
        .collect();
    let dgamma = grad.iter().zip(x).map(|(&g, &v)| g * v / rms).collect();
    Ok((dx, dgamma))
}

/// `y = R(θ)·x`：旋转矩阵正交，反向即转置 `R(−θ)`。
fn rope(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![crate::kernel::rope_grad(&ctx.inputs[0], ctx.incoming)])
}

/// 稠密注意力：查询与历史行数相同。
fn attention(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let d = nn::conv::attention::MODEL.shape.1;
    let rows = rows_for_width(ctx.inputs[0].len(), d, "attention_f32")?;
    Ok(attention_grad(attention_ctx(&ctx, d), rows, rows))
}

/// 潜空间注意力：读写宽解耦，写宽由上游梯度与查询行数反推。
fn latent_attention(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let w = nn::conv::attention::latent::MODEL.shape.1;
    let rows = rows_for_width(ctx.inputs[0].len(), w, "latent_attention_f32")?;
    let hist = rows_for_width(ctx.inputs[1].len(), w, "latent_attention_f32")?;
    let mut view = attention_ctx(&ctx, w);
    view.write = ctx.incoming.len() / rows.max(1);
    Ok(attention_grad(view, rows, hist))
}

/// decode 注意力：查询行数与历史行数解耦。
fn decode_attention(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let d = nn::conv::decode_attention::MODEL.shape.1;
    let rows = rows_for_width(ctx.inputs[0].len(), d, "decode_attention_f32")?;
    let hist = rows_for_width(ctx.inputs[1].len(), d, "decode_attention_f32")?;
    Ok(attention_grad(attention_ctx(&ctx, d), rows, hist))
}

/// 长度按行宽整除并返回行数；零宽或余数都是装配错误。
fn rows_for_width(len: usize, width: usize, op: &str) -> Result<usize> {
    validation_rules!(SITE;
        width != 0 && len.is_multiple_of(width) => format!("`{op}` 长度 {len} 不能按行宽 {width} 整除"),
    );
    Ok(len / width)
}

/// `W[n,k]` 的行主序形状：列宽已知，行数由权重长度推出。
fn matrix_shape(weight: &[f32], cols: usize) -> Result<(usize, usize)> {
    rows_for_width(weight.len(), cols, "权重").map(|rows| (rows, cols))
}

/// 把一段梯度整块累加进目标槽；长度已由调用方对齐。
fn accumulate_into(dst: &mut [f32], offset: usize, source: &[f32]) -> Result<()> {
    let Some(target) = dst.get_mut(offset..offset + source.len()) else {
        return Err(Error::new(
            Code::Internal,
            format!("梯度累加偏移 {offset} 越界（目标 {} 元素）", dst.len()),
        )
        .with(SITE));
    };
    for (slot, value) in target.iter_mut().zip(source) {
        *slot += value;
    }
    Ok(())
}

/// GELU 导数：`y = 0.5·x·(1+erf(x/√2))`，`y′ = Φ(x) + x·φ(x)`。
fn gelu_grad(x: f32) -> f32 {
    let normal = (-(x * x) / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
    let erf_term = 0.5 * (1.0 + nn::gelu_f32::erf(x * std::f32::consts::FRAC_1_SQRT_2));
    erf_term + x * normal
}

// ---- 注意力反向公式：自注意力 / 潜空间 / decode 三种前向共享的逐行反向本体 ----

/// 由公式上下文拼出注意力反向的只读视野。
fn attention_ctx<'a>(ctx: &Ctx<'a>, d: usize) -> AttentionCtx<'a> {
    AttentionCtx {
        q: &ctx.inputs[0],
        k: &ctx.inputs[1],
        v: &ctx.inputs[2],
        dout: ctx.incoming,
        d,
        write: d,
        scale: 1.0 / (d as f32).sqrt(),
    }
}

/// 注意力反向的只读视野：三个输入的前向值与输出上游梯度。
struct AttentionCtx<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    dout: &'a [f32],
    d: usize,
    /// 写宽：`V` / `dO` / `dV` 的列宽；稠密注意力与读宽相同，潜空间注意力更窄。
    write: usize,
    scale: f32,
}

/// 一条查询行的反向中间量：softmax 概率与分数梯度（`√d` 缩放前的口径）。
struct AttentionRow {
    probs: Vec<f32>,
    ds: Vec<f32>,
}

/// 注意力反向本体：逐查询行算概率与分数梯度，再散射回 `dq` / `dk` / `dv`。
fn attention_grad(ctx: AttentionCtx<'_>, q_rows: usize, hist: usize) -> Vec<Vec<f32>> {
    let mut dq = vec![0.0; q_rows * ctx.d];
    let mut dk = vec![0.0; hist * ctx.d];
    let mut dv = vec![0.0; hist * ctx.write];
    for i in 0..q_rows {
        let row = attention_row(&ctx, i, hist);
        for j in 0..hist {
            let s = row.ds[j] * ctx.scale;
            for h in 0..ctx.write {
                dv[j * ctx.write + h] += row.probs[j] * ctx.dout[i * ctx.write + h];
            }
            for h in 0..ctx.d {
                dq[i * ctx.d + h] += s * ctx.k[j * ctx.d + h];
                dk[j * ctx.d + h] += s * ctx.q[i * ctx.d + h];
            }
        }
    }
    vec![dq, dk, dv]
}

/// 一条查询行：`p = softmax(QKᵀ/√d)`、`dP = dO·V`、`dS = p·(dP − Σ p·dP)`。
fn attention_row(ctx: &AttentionCtx<'_>, i: usize, hist: usize) -> AttentionRow {
    let probs = probabilities(ctx.q, ctx.k, i, hist, ctx.d, ctx.scale);
    let dp = output_grad_row(ctx.dout, ctx.v, i, hist, ctx.write);
    let ds = score_grad_row(&probs, &dp);
    AttentionRow { probs, ds }
}

/// 数值稳定的 softmax 概率：先减行内最大分数再取指数，与前向同口径。
fn probabilities(q: &[f32], k: &[f32], i: usize, hist: usize, d: usize, scale: f32) -> Vec<f32> {
    // 分数缓冲原地转成概率：三条 `Vec` 变一条，取指/求和顺序不变，逐位与分流版一致。
    let mut probs = score_row(q, k, i, hist, d, scale);
    let max = probs.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0f32;
    for slot in &mut probs {
        *slot = (*slot - max).exp();
        sum += *slot;
    }
    for slot in &mut probs {
        *slot /= sum;
    }
    probs
}

/// 行 `idx` 的切片（行宽 `d`）：行起始地址只算一次，内层循环不再带下标乘法。
fn row(values: &[f32], idx: usize, d: usize) -> &[f32] {
    &values[idx * d..(idx + 1) * d]
}

/// 两条等长行的点积；`zip` 让边界检查落在行首一次，累加顺序与逐元素下标版一致。
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 一行分数 `S[i,j] = q[i]·k[j]/√d`。
fn score_row(q: &[f32], k: &[f32], i: usize, hist: usize, d: usize, scale: f32) -> Vec<f32> {
    let qrow = row(q, i, d);
    (0..hist).map(|j| dot(qrow, row(k, j, d)) * scale).collect()
}

/// `dP[i,j] = Σ_h dO[i,h]·v[j,h]`，`h` 走写宽（`V` 的列宽）。
fn output_grad_row(dout: &[f32], v: &[f32], i: usize, hist: usize, d: usize) -> Vec<f32> {
    let drow = row(dout, i, d);
    (0..hist).map(|j| dot(drow, row(v, j, d))).collect()
}

/// softmax 反向：`dS[i,j] = p[i,j]·(dP[i,j] − Σ_j p[i,j]·dP[i,j])`。
fn score_grad_row(probs: &[f32], dp: &[f32]) -> Vec<f32> {
    let weighted = dp.iter().zip(probs).map(|(d, p)| d * p).sum::<f32>();
    dp.iter()
        .zip(probs)
        .map(|(d, p)| p * (d - weighted))
        .collect()
}

#[cfg(test)]
#[path = "../tests/backprop_tests.rs"]
mod tests;
