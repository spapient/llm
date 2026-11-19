//! 反向数值执行器：按 [`Schedule`] 的逆拓扑序把 loss 梯度回传到参数终点。
//!
//! [`autograd`] 只产出图结构、反向拓扑序与扇出账本，不含梯度数值；本模块把算子的解析反向
//! 公式落到真实 f32 缓冲：从 loss 的上游梯度 `1` 出发，逐节点把 incoming 梯度按局部雅可比
//! 分给各输入并累加，每收满一份贡献就结算一步。
//!
//! 反向公式表与前向内核表同构：算子名、元数与派生几何都在 [`FORMULAS`] 闭集里登记，未接
//! 公式的算子遇到即返回 [`Code::Unsupported`]，执行循环不逐名字 `match` 散落分支。[`FORMULAS`]
//! 与 [`op::REGISTRY`] 的 `differentiable` 标记在编译期互锁：可微算子缺公式、公式指向不可微
//! 算子、或公式重复登记都是编译错误。量化权重 / 反量化算子（`gemv_i*` / `gemm_i*` /
//! `dequant_i*`）的权重输入是 u8 码，属推理期解码，不参与训练反向，保持 [`Code::Unsupported`]。

use autograd::{Node, Schedule, Tape, Var};
use diagnostic::{Code, Error, Result};

use crate::kernel::{Kernel, same_str};
use crate::pool::Buffers;

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::backprop";

/// 按已规划的 [`Schedule`] 回传梯度：起点 loss 梯度为全 `1`，逐节点分配到图上输入。
///
/// 梯度累加进 `buffers` 的梯度槽；图外（需求为无梯度）输入不收账。
///
/// # Errors
///
/// 前向值/上游梯度不在场、账本顺序不符，或遇到尚未接反向公式的算子时返回对应 [`Code`]。
pub(crate) fn backprop<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    schedule: &Schedule,
) -> Result<()> {
    seed_loss(buffers, schedule.loss())?;
    let mut account = schedule.account();
    for step in schedule.steps() {
        reverse_step(buffers, tape, step)?;
        account.settle(step).map_err(|error| error.with(SITE))?;
    }
    Ok(())
}

/// 单节点反向：取上游梯度、装配公式上下文，查表执行并把逐输入贡献喂回梯度槽。
fn reverse_step<K: Kernel>(buffers: &mut Buffers<K>, tape: &Tape, step: &Node) -> Result<()> {
    let incoming = incoming(buffers, step)?.to_vec();
    let input_values = collect_inputs(buffers, step)?;
    let output = values(buffers, step.output())?.to_vec();
    let context = Context {
        incoming: &incoming,
        inputs: &input_values,
        output: &output,
    };
    apply_formula(buffers, tape, step, context)?;
    Ok(())
}

/// 把各输入的前向值整块拷出只读快照；任一前向值不在场即回缺失。
fn collect_inputs<K: Kernel>(buffers: &Buffers<K>, step: &Node) -> Result<Vec<Vec<f32>>> {
    step.inputs()
        .iter()
        .map(|&input| values(buffers, input).map(|value| value.to_vec()))
        .collect()
}

/// 按算子名查公式目录执行，把返回的逐输入贡献喂回对应梯度槽；未登记公式即拒绝。
fn apply_formula<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    step: &Node,
    context: Context<'_>,
) -> Result<()> {
    let formula = lookup(step.op())?;
    let contributions = formula(context)?;
    for (&input, contrib) in step.inputs().iter().zip(contributions) {
        feed(buffers, tape, input, contrib)?;
    }
    Ok(())
}

/// 记一份梯度贡献；图外输入没有梯度缓冲，直接丢弃。
fn feed<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    var: Var,
    contrib: Vec<f32>,
) -> Result<()> {
    let receives = tape
        .requirement(var)
        .is_some_and(|requirement| !requirement.is_none());
    if receives {
        buffers
            .accumulate_gradient(var, &contrib)
            .map_err(|error| error.with(SITE))?;
    }
    Ok(())
}

/// loss 起点梯度：每个前向元素都从上游 `1` 出发。
fn seed_loss<K: Kernel>(buffers: &mut Buffers<K>, loss: Var) -> Result<()> {
    let width = values(buffers, loss)?.len();
    buffers
        .load_gradient(loss, &vec![1.0; width])
        .map_err(|error| error.with(SITE))?;
    Ok(())
}

/// 读变量在场的前向 f32 值；缺失是重放/装配口径问题。
fn values<K: Kernel>(buffers: &Buffers<K>, var: Var) -> Result<&[f32]> {
    buffers
        .value(var)
        .and_then(|view| view.f32())
        .ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!("变量 {} 的前向值不在场，无法反传", var.id()),
            )
            .with(SITE)
        })
}

/// 读节点输出已就位的上游梯度；缺失说明拓扑序上有前驱没回传到。
fn incoming<'a, K: Kernel>(buffers: &'a Buffers<K>, step: &Node) -> Result<&'a [f32]> {
    buffers.gradient(step.output()).ok_or_else(|| {
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
struct Context<'a> {
    incoming: &'a [f32],
    inputs: &'a [Vec<f32>],
    output: &'a [f32],
}

/// 反向公式：按上下文算出与输入等长的逐元素贡献，顺序与节点输入对齐。
type Formula = fn(Context<'_>) -> Result<Vec<Vec<f32>>>;

/// 一条公式目录项：算子名与其解析反向公式。
struct Entry {
    op: &'static str,
    formula: Formula,
}

/// 已接反向公式的可微算子目录；其余算子（量化权重 / 反量化）在 [`apply_formula`] 处回
/// [`Code::Unsupported`]。与 [`op::REGISTRY`] 的 `differentiable` 标记一一对应，由下方 `const`
/// 断言在编译期守住。
const FORMULAS: [Entry; 24] = [
    Entry {
        op: "add_f32",
        formula: passthrough,
    },
    Entry {
        op: "residual_add_f32",
        formula: passthrough,
    },
    Entry {
        op: "mul_f32",
        formula: mul,
    },
    Entry {
        op: "scale_f32",
        formula: scale,
    },
    Entry {
        op: "sub_f32",
        formula: sub,
    },
    Entry {
        op: "div_f32",
        formula: div,
    },
    Entry {
        op: "exp_f32",
        formula: exp,
    },
    Entry {
        op: "log_f32",
        formula: log,
    },
    Entry {
        op: "min_f32",
        formula: min,
    },
    Entry {
        op: "max_f32",
        formula: max,
    },
    Entry {
        op: "silu_f32",
        formula: silu,
    },
    Entry {
        op: "gelu_f32",
        formula: gelu,
    },
    Entry {
        op: "reduce_sum_f32",
        formula: reduce_sum,
    },
    Entry {
        op: "reduce_max_f32",
        formula: reduce_max,
    },
    Entry {
        op: "softmax_f32",
        formula: softmax,
    },
    Entry {
        op: "gemv_f32",
        formula: gemv,
    },
    Entry {
        op: "gemm_f32",
        formula: gemm,
    },
    Entry {
        op: "embed_f32",
        formula: embed,
    },
    Entry {
        op: "rmsnorm_f32",
        formula: rmsnorm,
    },
    Entry {
        op: "rope_f32",
        formula: rope,
    },
    Entry {
        op: "attention_f32",
        formula: attention,
    },
    Entry {
        op: "flash_attention_f32",
        formula: attention,
    },
    Entry {
        op: "decode_attention_f32",
        formula: decode_attention,
    },
    Entry {
        op: "decode_flash_attention_f32",
        formula: decode_attention,
    },
];

/// 编译期闭合：公式目录与 [`op::REGISTRY`] 的 `differentiable` 标记一一对应。
///
/// - 每个标记为可微的算子都有一条公式；
/// - 每条公式都指向一个可微且已注册的算子；
/// - 公式目录里没有重复算子名。
const _: () = formulas_cover(&FORMULAS, &op::REGISTRY);

/// 双向互锁：可微算子缺公式、公式指向不可微/未注册算子、或公式重复，都是编译错误。
const fn formulas_cover(formulas: &[Entry], registry: &[op::Registration]) {
    let mut i = 0;
    while i < registry.len() {
        assert!(
            !registry[i].differentiable || has_formula(formulas, registry[i].name),
            "存在标记为可微却缺少反向公式的算子"
        );
        i += 1;
    }
    let mut j = 0;
    while j < formulas.len() {
        assert!(
            differentiable(registry, formulas[j].op),
            "存在反向公式指向不可微或未注册的算子"
        );
        assert!(!duplicate(formulas, j), "存在重复登记的反向公式");
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
const fn differentiable(registry: &[op::Registration], name: &str) -> bool {
    let mut i = 0;
    while i < registry.len() {
        if same_str(registry[i].name, name) {
            return registry[i].differentiable;
        }
        i += 1;
    }
    false
}

/// 公式目录第 `at` 条之前是否已登记同名算子。
const fn duplicate(formulas: &[Entry], at: usize) -> bool {
    let mut i = 0;
    while i < at {
        if same_str(formulas[i].op, formulas[at].op) {
            return true;
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

/// y = a + b：两条路径都原样回传上游梯度。
fn passthrough(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![ctx.incoming.to_vec(); ctx.inputs.len()])
}

/// y = a·b：dA = in·b，dB = in·a。
fn mul(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let da = ctx.incoming.iter().zip(b).map(|(&g, w)| g * w).collect();
    let db = ctx.incoming.iter().zip(a).map(|(&g, w)| g * w).collect();
    Ok(vec![da, db])
}

/// y = α·a：dA = α·in。
fn scale(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let alpha = op::scale_f32::ALPHA;
    let da = ctx.incoming.iter().map(|&g| g * alpha).collect();
    Ok(vec![da])
}

/// y = a−b：dA = in，dB = −in。
fn sub(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx.incoming.to_vec();
    let db = ctx.incoming.iter().map(|&g| -g).collect();
    Ok(vec![da, db])
}

/// y = a/b：dA = in/b，dB = −in·a/b²。
fn div(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
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

/// y = exp(a)：dA = in·y（输出 y 即 exp(a)，避免重算）。
fn exp(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(ctx.output)
        .map(|(&g, &y)| g * y)
        .collect();
    Ok(vec![da])
}

/// y = ln(a)：dA = in/a。
fn log(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(&ctx.inputs[0])
        .map(|(&g, &x)| g / x)
        .collect();
    Ok(vec![da])
}

/// y = min(a,b)：梯度给较小者；相等给 `a`（避免双计）。
fn min(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let da = ctx
        .incoming
        .iter()
        .zip(a)
        .zip(b)
        .map(|((&g, &x), &y)| if x <= y { g } else { 0.0 })
        .collect();
    let db = ctx
        .incoming
        .iter()
        .zip(a)
        .zip(b)
        .map(|((&g, &x), &y)| if y < x { g } else { 0.0 })
        .collect();
    Ok(vec![da, db])
}

/// y = max(a,b)：梯度给较大者；相等给 `a`（避免双计）。
fn max(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let da = ctx
        .incoming
        .iter()
        .zip(a)
        .zip(b)
        .map(|((&g, &x), &y)| if x >= y { g } else { 0.0 })
        .collect();
    let db = ctx
        .incoming
        .iter()
        .zip(a)
        .zip(b)
        .map(|((&g, &x), &y)| if y > x { g } else { 0.0 })
        .collect();
    Ok(vec![da, db])
}

/// y = x·σ(x)：y′ = σ + x·σ(1−σ)，dA = in·y′。
fn silu(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let x = &ctx.inputs[0];
    let da = ctx
        .incoming
        .iter()
        .zip(x)
        .map(|(&g, &v)| {
            let sigmoid = 1.0 / (1.0 + (-v).exp());
            g * (sigmoid + v * sigmoid * (1.0 - sigmoid))
        })
        .collect();
    Ok(vec![da])
}

/// y = 0.5·x·(1 + erf(x/√2))：y′ = Φ(x) + x·φ(x)，φ 为标准正态密度、Φ 为累积分布。
fn gelu(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let x = &ctx.inputs[0];
    let da = ctx
        .incoming
        .iter()
        .zip(x)
        .map(|(&g, &v)| g * gelu_grad(v))
        .collect();
    Ok(vec![da])
}

/// y = Σa：上游标量梯度广播回每个元素，dA[i] = in[0]。
fn reduce_sum(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = vec![ctx.incoming[0]; ctx.inputs[0].len()];
    Ok(vec![da])
}

/// y = max(a)：只有最大值元素收到 `in[0]`，其余梯度为 0。
fn reduce_max(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let peak = a.iter().copied().fold(f32::MIN, f32::max);
    let da = a
        .iter()
        .map(|&v| if v == peak { ctx.incoming[0] } else { 0.0 })
        .collect();
    Ok(vec![da])
}

/// p = softmax(x)：dA[i] = p[i]·(in[i] − Σ_j in[j]·p[j])。
fn softmax(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
fn gemv(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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

/// `Y = A·B`：返回 `[dA, dB]`，收缩轴固定 `op::gemm::K`。
fn gemm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0][..];
    let b = &ctx.inputs[1][..];
    let grad = &ctx.incoming;
    let k = op::gemm::K;
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
fn embed(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let index = &ctx.inputs[0][..];
    let table = &ctx.inputs[1][..];
    let dim = ctx.incoming.len() / index.len();
    let mut dtable = vec![0.0; table.len()];
    for (token, &entry) in index.iter().enumerate() {
        let row = entry as usize * dim;
        let grad_row = &ctx.incoming[token * dim..][..dim];
        accumulate_into(&mut dtable, row, grad_row)?;
    }
    // 索引输入无雅可比：贡献置空，只回传表梯度。
    Ok(vec![vec![0.0; index.len()], dtable])
}

/// 训练内核 `rope_f32` 的固定位置：与 `kernel.rs` 的 `rope_f32` 同口径（标量槽尚未接进训练内核）。
const ROPE_POS: f32 = 7.0;

/// `y = γ·x / rms(x)`：`dx[j] = g[j]·γ[j]/r − x[j]·Σ_i g[i]·γ[i]·x[i] / (n·r³)`，
/// `dγ[i] = g[i]·x[i]/r`。
fn rmsnorm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
    let rms = (mean_sq + op::rmsnorm_f32::EPS).sqrt();
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
fn rope(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![rope_grad(&ctx.inputs[0], ctx.incoming)])
}

/// RoPE 反向本体：逐对把上游梯度按同一角度旋转回去；末尾不成对的元素原样透传。
fn rope_grad(x: &[f32], grad: &[f32]) -> Vec<f32> {
    let d = x.len();
    let pairs = d / 2;
    let mut dx = vec![0.0; d];
    for i in 0..pairs {
        let angle = ROPE_POS * op::rope_f32::BASE.powf(-(2.0 * i as f32) / d as f32);
        let (sin, cos) = angle.sin_cos();
        dx[2 * i] = grad[2 * i] * cos + grad[2 * i + 1] * sin;
        dx[2 * i + 1] = -grad[2 * i] * sin + grad[2 * i + 1] * cos;
    }
    if d % 2 == 1 {
        dx[d - 1] = grad[d - 1];
    }
    dx
}

/// 自注意力（`attention_f32` / `flash_attention_f32` 共用）：返回 `[dq, dk, dv]`。
fn attention(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let d = op::attention::MODEL.shape.1;
    let s = divide(ctx.inputs[0].len(), d, "attention_f32")?;
    let (dq, dk, dv) = attention_grad(attention_ctx(&ctx, d), s, s);
    Ok(vec![dq, dk, dv])
}

/// decode 注意力（`decode_attention_f32` / `decode_flash_attention_f32` 共用）：查询行数与历史
/// 行数解耦，返回 `[dq, dk, dv]`。
fn decode_attention(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = divide(ctx.inputs[0].len(), d, "decode_attention_f32")?;
    let hist = divide(ctx.inputs[1].len(), d, "decode_attention_f32")?;
    let (dq, dk, dv) = attention_grad(attention_ctx(&ctx, d), q_rows, hist);
    Ok(vec![dq, dk, dv])
}

/// 由公式上下文拼出注意力反向的只读视野。
fn attention_ctx<'a>(ctx: &Context<'a>, d: usize) -> AttentionCtx<'a> {
    AttentionCtx {
        q: &ctx.inputs[0],
        k: &ctx.inputs[1],
        v: &ctx.inputs[2],
        dout: ctx.incoming,
        d,
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
    scale: f32,
}

/// 一条查询行的反向中间量：softmax 概率与分数梯度（`√d` 缩放前的口径）。
struct AttentionRow {
    probs: Vec<f32>,
    ds: Vec<f32>,
}

/// 注意力反向本体：逐查询行算概率与分数梯度，再散射回 `dq` / `dk` / `dv`。
fn attention_grad(
    ctx: AttentionCtx<'_>,
    q_rows: usize,
    hist: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0; q_rows * ctx.d];
    let mut dk = vec![0.0; hist * ctx.d];
    let mut dv = vec![0.0; hist * ctx.d];
    for i in 0..q_rows {
        let row = attention_row(&ctx, i, hist);
        scatter(&row, &ctx, i, hist, &mut dq, &mut dk, &mut dv);
    }
    (dq, dk, dv)
}

/// 一条查询行：`p = softmax(QKᵀ/√d)`、`dP = dO·V`、`dS = p·(dP − Σ p·dP)`。
fn attention_row(ctx: &AttentionCtx<'_>, i: usize, hist: usize) -> AttentionRow {
    let probs = probabilities(ctx.q, ctx.k, i, hist, ctx.d, ctx.scale);
    let dp = output_grad_row(ctx.dout, ctx.v, i, hist, ctx.d);
    let ds = score_grad_row(&probs, &dp);
    AttentionRow { probs, ds }
}

/// 数值稳定的 softmax 概率：先减行内最大分数再取指数，与前向同口径。
fn probabilities(q: &[f32], k: &[f32], i: usize, hist: usize, d: usize, scale: f32) -> Vec<f32> {
    let scores = score_row(q, k, i, hist, d, scale);
    let max = scores.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// 一行分数 `S[i,j] = q[i]·k[j]/√d`。
fn score_row(q: &[f32], k: &[f32], i: usize, hist: usize, d: usize, scale: f32) -> Vec<f32> {
    let mut scores = vec![0.0; hist];
    for (j, slot) in scores.iter_mut().enumerate() {
        *slot = dot_rows(q, k, i, j, d) * scale;
    }
    scores
}

/// 两行向量的点积：`a[i]·b[j]`，均按行宽 `d` 的行主序展开。
fn dot_rows(a: &[f32], b: &[f32], i: usize, j: usize, d: usize) -> f32 {
    let mut sum = 0.0;
    for h in 0..d {
        sum += a[i * d + h] * b[j * d + h];
    }
    sum
}

/// `dP[i,j] = Σ_h dO[i,h]·v[j,h]`。
fn output_grad_row(dout: &[f32], v: &[f32], i: usize, hist: usize, d: usize) -> Vec<f32> {
    let mut dp = vec![0.0; hist];
    for (j, slot) in dp.iter_mut().enumerate() {
        *slot = dot_rows(dout, v, i, j, d);
    }
    dp
}

/// softmax 反向：`dS[i,j] = p[i,j]·(dP[i,j] − Σ_j p[i,j]·dP[i,j])`。
fn score_grad_row(probs: &[f32], dp: &[f32]) -> Vec<f32> {
    let weighted = dp.iter().zip(probs).map(|(d, p)| d * p).sum::<f32>();
    dp.iter()
        .zip(probs)
        .map(|(d, p)| p * (d - weighted))
        .collect()
}

/// 把一条查询行的贡献散射进 `dq` / `dk` / `dv`。
fn scatter(
    row: &AttentionRow,
    ctx: &AttentionCtx<'_>,
    i: usize,
    hist: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) {
    for j in 0..hist {
        accumulate_row(row, ctx, i, j, dq, dk, dv);
    }
}

/// 单点散射：`dV += p·dO`，`dQ += dS·k/√d`，`dK += dS·q/√d`。
fn accumulate_row(
    row: &AttentionRow,
    ctx: &AttentionCtx<'_>,
    i: usize,
    j: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) {
    let s = row.ds[j] * ctx.scale;
    for h in 0..ctx.d {
        dv[j * ctx.d + h] += row.probs[j] * ctx.dout[i * ctx.d + h];
        dq[i * ctx.d + h] += s * ctx.k[j * ctx.d + h];
        dk[j * ctx.d + h] += s * ctx.q[i * ctx.d + h];
    }
}

/// 行数派生：长度必须能被头宽整除，否则是装配口径问题。
fn divide(len: usize, d: usize, op: &str) -> Result<usize> {
    if d == 0 || !len.is_multiple_of(d) {
        return Err(Error::new(
            Code::Invalid,
            format!("`{op}` 长度 {len} 不能按头宽 {d} 整除"),
        )
        .with(SITE));
    }
    Ok(len / d)
}

/// `W[n,k]` 的行主序形状：`k` 由输入宽度给定，`n` 由权重长度整除推出；不整除是装配口径问题。
fn matrix_shape(weight: &[f32], cols: usize) -> Result<(usize, usize)> {
    if cols == 0 || !weight.len().is_multiple_of(cols) {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {} 不能按输入宽度 {cols} 整除出行数", weight.len()),
        )
        .with(SITE));
    }
    Ok((weight.len() / cols, cols))
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
    let erf_term = 0.5 * (1.0 + op::gelu_f32::erf(x * std::f32::consts::FRAC_1_SQRT_2));
    erf_term + x * normal
}

#[cfg(test)]
mod tests {
//! 反向数值对拍：自动回算的解析梯度与中心差分逐参数比对，并核对图外叶子不收梯度。

use crate::{Data, Reference, Session, Sgd, TrainConfig};
use autograd::{Sizes, Store, Tape, Var};
use diagnostic::{Code, Result};
use nn::ParamRole;
use tensor::{DType, Shape};

/// 夹具向量宽度。
const N: usize = 3;
/// 中心差分步长：f32 下在舍入与截断误差之间。
const STEP: f32 = 1e-4;
/// 绝对容差（与 autograd gradcheck 同口径）。
const ABSOLUTE: f32 = 2e-3;
/// 相对容差：大梯度按比例放宽。
const RELATIVE: f32 = 0.05;

fn config() -> TrainConfig {
    TrainConfig::default()
}

/// 把最终输出 `y` 归约成标量 `loss` 后 plan + backward：链式损失测试的公共收尾。
fn sum_backward(session: &mut Session<Reference, Sgd>, y: Var, loss: Var) -> Result<()> {
    session.buffers().compute("reduce_sum_f32", &[y], loss)?;
    session.plan(loss)?;
    session.backward()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 单行 softmax（与 `op::softmax_f32` 同口径：整缓冲当作一行，先减 max 再 exp）。
fn softmax_row(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    let exps: Vec<f32> = x
        .iter()
        .map(|&v| {
            let e = (v - max).exp();
            sum += e;
            e
        })
        .collect();
    exps.into_iter().map(|e| e / sum).collect()
}

/// 单行 KL：`sum_i p_s[i] · (ln p_s[i] - ln p_t[i])`（温度 T=1，教师冻结）。
fn kl_single_row(s: &[f32], t: &[f32]) -> f32 {
    let ps = softmax_row(s);
    let pt = softmax_row(t);
    ps.iter().zip(&pt).map(|(p, q)| p * (p.ln() - q.ln())).sum()
}

/// 数值裁剪：`x` 夹到 `[lo, hi]`。
fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

/// 裁剪代理和（未取负号，负号只是常数因子）：`Σ min(ratio·A, clamp(ratio,1±ε)·A)`。
fn clipped_surrogate_sum(new_logp: &[f32], old_logp: &[f32], adv: &[f32], eps: f32) -> f32 {
    let lo = 1.0 - eps;
    let hi = 1.0 + eps;
    new_logp
        .iter()
        .zip(old_logp)
        .zip(adv)
        .map(|((&n, &o), &a)| {
            let ratio = (n - o).exp();
            let clipped = clampf(ratio, lo, hi);
            (ratio * a).min(clipped * a)
        })
        .sum()
}

fn close(actual: f32, numeric: f32) -> bool {
    let slack = ABSOLUTE + RELATIVE * numeric.abs().max(1.0);
    (actual - numeric).abs() <= slack
}

/// `loss = reduce_sum(mul(p, w))`：d loss/d p[i] = w[i]，乘法与归约两段链式回传。
#[test]
fn mul_then_reduce_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let w = tape.leaf();
    let z = tape.record("mul_f32", &[p, w]).expect("记录 z");
    let loss = tape.record("reduce_sum_f32", &[z]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(w, Shape::new([N]), DType::F32);
    sizes.set(z, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let p0 = vec![0.3_f32, -0.5, 0.8];
    let w0 = vec![1.0_f32, -0.2, 0.6];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().load(w, Data::F32(w0.clone()))?;
    session.buffers().compute("mul_f32", &[p, w], z)?;
    sum_backward(&mut session, z, loss)?;

    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        |x| dot(x, &w0),
    );
    Ok(())
}

/// `loss = reduce_sum(silu(scale(p)))`：缩放与 SiLU 激活两段链式对拍。
#[test]
fn scale_silu_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("scale_f32", &[p]).expect("记录 h");
    let y = tape.record("silu_f32", &[h]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(h, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let p0 = vec![-0.6_f32, 0.4, 1.2];
    let chain_loss = |params: &[f32]| params.iter().map(|&x| silu(0.5 * x)).sum::<f32>();

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().compute("scale_f32", &[p], h)?;
    session.buffers().compute("silu_f32", &[h], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// 图外常量叶子不收梯度：add(p, x) 中 x 是 detached 叶子，梯度槽保持空；参数 p 收到上游 1。
#[test]
fn detached_leaf_receives_no_gradient() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let x = tape.leaf();
    let loss = tape.record("add_f32", &[p, x]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.buffers().load(x, Data::F32(vec![0.1; N]))?;
    session.buffers().compute("add_f32", &[p, x], loss)?;
    session.plan(loss)?;
    session.backward()?;

    assert!(
        session.buffers().gradient(x).is_none(),
        "图外叶子 x 不该有梯度槽"
    );
    assert_eq!(
        session.buffers().gradient(p),
        Some(&[1.0; N][..]),
        "add 把上游 1 原样传给参数 p"
    );
    Ok(())
}

/// `loss = reduce_sum(gelu(p))`：GELU 导数（正态密度 + 乘 Φ）逐元素对拍。
#[test]
fn gelu_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("gelu_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(h, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let p0 = vec![-1.2_f32, 0.4, 1.1];
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| 0.5 * x * (1.0 + op::gelu_f32::erf(x * std::f32::consts::FRAC_1_SQRT_2)))
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().compute("gelu_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// `loss = reduce_max(p)`：最大值元素收到上游 1，其余为 0；中心差分核对梯度口径。
#[test]
fn reduce_max_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("reduce_max_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(h, Shape::new([1]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let p0 = vec![-0.7_f32, 1.3, 0.2];
    let chain_loss = |params: &[f32]| params.iter().copied().fold(f32::MIN, f32::max);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().compute("reduce_max_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// `loss = reduce_sum(gemv(x, w))`：权重 `dW[n,k]=x[k]·in[n]`，输入 `dx[k]=Σ_n W[n,k]·in[n]`。
#[test]
fn gemv_matches_central_differences() -> Result<()> {
    let rows = 2usize;
    let cols = 3usize;
    let mut tape = Tape::new();
    let w = tape.parameter();
    let x = tape.leaf();
    // gemv 反向按输入宽度从权重长度反推行数：权重在前、输入在后与前向调用对齐。
    let y = tape.record("gemv_f32", &[w, x]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(w, Shape::new([rows * cols]), DType::F32);
    sizes.set(x, Shape::new([cols]), DType::F32);
    sizes.set(y, Shape::new([rows]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let w0: Vec<f32> = (0..rows * cols).map(|i| 0.1 * i as f32 - 0.5).collect();
    let x0 = vec![0.3_f32, -0.2, 0.8];
    let chain_loss = |w: &[f32], x: &[f32]| {
        (0..rows)
            .map(|n| (0..cols).map(|k| w[n * cols + k] * x[k]).sum::<f32>())
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(w, ParamRole::Trainable, w0.clone())?;
    session.buffers().load(x, Data::F32(x0.clone()))?;
    session.buffers().compute("gemv_f32", &[w, x], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(w).expect("权重梯度应在场"),
        &w0,
        "w",
        |w| chain_loss(w, &x0),
    );
    Ok(())
}

/// `loss = reduce_sum(gemm(a, b))`：收缩轴固定 `op::gemm::K`，对拍 `dA` / `dB`。
#[test]
fn gemm_matches_central_differences() -> Result<()> {
    let k = op::gemm::K;
    let m = 1usize;
    let n = 2usize;
    let mut tape = Tape::new();
    let a = tape.parameter();
    let b = tape.leaf();
    let y = tape.record("gemm_f32", &[a, b]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(a, Shape::new([m * k]), DType::F32);
    sizes.set(b, Shape::new([k * n]), DType::F32);
    sizes.set(y, Shape::new([m * n]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let a0: Vec<f32> = (0..m * k).map(|i| 0.1 * i as f32 - 0.3).collect();
    let b0: Vec<f32> = (0..k * n).map(|i| 0.08 * i as f32 - 0.4).collect();
    let chain_loss = |a: &[f32], b: &[f32]| {
        (0..m)
            .flat_map(|i| {
                (0..n).map(move |j| (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum::<f32>())
            })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(a, ParamRole::Trainable, a0.clone())?;
    session.buffers().load(b, Data::F32(b0.clone()))?;
    session.buffers().compute("gemm_f32", &[a, b], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(a).expect("A 梯度应在场"),
        &a0,
        "a",
        |a| chain_loss(a, &b0),
    );
    Ok(())
}

/// `loss = reduce_sum(embed(idx, table))`：被访问行累加上游梯度；表行不访问则保持 0。
#[test]
fn embed_table_gradient_matches_central_differences() -> Result<()> {
    let tokens = 2usize;
    let dim = 2usize;
    let rows = 3usize;
    let mut tape = Tape::new();
    let table = tape.parameter();
    let idx = tape.leaf();
    let y = tape.record("embed_f32", &[idx, table]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(table, Shape::new([rows * dim]), DType::F32);
    sizes.set(idx, Shape::new([tokens]), DType::F32);
    sizes.set(y, Shape::new([tokens * dim]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let table0: Vec<f32> = (0..rows * dim).map(|i| 0.2 * i as f32 - 0.6).collect();
    let idx0 = vec![0.0_f32, 2.0];
    let chain_loss = |table: &[f32]| {
        (0..tokens)
            .flat_map(|t| {
                let row = idx0[t] as usize * dim;
                (0..dim).map(move |d| table[row + d])
            })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(table, ParamRole::Trainable, table0.clone())?;
    session.buffers().load(idx, Data::F32(idx0.clone()))?;
    session.buffers().compute("embed_f32", &[idx, table], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(table).expect("表梯度应在场"),
        &table0,
        "table",
        chain_loss,
    );
    Ok(())
}

/// 不可微算子在回传时显式报 [`Code::Unsupported`]，不静默忽略。反量化是推理期解码，
/// 权重/码值无训练梯度，因此不登记反向公式。
#[test]
fn unsupported_operator_backward_is_rejected() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record("dequant_i8", &[p]).expect("记录 y");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.buffers().compute("dequant_i8", &[p], y)?;
    session.plan(y)?;

    let error = session.backward().expect_err("反量化没有解析反向");
    assert_eq!(error.code(), Code::Unsupported);
    Ok(())
}

/// 中心差分：把 `base[index]` 扰动 ±STEP，用闭式 `f` 重算损失。
fn central(f: impl Fn(&[f32]) -> f32, base: &[f32], index: usize) -> f32 {
    let mut plus = base.to_vec();
    plus[index] += STEP;
    let mut minus = base.to_vec();
    minus[index] -= STEP;
    (f(&plus) - f(&minus)) / (2.0 * STEP)
}

/// 逐元素对拍解析梯度与中心差分：`f` 是参数向量的闭式损失，`name` 只进断言文案。
fn check_grad(grad: &[f32], base: &[f32], name: &str, f: impl Fn(&[f32]) -> f32) {
    for (index, actual) in grad.iter().enumerate() {
        let numeric = central(&f, base, index);
        assert!(
            close(*actual, numeric),
            "{name}[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
}

/// `loss = reduce_sum(op(a, b))` 的公共夹具：两路参数各做一次中心差分，`loss(a, b)` 给整标量。
/// `sub`/`div`/`min`/`max`/`rmsnorm` 这类"两参数、单算子、reduce_sum 收尾"的反向对拍共用。
fn check_binary_grad(
    op: &str,
    a0: &[f32],
    b0: &[f32],
    loss: impl Fn(&[f32], &[f32]) -> f32,
) -> Result<()> {
    let mut tape = Tape::new();
    let a = tape.parameter();
    let b = tape.parameter();
    let y = tape.record(op, &[a, b]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(a, Shape::new([N]), DType::F32);
    sizes.set(b, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(out, Shape::new([1]), DType::F32);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(a, ParamRole::Trainable, a0.to_vec())?;
    session.register_parameter(b, ParamRole::Trainable, b0.to_vec())?;
    session.buffers().compute(op, &[a, b], y)?;
    sum_backward(&mut session, y, out)?;

    // `loss` 借给两个闭包（按引用捕获，避免一次 move 后第二次取不到）。
    let f = &loss;
    check_grad(
        session.buffers().gradient(a).expect("a 梯度应在场"),
        a0,
        "a",
        |pa| f(pa, b0),
    );
    check_grad(
        session.buffers().gradient(b).expect("b 梯度应在场"),
        b0,
        "b",
        |pb| f(a0, pb),
    );
    Ok(())
}

/// `loss = reduce_sum(op(p))` 的公共夹具：单路参数中心差分，`loss(p)` 给整标量。
/// `exp`/`log`/`rope` 这类"单参数、单算子、reduce_sum 收尾"的反向对拍共用。
fn check_unary_grad(op: &str, p0: &[f32], loss: impl Fn(&[f32]) -> f32) -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record(op, &[p]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(out, Shape::new([1]), DType::F32);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.to_vec())?;
    session.buffers().compute(op, &[p], y)?;
    sum_backward(&mut session, y, out)?;

    check_grad(
        session.buffers().gradient(p).expect("p 梯度应在场"),
        p0,
        "p",
        loss,
    );
    Ok(())
}

/// RMSNorm 闭式：`y = γ·x / sqrt(mean(x²)+eps)`，返回逐元素和。
fn rmsnorm_loss(x: &[f32], gamma: &[f32]) -> f32 {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rms = (mean_sq + op::rmsnorm_f32::EPS).sqrt();
    x.iter().zip(gamma).map(|(x, g)| g * x / rms).sum()
}

/// RoPE 闭式：位置固定 7，偶数下标与奇数下标成对旋转，奇数长度末尾透传；返回逐元素和。
fn rope_loss(x: &[f32]) -> f32 {
    let d = x.len();
    let pairs = d / 2;
    let mut out = vec![0.0; d];
    for i in 0..pairs {
        let freq = op::rope_f32::BASE.powf(-(2.0 * i as f32) / d as f32);
        let angle = 7.0 * freq;
        let (sin, cos) = angle.sin_cos();
        out[2 * i] = x[2 * i] * cos - x[2 * i + 1] * sin;
        out[2 * i + 1] = x[2 * i] * sin + x[2 * i + 1] * cos;
    }
    if d % 2 == 1 {
        out[d - 1] = x[d - 1];
    }
    out.iter().sum()
}

/// 两行向量的点积：`a[i]·b[j]`，行主序、行宽 `d`。
fn row_dot(a: &[f32], b: &[f32], i: usize, j: usize, d: usize) -> f32 {
    (0..d).map(|h| a[i * d + h] * b[j * d + h]).sum()
}

/// 注意力闭式：`O = softmax(QKᵀ/√d)·V`，查询行数与历史行数解耦，返回全部输出的和。
fn attention_loss(q: &[f32], k: &[f32], v: &[f32], q_rows: usize, hist: usize, d: usize) -> f32 {
    let scale = 1.0 / (d as f32).sqrt();
    let mut total = 0.0;
    for i in 0..q_rows {
        let scores: Vec<f32> = (0..hist).map(|j| row_dot(q, k, i, j, d) * scale).collect();
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        total += (0..d)
            .map(|h| {
                (0..hist)
                    .map(|j| (exps[j] / sum) * v[j * d + h])
                    .sum::<f32>()
            })
            .sum::<f32>();
    }
    total
}

/// `loss = reduce_sum(rmsnorm(x, gamma))`：`dx` 与 `dgamma` 都闭式对拍。
#[test]
fn rmsnorm_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let gamma = tape.parameter();
    let y = tape.record("rmsnorm_f32", &[x, gamma]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(gamma, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let x0 = vec![0.4_f32, -0.9, 1.3];
    let gamma0 = vec![0.7_f32, 1.1, 0.5];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(x, ParamRole::Trainable, x0.clone())?;
    session.register_parameter(gamma, ParamRole::Trainable, gamma0.clone())?;
    session.buffers().compute("rmsnorm_f32", &[x, gamma], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(x).expect("x 梯度应在场"),
        &x0,
        "x",
        |x| rmsnorm_loss(x, &gamma0),
    );
    check_grad(
        session.buffers().gradient(gamma).expect("gamma 梯度应在场"),
        &gamma0,
        "gamma",
        |gamma| rmsnorm_loss(&x0, gamma),
    );
    Ok(())
}

/// `loss = reduce_sum(rope(x))`：旋转正交，反向即转置；奇数长度末尾透传。
#[test]
fn rope_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.record("rope_f32", &[x]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let x0 = vec![0.5_f32, -0.7, 1.1];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(x, ParamRole::Trainable, x0.clone())?;
    session.buffers().compute("rope_f32", &[x], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(x).expect("x 梯度应在场"),
        &x0,
        "x",
        rope_loss,
    );
    Ok(())
}

/// `loss = reduce_sum(attention(q,k,v))`：`dq` / `dk` / `dv` 三路闭式对拍（头宽固定 8）。
#[test]
fn attention_matches_central_differences() -> Result<()> {
    let d = op::attention::MODEL.shape.1;
    let s = 2usize;
    let len = s * d;
    let mut tape = Tape::new();
    let q = tape.parameter();
    let k = tape.parameter();
    let v = tape.parameter();
    let y = tape.record("attention_f32", &[q, k, v]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(q, Shape::new([len]), DType::F32);
    sizes.set(k, Shape::new([len]), DType::F32);
    sizes.set(v, Shape::new([len]), DType::F32);
    sizes.set(y, Shape::new([len]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let q0: Vec<f32> = (0..len).map(|i| 0.07 * i as f32 - 0.5).collect();
    let k0: Vec<f32> = (0..len).map(|i| -0.05 * i as f32 + 0.4).collect();
    let v0: Vec<f32> = (0..len).map(|i| 0.09 * i as f32 - 0.3).collect();

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(q, ParamRole::Trainable, q0.clone())?;
    session.register_parameter(k, ParamRole::Trainable, k0.clone())?;
    session.register_parameter(v, ParamRole::Trainable, v0.clone())?;
    session.buffers().compute("attention_f32", &[q, k, v], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(q).expect("q 梯度应在场"),
        &q0,
        "q",
        |q| attention_loss(q, &k0, &v0, s, s, d),
    );
    check_grad(
        session.buffers().gradient(k).expect("k 梯度应在场"),
        &k0,
        "k",
        |k| attention_loss(&q0, k, &v0, s, s, d),
    );
    check_grad(
        session.buffers().gradient(v).expect("v 梯度应在场"),
        &v0,
        "v",
        |v| attention_loss(&q0, &k0, v, s, s, d),
    );
    Ok(())
}

/// `loss = reduce_sum(decode_attention(q,k,v))`：查询一行、历史两行，三路梯度闭式对拍。
#[test]
fn decode_attention_matches_central_differences() -> Result<()> {
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = 1usize;
    let hist = 2usize;
    let q_len = q_rows * d;
    let kv_len = hist * d;
    let mut tape = Tape::new();
    let q = tape.parameter();
    let k = tape.parameter();
    let v = tape.parameter();
    let y = tape
        .record("decode_attention_f32", &[q, k, v])
        .expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(q, Shape::new([q_len]), DType::F32);
    sizes.set(k, Shape::new([kv_len]), DType::F32);
    sizes.set(v, Shape::new([kv_len]), DType::F32);
    sizes.set(y, Shape::new([q_len]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let q0: Vec<f32> = (0..q_len).map(|i| 0.11 * i as f32 - 0.4).collect();
    let k0: Vec<f32> = (0..kv_len).map(|i| -0.08 * i as f32 + 0.5).collect();
    let v0: Vec<f32> = (0..kv_len).map(|i| 0.12 * i as f32 - 0.2).collect();

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(q, ParamRole::Trainable, q0.clone())?;
    session.register_parameter(k, ParamRole::Trainable, k0.clone())?;
    session.register_parameter(v, ParamRole::Trainable, v0.clone())?;
    session
        .buffers()
        .compute("decode_attention_f32", &[q, k, v], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.buffers().gradient(q).expect("q 梯度应在场"),
        &q0,
        "q",
        |q| attention_loss(q, &k0, &v0, q_rows, hist, d),
    );
    check_grad(
        session.buffers().gradient(k).expect("k 梯度应在场"),
        &k0,
        "k",
        |k| attention_loss(&q0, k, &v0, q_rows, hist, d),
    );
    check_grad(
        session.buffers().gradient(v).expect("v 梯度应在场"),
        &v0,
        "v",
        |v| attention_loss(&q0, &k0, v, q_rows, hist, d),
    );
    Ok(())
}

/// `loss = reduce_sum(sub(a, b))`：dA = +1、dB = −1，两路逐元素对拍。
#[test]
fn sub_matches_central_differences() -> Result<()> {
    check_binary_grad("sub_f32", &[0.3, -0.5, 0.8], &[1.0, -0.2, 0.6], |a, b| {
        a.iter().zip(b).map(|(x, y)| x - y).sum()
    })
}

/// `loss = reduce_sum(div(a, b))`：dA = 1/b、dB = −a/b²，两路逐元素对拍（`b` 恒正避除零）。
#[test]
fn div_matches_central_differences() -> Result<()> {
    check_binary_grad("div_f32", &[0.6, -0.3, 0.9], &[1.0, 2.0, 0.5], |a, b| {
        a.iter().zip(b).map(|(x, y)| x / y).sum()
    })
}

/// `loss = reduce_sum(exp(p))`：dP = exp(p)（即输出 y）。
#[test]
fn exp_matches_central_differences() -> Result<()> {
    check_unary_grad("exp_f32", &[0.4, -0.6, 1.1], |p| {
        p.iter().map(|&x| x.exp()).sum()
    })
}

/// `loss = reduce_sum(log(p))`：dP = 1/p（夹具恒正）。
#[test]
fn log_matches_central_differences() -> Result<()> {
    check_unary_grad("log_f32", &[0.6, 1.3, 2.1], |p| {
        p.iter().map(|&x| x.ln()).sum()
    })
}

/// `loss = reduce_sum(min(a, b))`：梯度给较小者，两路逐元素对拍（夹具无相等项，避开折点）。
#[test]
fn min_matches_central_differences() -> Result<()> {
    check_binary_grad("min_f32", &[0.3, -0.5, 0.8], &[1.0, -0.2, 0.6], |a, b| {
        a.iter().zip(b).map(|(x, y)| x.min(*y)).sum()
    })
}

/// `loss = reduce_sum(max(a, b))`：梯度给较大者，两路逐元素对拍（夹具无相等项）。
#[test]
fn max_matches_central_differences() -> Result<()> {
    check_binary_grad("max_f32", &[0.3, -0.5, 0.8], &[1.0, -0.2, 0.6], |a, b| {
        a.iter().zip(b).map(|(x, y)| x.max(*y)).sum()
    })
}

/// 端到端蒸馏 KL：`kl = sum(p_s · (ln p_s - ln p_t))` 用图上原语搭出来
/// （softmax→log→sub→mul→reduce_sum），反传后与解析单行 KL 中心差分对拍。
///
/// 这是「损失层从纯函数升级为图上可训练」的闭环证据：五个算子串成一条链，
/// 整链梯度等于解析梯度，而不只是各算子单独可微。
#[test]
fn distillation_kl_graph_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let s = tape.parameter();
    let t = tape.leaf();
    let ps = tape.record("softmax_f32", &[s]).expect("p_s");
    let pt = tape.record("softmax_f32", &[t]).expect("p_t");
    let lps = tape.record("log_f32", &[ps]).expect("ln p_s");
    let lpt = tape.record("log_f32", &[pt]).expect("ln p_t");
    let diff = tape
        .record("sub_f32", &[lps, lpt])
        .expect("ln p_s - ln p_t");
    let prod = tape
        .record("mul_f32", &[ps, diff])
        .expect("p_s·(ln p_s - ln p_t)");
    let kl = tape.record("reduce_sum_f32", &[prod]).expect("kl");

    let mut sizes = Sizes::new(tape.vars());
    for v in [s, t, ps, pt, lps, lpt, diff, prod] {
        sizes.set(v, Shape::new([N]), DType::F32);
    }
    sizes.set(kl, Shape::new([1]), DType::F32);

    let s0 = vec![0.5_f32, -0.3, 0.2];
    let t0 = vec![1.0_f32, 0.1, -0.5];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(s, ParamRole::Trainable, s0.clone())?;
    session.buffers().load(t, Data::F32(t0.clone()))?;
    session.buffers().compute("softmax_f32", &[s], ps)?;
    session.buffers().compute("softmax_f32", &[t], pt)?;
    session.buffers().compute("log_f32", &[ps], lps)?;
    session.buffers().compute("log_f32", &[pt], lpt)?;
    session.buffers().compute("sub_f32", &[lps, lpt], diff)?;
    session.buffers().compute("mul_f32", &[ps, diff], prod)?;
    session.buffers().compute("reduce_sum_f32", &[prod], kl)?;
    session.plan(kl)?;
    session.backward()?;

    check_grad(
        session.buffers().gradient(s).expect("s 梯度应在场"),
        &s0,
        "s",
        |s| kl_single_row(s, &t0),
    );
    Ok(())
}

/// 端到端 GRPO 裁剪：`Σ min(ratio·A, clamp(ratio,1±ε)·A)` 用图上原语搭出来
/// （sub→exp→max→min→mul→mul→min→reduce_sum），反传后与解析裁剪代理中心差分对拍。
///
/// 夹具三个元素各走一路：裁上界（ratio>1+ε, A>0）、裁下界（ratio<1−ε, A<0）、不裁
/// （ratio 在界内）。裁断处梯度为 0（clamp 平坦），不裁处梯度沿 ratio 回传。
#[test]
fn grpo_clipped_surrogate_graph_matches_central_differences() -> Result<()> {
    let eps: f32 = 0.2;
    let lo: f32 = 1.0 - eps;
    let hi: f32 = 1.0 + eps;

    let mut tape = Tape::new();
    let new = tape.parameter();
    let old = tape.leaf();
    let adv = tape.leaf();
    let lo_leaf = tape.leaf();
    let hi_leaf = tape.leaf();
    let diff = tape.record("sub_f32", &[new, old]).expect("new - old");
    let ratio = tape.record("exp_f32", &[diff]).expect("ratio");
    let r_lo = tape
        .record("max_f32", &[ratio, lo_leaf])
        .expect("max(ratio, 1-ε)");
    let r_clamp = tape
        .record("min_f32", &[r_lo, hi_leaf])
        .expect("clamp(ratio)");
    let term1 = tape.record("mul_f32", &[ratio, adv]).expect("ratio·A");
    let term2 = tape
        .record("mul_f32", &[r_clamp, adv])
        .expect("clamp(ratio)·A");
    let surr = tape.record("min_f32", &[term1, term2]).expect("min 裁剪");
    let loss = tape.record("reduce_sum_f32", &[surr]).expect("loss");

    let mut sizes = Sizes::new(tape.vars());
    for v in [
        new, old, adv, lo_leaf, hi_leaf, diff, ratio, r_lo, r_clamp, term1, term2, surr,
    ] {
        sizes.set(v, Shape::new([N]), DType::F32);
    }
    sizes.set(loss, Shape::new([1]), DType::F32);

    let new0 = vec![0.4_f32, -0.5, 0.0];
    let old0 = vec![0.0_f32; N];
    let adv0 = vec![1.0_f32, -0.5, 0.7];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(new, ParamRole::Trainable, new0.clone())?;
    session.buffers().load(old, Data::F32(old0.clone()))?;
    session.buffers().load(adv, Data::F32(adv0.clone()))?;
    session.buffers().load(lo_leaf, Data::F32(vec![lo; N]))?;
    session.buffers().load(hi_leaf, Data::F32(vec![hi; N]))?;
    session.buffers().compute("sub_f32", &[new, old], diff)?;
    session.buffers().compute("exp_f32", &[diff], ratio)?;
    session
        .buffers()
        .compute("max_f32", &[ratio, lo_leaf], r_lo)?;
    session
        .buffers()
        .compute("min_f32", &[r_lo, hi_leaf], r_clamp)?;
    session.buffers().compute("mul_f32", &[ratio, adv], term1)?;
    session
        .buffers()
        .compute("mul_f32", &[r_clamp, adv], term2)?;
    session
        .buffers()
        .compute("min_f32", &[term1, term2], surr)?;
    session.buffers().compute("reduce_sum_f32", &[surr], loss)?;
    session.plan(loss)?;
    session.backward()?;

    check_grad(
        session.buffers().gradient(new).expect("new 梯度应在场"),
        &new0,
        "new",
        |new| clipped_surrogate_sum(new, &old0, &adv0, eps),
    );
    Ok(())
}
}

