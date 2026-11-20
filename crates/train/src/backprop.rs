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
const FORMULAS: [Entry; 35] = [
    Entry {
        op: "adaptive_pool_f32",
        formula: adaptive_pool,
    },
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
        op: "max_pool1d_f32",
        formula: max_pool1d,
    },
    Entry {
        op: "max_pool2d_f32",
        formula: max_pool2d,
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
        op: "hard_sigmoid_f32",
        formula: hard_sigmoid,
    },
    Entry {
        op: "instance_norm_f32",
        formula: instance_norm,
    },
    Entry {
        op: "interpolate_f32",
        formula: interpolate,
    },
    Entry {
        op: "layer_norm_f32",
        formula: layer_norm,
    },
    Entry {
        op: "leaky_relu_f32",
        formula: leaky_relu,
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
        op: "group_norm_f32",
        formula: group_norm,
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
        op: "avg_pool2d_f32",
        formula: avg_pool2d,
    },
    Entry {
        op: "batch_norm_f32",
        formula: batch_norm,
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

/// 声明档几何核对：这些公式只对算子的声明档成立（层侧真实几何在别的路径上），
/// 形状不符时显式拒绝，避免把别的几何静默算成错梯度。
fn expect(what: &str, actual: usize, want: usize) -> Result<()> {
    if actual == want {
        Ok(())
    } else {
        Err(Error::new(
            Code::Invalid,
            format!("{what}: 声明档 {want} 个元素，实际 {actual}"),
        )
        .with(SITE))
    }
}

/// 一维最大池化反向：梯度只落在窗口最大者的位置（与前向同一比较规则，平手取窗口首元素）。
fn max_pool1d(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::max_pool1d_f32 as decl;
    let x = &ctx.inputs[0];
    expect("max_pool1d 输入", x.len(), decl::L)?;
    expect("max_pool1d 上游梯度", ctx.incoming.len(), decl::OL)?;
    let mut dx = vec![0.0f32; x.len()];
    for (o, &g) in ctx.incoming.iter().enumerate() {
        let pick = if x[2 * o] >= x[2 * o + 1] { 2 * o } else { 2 * o + 1 };
        dx[pick] += g;
    }
    Ok(vec![dx])
}

/// 二维最大池化反向：逐输出位重放窗口最大值位置（补零位不参与）。
fn max_pool2d(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::max_pool2d_f32 as decl;
    let x = &ctx.inputs[0];
    let side = decl::H;
    expect("max_pool2d 输入", x.len(), side * side)?;
    expect("max_pool2d 上游梯度", ctx.incoming.len(), side * side)?;
    let mut dx = vec![0.0f32; x.len()];
    for i in 0..side {
        for j in 0..side {
            let mut best = f32::MIN;
            let mut pick = (i, j);
            for di in -1i32..=1 {
                for dj in -1i32..=1 {
                    let ii = i as i32 + di;
                    let jj = j as i32 + dj;
                    if ii < 0 || jj < 0 || ii as usize >= side || jj as usize >= side {
                        continue;
                    }
                    let at = (ii as usize, jj as usize);
                    let v = x[at.0 * side + at.1];
                    if v > best {
                        best = v;
                        pick = at;
                    }
                }
            }
            dx[pick.0 * side + pick.1] += ctx.incoming[i * side + j];
        }
    }
    Ok(vec![dx])
}

/// 二维平均池化反向：窗口内每个有效元素均分（补零不计入分母，角落窗口按真统计）。
fn avg_pool2d(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::avg_pool2d_f32 as decl;
    let x = &ctx.inputs[0];
    let side = decl::H;
    expect("avg_pool2d 输入", x.len(), side * side)?;
    expect("avg_pool2d 上游梯度", ctx.incoming.len(), side * side)?;
    let mut dx = vec![0.0f32; x.len()];
    for i in 0..side {
        for j in 0..side {
            let lo = (i.saturating_sub(1), j.saturating_sub(1));
            let hi = ((i + 1).min(side - 1), (j + 1).min(side - 1));
            let count = ((hi.0 - lo.0 + 1) * (hi.1 - lo.1 + 1)) as f32;
            let share = ctx.incoming[i * side + j] / count;
            for ii in lo.0..=hi.0 {
                for jj in lo.1..=hi.1 {
                    dx[ii * side + jj] += share;
                }
            }
        }
    }
    Ok(vec![dx])
}

/// 自适应平均池化反向（声明档 `4×4 → 2×2`，每格固定 2×2、均分）。
fn adaptive_pool(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::adaptive_pool_f32 as decl;
    let x = &ctx.inputs[0];
    let (side, out) = (decl::H, decl::OH);
    let span = side / out;
    expect("adaptive_pool 输入", x.len(), side * side)?;
    expect("adaptive_pool 上游梯度", ctx.incoming.len(), out * out)?;
    let mut dx = vec![0.0f32; x.len()];
    for i in 0..out {
        for j in 0..out {
            let share = ctx.incoming[i * out + j] / (span * span) as f32;
            for di in 0..span {
                for dj in 0..span {
                    dx[(i * span + di) * side + j * span + dj] += share;
                }
            }
        }
    }
    Ok(vec![dx])
}

/// 双线性插值反向（声明档 `2×2 → 4×4`，`align_corners = false`）：权重与前向同一套坐标夹取。
fn interpolate(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::interpolate_f32 as decl;
    let x = &ctx.inputs[0];
    let (h, w, out) = (decl::H, decl::W, decl::OH);
    expect("interpolate 输入", x.len(), h * w)?;
    expect("interpolate 上游梯度", ctx.incoming.len(), out * out)?;
    let mut dx = vec![0.0f32; x.len()];
    for oi in 0..out {
        let fy = ((oi as f32 + 0.5) * h as f32 / out as f32 - 0.5).max(0.0);
        let y0 = fy.floor().min((h - 1) as f32) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let ly = if y0 == y1 { 0.0 } else { fy - y0 as f32 };
        for oj in 0..out {
            let fx = ((oj as f32 + 0.5) * w as f32 / out as f32 - 0.5).max(0.0);
            let x0 = fx.floor().min((w - 1) as f32) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let lx = if x0 == x1 { 0.0 } else { fx - x0 as f32 };
            let g = ctx.incoming[oi * out + oj];
            dx[y0 * w + x0] += g * (1.0 - lx) * (1.0 - ly);
            dx[y0 * w + x1] += g * lx * (1.0 - ly);
            dx[y1 * w + x0] += g * (1.0 - lx) * ly;
            dx[y1 * w + x1] += g * lx * ly;
        }
    }
    Ok(vec![dx])
}

/// 归一化反向的公共骨架：对一组元素按 `(均值, 方差)` 求 `dx`，同时把 `dgamma`/`dbeta` 累加进
/// 按通道的输出。`members` 给出该组元素下标与其通道号。
fn norm_group(
    x: &[f32],
    gamma: &[f32],
    incoming: &[f32],
    members: &[(usize, usize)],
    eps: f32,
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    let count = members.len() as f32;
    let (mut sum, mut sq) = (0.0f32, 0.0f32);
    for &(idx, _) in members {
        sum += x[idx];
        sq += x[idx] * x[idx];
    }
    let mean = sum / count;
    let var = (sq / count - mean * mean).max(0.0);
    let inv = 1.0 / (var + eps).sqrt();

    let (mut sum_dxhat, mut sum_dxhat_xhat) = (0.0f32, 0.0f32);
    for &(idx, c) in members {
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[c];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
        // 对 gamma 的偏导只经 xhat 一条路：dxhat 已乘过 gamma，不能再乘第二次。
        dgamma[c] += incoming[idx] * xhat;
        dbeta[c] += incoming[idx];
    }
    for &(idx, c) in members {
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[c];
        dx[idx] = inv * (dxhat - sum_dxhat / count - xhat * sum_dxhat_xhat / count);
    }
}

/// 批归一化反向（NHWC，统计域是同通道跨批：`N·S` 个元素）。
fn batch_norm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::batch_norm_f32 as decl;
    let (x, gamma, beta) = (&ctx.inputs[0], &ctx.inputs[1], &ctx.inputs[2]);
    let total = decl::C * decl::S * decl::N;
    expect("batch_norm 输入", x.len(), total)?;
    expect("batch_norm gamma", gamma.len(), decl::C)?;
    expect("batch_norm beta", beta.len(), decl::C)?;
    expect("batch_norm 上游梯度", ctx.incoming.len(), total)?;
    let mut dx = vec![0.0f32; total];
    let mut dgamma = vec![0.0f32; decl::C];
    let mut dbeta = vec![0.0f32; decl::C];
    let per = decl::N * decl::S;
    for c in 0..decl::C {
        let members: Vec<(usize, usize)> = (0..per).map(|i| (i * decl::C + c, c)).collect();
        norm_group(x, gamma, ctx.incoming, &members, decl::EPS, &mut dx, &mut dgamma, &mut dbeta);
    }
    Ok(vec![dx, dgamma, dbeta])
}

/// 组归一化反向：统计域是 `(样本, 组)` 的 `C/G·S` 个元素。
fn group_norm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::group_norm_f32 as decl;
    let (x, gamma, beta) = (&ctx.inputs[0], &ctx.inputs[1], &ctx.inputs[2]);
    let total = decl::C * decl::S * decl::N;
    expect("group_norm 输入", x.len(), total)?;
    expect("group_norm gamma", gamma.len(), decl::C)?;
    expect("group_norm beta", beta.len(), decl::C)?;
    expect("group_norm 上游梯度", ctx.incoming.len(), total)?;
    let per = decl::C / decl::G;
    let mut dx = vec![0.0f32; total];
    let mut dgamma = vec![0.0f32; decl::C];
    let mut dbeta = vec![0.0f32; decl::C];
    for bi in 0..decl::N {
        for g in 0..decl::G {
            let c0 = g * per;
            let members: Vec<(usize, usize)> = (0..decl::S)
                .flat_map(|s| (c0..c0 + per).map(move |c| ((bi * decl::S + s) * decl::C + c, c)))
                .collect();
            norm_group(
                x,
                gamma,
                ctx.incoming,
                &members,
                decl::EPS,
                &mut dx,
                &mut dgamma,
                &mut dbeta,
            );
        }
    }
    Ok(vec![dx, dgamma, dbeta])
}

/// 实例归一化反向：等价于 `GroupNorm(groups = channels)`，统计域是每个 `(样本, 通道)` 的 `S` 个元素。
fn instance_norm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::instance_norm_f32 as decl;
    let (x, gamma, beta) = (&ctx.inputs[0], &ctx.inputs[1], &ctx.inputs[2]);
    let total = decl::C * decl::S * decl::N;
    expect("instance_norm 输入", x.len(), total)?;
    expect("instance_norm gamma", gamma.len(), decl::C)?;
    expect("instance_norm beta", beta.len(), decl::C)?;
    expect("instance_norm 上游梯度", ctx.incoming.len(), total)?;
    let mut dx = vec![0.0f32; total];
    let mut dgamma = vec![0.0f32; decl::C];
    let mut dbeta = vec![0.0f32; decl::C];
    for bi in 0..decl::N {
        for c in 0..decl::C {
            let members: Vec<(usize, usize)> = (0..decl::S)
                .map(|s| ((bi * decl::S + s) * decl::C + c, c))
                .collect();
            norm_group(
                x,
                gamma,
                ctx.incoming,
                &members,
                decl::EPS,
                &mut dx,
                &mut dgamma,
                &mut dbeta,
            );
        }
    }
    Ok(vec![dx, dgamma, dbeta])
}

/// 层归一化反向：逐行在最后一维统计，`gamma`/`beta` 按列广播，梯度跨行累加。
fn layer_norm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    use op::layer_norm_f32 as decl;
    let (x, gamma, beta) = (&ctx.inputs[0], &ctx.inputs[1], &ctx.inputs[2]);
    let total = decl::R * decl::W;
    expect("layer_norm 输入", x.len(), total)?;
    expect("layer_norm gamma", gamma.len(), decl::W)?;
    expect("layer_norm beta", beta.len(), decl::W)?;
    expect("layer_norm 上游梯度", ctx.incoming.len(), total)?;
    let mut dx = vec![0.0f32; total];
    let mut dgamma = vec![0.0f32; decl::W];
    let mut dbeta = vec![0.0f32; decl::W];
    for r in 0..decl::R {
        let base = r * decl::W;
        let members: Vec<(usize, usize)> = (0..decl::W).map(|j| (base + j, j)).collect();
        norm_group(
            x,
            gamma,
            ctx.incoming,
            &members,
            decl::EPS,
            &mut dx,
            &mut dgamma,
            &mut dbeta,
        );
    }
    Ok(vec![dx, dgamma, dbeta])
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

/// y = x（x ≥ 0）或 slope·x：y′ = 1 或 slope，dA = in·y′。
fn leaky_relu(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let x = &ctx.inputs[0];
    let slope = op::leaky_relu_f32::SLOPE;
    let da = ctx
        .incoming
        .iter()
        .zip(x)
        .map(|(&g, &v)| g * if v >= 0.0 { 1.0 } else { slope })
        .collect();
    Ok(vec![da])
}

/// y = clamp(x/6 + 0.5, 0, 1)：饱和区梯度为 0，线性区为 1/6。
fn hard_sigmoid(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let x = &ctx.inputs[0];
    let da = ctx
        .incoming
        .iter()
        .zip(x)
        .map(|(&g, &v)| {
            let v = v as f64;
            let inside = (v / 6.0 + 0.5) > 0.0 && (v / 6.0 + 0.5) < 1.0;
            g * if inside { 1.0 / 6.0 } else { 0.0 }
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
#[path = "../tests/backprop_tests.rs"]
mod tests;
