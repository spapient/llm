//! 反向数值执行器：按 [`Schedule`] 的逆拓扑序把 loss 梯度回传到参数终点。
//!
//! [`autograd`] 只产出图结构、反向拓扑序与扇出账本，不含梯度数值；本模块把算子的解析反向
//! 公式落到真实 f32 缓冲：从 loss 的上游梯度 `1` 出发，逐节点把 incoming 梯度按局部雅可比
//! 分给各输入并累加，每收满一份贡献就结算一步。
//!
//! 反向公式表与前向内核表同构：算子名、元数与派生几何都在 [`FORMULAS`] 闭集里登记，未接
//! 公式的算子遇到即返回 [`Code::Unsupported`]，执行循环不逐名字 `match` 散落分支。归一化、
//! 位置编码、注意力、反量化与量化权重通路本轮未接，公式目录后续按算子逐个补。

use autograd::{Node, Schedule, Tape, Var};
use diagnostic::{Code, Error, Result};

use crate::kernel::Kernel;
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
fn reverse_step<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    step: &Node,
) -> Result<()> {
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
    Error::new(
        Code::Unsupported,
        format!("算子 `{op}` 还没有解析反向公式"),
    )
    .with(SITE)
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

/// 已接反向公式的算子目录；其余算子在 [`apply_formula`] 处回 [`Code::Unsupported`]。
const FORMULAS: [Entry; 12] = [
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
];

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
    eprintln!("DEBUG gelu inputs={x:?} incoming={:?}", ctx.incoming);
    let da = ctx
        .incoming
        .iter()
        .zip(x)
        .map(|(&g, &v)| g * gelu_grad(v))
        .collect();
    eprintln!("DEBUG gelu da={da:?}");
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
    let out = ctx.output;
    let k = op::gemm::K;
    let m = a.len() / k;
    let n = out.len() / m;
    let mut da = vec![0.0; a.len()];
    let mut db = vec![0.0; b.len()];
    for m_row in 0..m {
        for k_dim in 0..k {
            for n_col in 0..n {
                let grad = out[m_row * n + n_col];
                da[m_row * k + k_dim] += grad * b[k_dim * n + n_col];
                db[k_dim * n + n_col] += a[m_row * k + k_dim] * grad;
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

/// GELU 导数：`y = 0.5·x·(1+erf(x/√2))`，`y′ = φ(x) + x·Φ(x)`。
fn gelu_grad(x: f32) -> f32 {
    let normal = (-(x * x) / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
    let erf_term = 0.5 * (1.0 + erf_scalar(x / std::f32::consts::FRAC_1_SQRT_2));
    erf_term + x * normal
}

/// 标量误差函数：Abramowitz-Stegun 7.1.26，与 `op::gelu_f32` 同一份系数（|误差| < 1.5e-7）。
fn erf_scalar(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.327_591_1;
    let a1 = 0.254_829_6;
    let a2 = -0.284_496_72;
    let a3 = 1.421_413_8;
    let a4 = -1.453_152_1;
    let a5 = 1.061_405_4;
    let t = 1.0 / (1.0 + p * x);
    let series = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let y = 1.0 - series * (-x * x).exp();
    sign * y
}

#[cfg(test)]
#[path = "../tests/backprop_tests.rs"]
mod tests;
