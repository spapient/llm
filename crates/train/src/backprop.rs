//! 反向数值执行器：按 [`Schedule`] 的逆拓扑序把 loss 梯度回传到参数终点。
//!
//! [`autograd`] 只产出图结构、反向拓扑序与扇出账本，不含梯度数值；本模块把算子的解析反向
//! 公式落到真实 f32 缓冲：从 loss 的上游梯度 `1` 出发，逐节点把 incoming 梯度按局部雅可比
//! 分给各输入并累加，每收满一份贡献就结算一步。
//!
//! 已接解析反向的是同形逐元素、归约与 softmax 这一类可由本元素前向值推出雅可比的算子；
//! 矩阵乘、查表嵌入、归一化、位置编码、注意力与反量化算子本轮未接，遇到即返回
//! [`Code::Unsupported`]，公式目录后续按算子逐个补。

use autograd::{Node, Schedule, Tape, Var};
use diagnostic::{Code, Error, Result};

use crate::kernel::Kernel;
use crate::store::Buffers;

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::backprop";

/// 按已规划的 [`Schedule`] 回传梯度：起点 loss 梯度为全 `1`，逐节点分配到图上输入。
///
/// 梯度累加进 `buffers` 的梯度槽；图外（[`Requirement::None`]）输入不收账。
///
/// # Errors
///
/// 前向值/上游梯度不在场、账本顺序不符，或遇到尚未接反向公式的算子时返回对应 [`Code`]。
pub(crate) fn backprop<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    schedule: &Schedule,
) -> Result<()> {
    let loss = schedule.loss();
    let loss_len = values(buffers, loss)?.len();
    buffers
        .load_gradient(loss, &vec![1.0; loss_len])
        .map_err(|error| error.with(SITE))?;

    let mut account = schedule.account();
    for step in schedule.steps() {
        reverse_step(buffers, tape, step)?;
        account.settle(step).map_err(|error| error.with(SITE))?;
    }
    Ok(())
}

/// 单节点反向：取输出上游梯度，按算子公式算出各输入的梯度贡献并累加。
fn reverse_step<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    step: &Node,
) -> Result<()> {
    let incoming = incoming(buffers, step)?.to_vec();
    let inputs = step.inputs();
    match step.op() {
        // y = a + b：两条路径都原样回传上游梯度。
        "add_f32" | "residual_add_f32" => {
            for &input in inputs {
                feed(buffers, tape, input, incoming.clone())?;
            }
        }
        // y = a·b：dA = in·b，dB = in·a。
        "mul_f32" => {
            let a = values(buffers, inputs[0])?.to_vec();
            let b = values(buffers, inputs[1])?.to_vec();
            let da = incoming.iter().zip(&b).map(|(&g, &w)| g * w).collect();
            let db = incoming.iter().zip(&a).map(|(&g, &w)| g * w).collect();
            feed(buffers, tape, inputs[0], da)?;
            feed(buffers, tape, inputs[1], db)?;
        }
        // y = α·a：dA = α·in。
        "scale_f32" => {
            let alpha = op::scale_f32::ALPHA;
            let da = incoming.iter().map(|&g| g * alpha).collect();
            feed(buffers, tape, inputs[0], da)?;
        }
        // y = x·σ(x)：y′ = σ + x·σ(1−σ)，dA = in·y′。
        "silu_f32" => {
            let x = values(buffers, inputs[0])?.to_vec();
            let da = incoming
                .iter()
                .zip(&x)
                .map(|(&g, &v)| {
                    let sigmoid = 1.0 / (1.0 + (-v).exp());
                    g * (sigmoid + v * sigmoid * (1.0 - sigmoid))
                })
                .collect();
            feed(buffers, tape, inputs[0], da)?;
        }
        // y = Σa：上游标量梯度广播回每个元素，dA[i] = in[0]。
        "reduce_sum_f32" => {
            let width = values(buffers, inputs[0])?.len();
            let da = vec![incoming[0]; width];
            feed(buffers, tape, inputs[0], da)?;
        }
        // p = softmax(x)：dA[i] = p[i]·(in[i] − Σ_j in[j]·p[j])。
        "softmax_f32" => {
            let probability = values(buffers, step.output())?.to_vec();
            let dot: f32 = incoming
                .iter()
                .zip(&probability)
                .map(|(g, &p)| g * p)
                .sum();
            let da = incoming
                .iter()
                .zip(&probability)
                .map(|(&g, &p)| p * (g - dot))
                .collect();
            feed(buffers, tape, inputs[0], da)?;
        }
        other => {
            return Err(Error::new(
                Code::Unsupported,
                format!("算子 `{other}` 还没有解析反向公式"),
            )
            .with(SITE));
        }
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

/// 读变量在场的前向 f32 值；缺失是重放/装配口径问题。
fn values<'a, K: Kernel>(buffers: &'a Buffers<K>, var: Var) -> Result<&'a [f32]> {
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
