//! 反向数值执行器：按 [`Schedule`] 的逆拓扑序把 loss 梯度回传到参数终点。
//!
//! [`autograd`] 只产出图结构、反向拓扑序与扇出账本，不含梯度数值；本模块把算子的解析反向
//! 公式落到真实 f32 缓冲：从 loss 的上游梯度 `1` 出发，逐节点把 incoming 梯度按局部雅可比
//! 分给各输入并累加，每收满一份贡献就结算一步。
//!
//! 反向公式表与前向内核表同构：算子名、元数与派生几何都在 [`BACKWARD`] 闭集里登记，未接公式
//! 的算子遇到即返回 [`Code::Unsupported`]，执行循环不逐名字 `match` 散落分支。归一化、位置
//! 编码、注意力、反量化与量化权重通路本轮未接，公式目录后续按算子逐个补。

use autograd::{Node, Schedule, Tape, Var};
use diagnostic::{Code, Error, Result};

use crate::kernel::Kernel;
use crate::pool::Buffers;

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

/// 反向公式的分量归属：一条算子反传到哪些输入、各输入的雅可比从什么前向量派生。
///
/// 枚举由反传目录单独维护；新增公式只加一行表项，不动执行循环。
enum Backward {
    /// `y = a + b`：两条路径都原样回传上游梯度。
    CopyBoth,
    /// `y = a·b`：`dA = in·b`，`dB = in·a`。
    Mul,
    /// `y = α·a`：`dA = α·in`，系数取算子登记常量。
    Scale(f32),
    /// `y = x·σ(x)`：`y′ = σ + x·σ(1−σ)`，`dA = in·y′`。
    Silu,
    /// `y = 0.5·x·(1 + erf(x/√2))`：`y′ = φ(x) + x·Φ(x)`，φ 为标准正态密度。
    Gelu,
    /// `y = Σa`：上游标量梯度广播回每个元素，`dA[i] = in[0]`。
    ReduceSum,
    /// `y = max(a)`：只有最大值元素收到 `in[0]`，其余梯度为 0。
    ReduceMax,
    /// `p = softmax(x)`：`dA[i] = p[i]·(in[i] − Σ_j in[j]·p[j])`。
    Softmax,
    /// `y = W·x`（`[n] = Σ_k W[n,k]·x[k]`）：`dW[n,k] = in[n]·x[k]`，`dx[k] = Σ_n in[n]·W[n,k]`。
    Gemv,
    /// `Y = A·B`（`[m,n] = Σ_k A[m,k]·B[k,n]`，收缩轴固定 `op::gemm::K`）：
    /// `dA[m,k] = Σ_n in[m,n]·B[k,n]`，`dB[k,n] = Σ_m A[m,k]·in[m,n]`。
    Gemm,
    /// `y = table[idx]`（行 gather）：索引不连续、无雅可比；权重表按被访问行累加
    /// `dTable[idx[i], :] += in[i, :]`。
    Embed,
}

/// 反传目录：算子名 → 公式分量。表长即已接公式的算子数，新增公式在这张表里补一行。
const BACKWARD: &[(&str, Backward)] = &[
    ("add_f32", Backward::CopyBoth),
    ("residual_add_f32", Backward::CopyBoth),
    ("mul_f32", Backward::Mul),
    ("scale_f32", Backward::Scale(op::scale_f32::ALPHA)),
    ("silu_f32", Backward::Silu),
    ("gelu_f32", Backward::Gelu),
    ("reduce_sum_f32", Backward::ReduceSum),
    ("reduce_max_f32", Backward::ReduceMax),
    ("softmax_f32", Backward::Softmax),
    ("gemv_f32", Backward::Gemv),
    ("gemm_f32", Backward::Gemm),
    ("embed_f32", Backward::Embed),
];

/// 单节点反向：取输出上游梯度，按算子公式算出各输入的梯度贡献并累加。
fn reverse_step<K: Kernel>(
    buffers: &mut Buffers<K>,
    tape: &Tape,
    step: &Node,
) -> Result<()> {
    let incoming = incoming(buffers, step)?.to_vec();
    let inputs = step.inputs();
    let Some((_, formula)) = BACKWARD.iter().find(|(name, _)| *name == step.op()) else {
        return Err(Error::new(
            Code::Unsupported,
            format!("算子 `{}` 还没有解析反向公式", step.op()),
        )
        .with(SITE));
    };

    match formula {
        Backward::CopyBoth => {
            for &input in inputs {
                feed(buffers, tape, input, incoming.clone())?;
            }
        }
        Backward::Mul => {
            let a = values(buffers, inputs[0])?.to_vec();
            let b = values(buffers, inputs[1])?.to_vec();
            let da = incoming.iter().zip(&b).map(|(&g, &w)| g * w).collect();
            let db = incoming.iter().zip(&a).map(|(&g, &w)| g * w).collect();
            feed(buffers, tape, inputs[0], da)?;
            feed(buffers, tape, inputs[1], db)?;
        }
        Backward::Scale(alpha) => {
            let da = incoming.iter().map(|&g| g * alpha).collect();
            feed(buffers, tape, inputs[0], da)?;
        }
        Backward::Silu => {
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
        Backward::Gelu => {
            let x = values(buffers, inputs[0])?.to_vec();
            let da = incoming.iter().zip(&x).map(|(&g, &v)| g * gelu_grad(v)).collect();
            feed(buffers, tape, inputs[0], da)?;
        }
        Backward::ReduceSum => {
            let width = values(buffers, inputs[0])?.len();
            feed(buffers, tape, inputs[0], vec![incoming[0]; width])?;
        }
        Backward::ReduceMax => {
            let a = values(buffers, inputs[0])?.to_vec();
            let peak = a.iter().copied().fold(f32::MIN, f32::max);
            let da = a.iter().map(|&v| if v == peak { incoming[0] } else { 0.0 }).collect();
            feed(buffers, tape, inputs[0], da)?;
        }
        Backward::Softmax => {
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
        Backward::Gemv => {
            let w = values(buffers, inputs[0])?.to_vec();
            let x = values(buffers, inputs[1])?.to_vec();
            let (rows, cols) = matrix_shape(&w, x.len())?;
            // dW[n,k] = in[n]·x[k]：外层行号决定被广播的上游分量，内层整块复制输入行。
            let dtable = w_slot(buffers, inputs[0])?;
            for n in 0..rows {
                let row_base = n * cols;
                let mut drow = vec![0.0; cols];
                scale_into(&mut drow, &x, incoming[n]);
                accumulate_into(dtable, row_base, &drow)?;
            }
            // dx[k] = Σ_n in[n]·W[n,k]：按列累加，W 按行主序展开。
            let mut dx = vec![0.0; cols];
            for n in 0..rows {
                for k in 0..cols {
                    dx[k] += incoming[n] * w[n * cols + k];
                }
            }
            feed(buffers, tape, inputs[1], dx)?;
        }
        Backward::Gemm => {
            let a = values(buffers, inputs[0])?;
            let b = values(buffers, inputs[1])?;
            let out = values(buffers, step.output())?;
            let k = op::gemm::K;
            let m = a.len() / k;
            let n = out.len() / m;
            // dA[m,k] = Σ_n in[m,n]·B[k,n]；dB[k,n] = Σ_m A[m,k]·in[m,n]。
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
            feed(buffers, tape, inputs[0], da)?;
            feed(buffers, tape, inputs[1], db)?;
        }
        Backward::Embed => {
            let idx = values(buffers, inputs[0])?;
            let table = values(buffers, inputs[1])?;
            let dim = incoming.len() / idx.len();
            let mut dtable = vec![0.0; table.len()];
            // 每个 token 一行：上游行整块累加进被查表行；索引不连续，逐行归位无法整块搬运。
            for (token, &index) in idx.iter().enumerate() {
                let row = index as usize * dim;
                let grad_row = &incoming[token * dim..][..dim];
                accumulate_into(&mut dtable, row, grad_row)?;
            }
            feed(buffers, tape, inputs[1], dtable)?;
        }
    }
    Ok(())
}

/// GELU 导数：`y = 0.5·x·(1+erf(x/√2))`，`y′ = φ(x) + x·Φ(x)`。
fn gelu_grad(x: f32) -> f32 {
    let normal = (-(x * x) / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
    let erf_term = 0.5 * (1.0 + erf_scalar(x / std::f32::consts::FRAC_1_SQRT_2));
    normal + x * erf_term
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

/// 整块缩放进目标行：`dst[i] = factor · src[i]`，一次 zip 完成。
fn scale_into(dst: &mut [f32], src: &[f32], factor: f32) {
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot = factor * value;
    }
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

/// 取权重槽当前的可变梯度视图；未装入是反传前置缺失（[`Code::Missing`]）。
fn w_slot<K: Kernel>(buffers: &mut Buffers<K>, var: Var) -> Result<&mut [f32]> {
    buffers.gradient_mut(var).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!("权重变量 {} 的梯度槽不在场", var.id()),
        )
        .with(SITE)
    })
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

#[cfg(test)]
#[path = "../tests/backprop_tests.rs"]
mod tests;
