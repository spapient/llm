//! 反向公式表里的 36 条解析公式与共享数值助手。
//!
//! 几何交给 [`Geometry`]，公式本身只负责"取实例、回传"或逐元素/块级局部雅可比；声明档
//! 与实例化共用同一份算术，不存在"前向一套、反向另一套"。

use super::{Context, SITE};
use diagnostic::{Code, Error, Result};

/// 经典算子的反向公式：几何交给 [`Geometry`]，公式本身只剩"取实例、回传"。
///
/// 声明档与实例化共用同一份算术——声明档就是 `Geometry::declared` 的取值，不存在
/// "前向一套、反向另一套"的可能。这十条"取实例回传"的例程同构，用一张表展开。
macro_rules! instance_backward {
    ($($name:ident => $op:literal),* $(,)?) => {$(
        pub(super) fn $name(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
            ctx.instance($op)?.backward(ctx.inputs, ctx.incoming)
        }
    )*};
}

instance_backward!(
    max_pool1d => "max_pool1d_f32",
    max_pool2d => "max_pool2d_f32",
    avg_pool2d => "avg_pool2d_f32",
    adaptive_pool => "adaptive_pool_f32",
    interpolate => "interpolate_f32",
    batch_norm => "batch_norm_f32",
    conv2d => "conv2d_f32",
    group_norm => "group_norm_f32",
    instance_norm => "instance_norm_f32",
    layer_norm => "layer_norm_f32",
);

/// y = a + b：两条路径都原样回传上游梯度。
pub(super) fn passthrough(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![ctx.incoming.to_vec(); ctx.inputs.len()])
}

/// y = a·b：dA = in·b，dB = in·a。
pub(super) fn mul(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let da = ctx.incoming.iter().zip(b).map(|(&g, w)| g * w).collect();
    let db = ctx.incoming.iter().zip(a).map(|(&g, w)| g * w).collect();
    Ok(vec![da, db])
}

/// y = α·a：dA = α·in。
pub(super) fn scale(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let alpha = op::scale_f32::ALPHA;
    let da = ctx.incoming.iter().map(|&g| g * alpha).collect();
    Ok(vec![da])
}

/// y = a−b：dA = in，dB = −in。
pub(super) fn sub(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx.incoming.to_vec();
    let db = ctx.incoming.iter().map(|&g| -g).collect();
    Ok(vec![da, db])
}

/// y = a/b：dA = in/b，dB = −in·a/b²。
pub(super) fn div(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn exp(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(ctx.output)
        .map(|(&g, &y)| g * y)
        .collect();
    Ok(vec![da])
}

/// y = ln(a)：dA = in/a。
pub(super) fn log(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(&ctx.inputs[0])
        .map(|(&g, &x)| g / x)
        .collect();
    Ok(vec![da])
}

/// y = min(a,b)：梯度给较小者；相等给 `a`（避免双计）。
pub(super) fn min(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn max(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn silu(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn leaky_relu(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn hard_sigmoid(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn gelu(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn reduce_sum(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let da = vec![ctx.incoming[0]; ctx.inputs[0].len()];
    Ok(vec![da])
}

/// y = max(a)：只有最大值元素收到 `in[0]`，其余梯度为 0。
pub(super) fn reduce_max(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let peak = a.iter().copied().fold(f32::MIN, f32::max);
    let da = a
        .iter()
        .map(|&v| if v == peak { ctx.incoming[0] } else { 0.0 })
        .collect();
    Ok(vec![da])
}

/// p = softmax(x)：dA[i] = p[i]·(in[i] − Σ_j in[j]·p[j])。
pub(super) fn softmax(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn gemv(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn gemm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn embed(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn rmsnorm(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn rope(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn attention(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
    let d = op::attention::MODEL.shape.1;
    let s = divide(ctx.inputs[0].len(), d, "attention_f32")?;
    let (dq, dk, dv) = attention_grad(attention_ctx(&ctx, d), s, s);
    Ok(vec![dq, dk, dv])
}

/// decode 注意力（`decode_attention_f32` / `decode_flash_attention_f32` 共用）：查询行数与历史
/// 行数解耦，返回 `[dq, dk, dv]`。
pub(super) fn decode_attention(ctx: Context<'_>) -> Result<Vec<Vec<f32>>> {
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
