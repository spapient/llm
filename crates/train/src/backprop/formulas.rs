//! 反向公式表里的 36 条解析公式与共享数值助手。

use super::{Ctx, SITE};
use diagnostic::{Code, Error, Result};
use std::cell::RefCell;
use std::rc::Rc;

/// 经典算子的反向公式：几何交给 [`Geometry`]，公式本身只剩"取实例、回传"。
macro_rules! instance_backward {
    ($($name:ident => $op:literal),* $(,)?) => {$(
        pub(super) fn $name(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn passthrough(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![ctx.incoming.to_vec(); ctx.inputs.len()])
}

/// y = a·b：dA = in·b，dB = in·a。
pub(super) fn mul(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let da = ctx.incoming.iter().zip(b).map(|(&g, w)| g * w).collect();
    let db = ctx.incoming.iter().zip(a).map(|(&g, w)| g * w).collect();
    Ok(vec![da, db])
}

/// y = α·a：dA = α·in。
pub(super) fn scale(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let alpha = op::scale_f32::ALPHA;
    let da = ctx.incoming.iter().map(|&g| g * alpha).collect();
    Ok(vec![da])
}

/// y = a−b：dA = in，dB = −in。
pub(super) fn sub(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx.incoming.to_vec();
    let db = ctx.incoming.iter().map(|&g| -g).collect();
    Ok(vec![da, db])
}

/// y = a/b：dA = in/b，dB = −in·a/b²。
pub(super) fn div(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn exp(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(ctx.output)
        .map(|(&g, &y)| g * y)
        .collect();
    Ok(vec![da])
}

/// y = ln(a)：dA = in/a。
pub(super) fn log(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(&ctx.inputs[0])
        .map(|(&g, &x)| g / x)
        .collect();
    Ok(vec![da])
}

/// y = min(a,b)：梯度给较小者；相等给 `a`（避免双计）。
pub(super) fn min(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    extremum(ctx, |x, y| x <= y, |x, y| y < x)
}

/// y = max(a,b)：梯度给较大者；相等给 `a`（避免双计）。
pub(super) fn max(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    extremum(ctx, |x, y| x >= y, |x, y| y > x)
}

/// `min`/`max` 共用的选点反向：`a_wins` / `b_wins` 各自判"该元素拿走上游梯度"，等号只归 `a`。
fn extremum(
    ctx: Ctx<'_>,
    a_wins: impl Fn(f32, f32) -> bool,
    b_wins: impl Fn(f32, f32) -> bool,
) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let b = &ctx.inputs[1];
    let pick = |wins: &dyn Fn(f32, f32) -> bool| -> Vec<f32> {
        ctx.incoming
            .iter()
            .zip(a)
            .zip(b)
            .map(|((&g, &x), &y)| if wins(x, y) { g } else { 0.0 })
            .collect()
    };
    Ok(vec![pick(&a_wins), pick(&b_wins)])
}

/// 一元逐元素算子：`y′` 只看本元素，`dA[i] = in[i]·f(x[i])`。
fn unary(ctx: Ctx<'_>, grad: impl Fn(f32) -> f32) -> Result<Vec<Vec<f32>>> {
    let x = &ctx.inputs[0];
    let da = ctx
        .incoming
        .iter()
        .zip(x)
        .map(|(&g, &v)| g * grad(v))
        .collect();
    Ok(vec![da])
}

/// y = x·σ(x)：y′ = σ + x·σ(1−σ)，dA = in·y′。
pub(super) fn silu(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    unary(ctx, |v| {
        let sigmoid = 1.0 / (1.0 + (-v).exp());
        sigmoid + v * sigmoid * (1.0 - sigmoid)
    })
}

/// y = σ(x)：y′ = y·(1−y)，输出即 σ，避免重算。
pub(super) fn sigmoid(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(ctx.output)
        .map(|(&g, &y)| g * y * (1.0 - y))
        .collect();
    Ok(vec![da])
}

/// y = tanh(x)：y′ = 1−y²，输出即 tanh。
pub(super) fn tanh(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = ctx
        .incoming
        .iter()
        .zip(ctx.output)
        .map(|(&g, &y)| g * (1.0 - y * y))
        .collect();
    Ok(vec![da])
}

/// y = x（x ≥ 0）或 slope·x：y′ = 1 或 slope，dA = in·y′。
pub(super) fn leaky_relu(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    unary(ctx, |v| {
        if v >= 0.0 {
            1.0
        } else {
            op::leaky_relu_f32::SLOPE
        }
    })
}

/// y = max(x, 0)：y′ = 1（x > 0）或 0；负区的斜率是 0，不是 LeakyReLU 那份 0.01。
pub(super) fn relu(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    unary(ctx, |v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// y = clamp(x/6 + 0.5, 0, 1)：饱和区梯度为 0，线性区为 1/6。
pub(super) fn hard_sigmoid(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    unary(ctx, |v| {
        let v = f64::from(v);
        let inside = (v / 6.0 + 0.5) > 0.0 && (v / 6.0 + 0.5) < 1.0;
        if inside { 1.0 / 6.0 } else { 0.0 }
    })
}

/// y = 0.5·x·(1 + erf(x/√2))：y′ = Φ(x) + x·φ(x)，φ 为标准正态密度、Φ 为累积分布。
pub(super) fn gelu(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    unary(ctx, gelu_grad)
}

/// y = Σa：上游标量梯度广播回每个元素，dA[i] = in[0]。
pub(super) fn reduce_sum(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let da = vec![ctx.incoming[0]; ctx.inputs[0].len()];
    Ok(vec![da])
}

/// y = max(a)：只有最大值元素收到 `in[0]`，其余梯度为 0。
pub(super) fn reduce_max(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let a = &ctx.inputs[0];
    let peak = a.iter().copied().fold(f32::MIN, f32::max);
    let da = a
        .iter()
        .map(|&v| if v == peak { ctx.incoming[0] } else { 0.0 })
        .collect();
    Ok(vec![da])
}

/// p = softmax(x)：dA[i] = p[i]·(in[i] − Σ_j in[j]·p[j])。
pub(super) fn softmax(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn gemv(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn gemm(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn embed(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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

/// 训练内核 `rope_f32` 的固定位置：与 `kernel.rs` 的 `rope_f32` 同口径（标量槽尚未接进训练内核）。
const ROPE_POS: f32 = 7.0;

/// `y = γ·x / rms(x)`：`dx[j] = g[j]·γ[j]/r − x[j]·Σ_i g[i]·γ[i]·x[i] / (n·r³)`，
/// `dγ[i] = g[i]·x[i]/r`。
pub(super) fn rmsnorm(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
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
pub(super) fn rope(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    Ok(vec![rope_grad(&ctx.inputs[0], ctx.incoming)])
}

/// 旋转表缓存槽：段长 + 该段长下的 `(sin, cos)` 表。
type RotationCache = RefCell<Option<(usize, Rc<[(f32, f32)]>)>>;

thread_local! {
    /// 段长 → 旋转表。`θ_i = ROPE_POS·base^(-2i/d)` 只依赖 `d`，与数据无关，所以 `(sin, cos)`
    /// 也可以一次性算好：逐对重算 `powf` 与 `sin_cos` 是这条路径上唯一的纯浪费。
    static ROTATIONS: RotationCache = const { RefCell::new(None) };
}

/// 段长 `d` 的旋转表：`d/2` 对 `(sin θ_i, cos θ_i)`，与逐对计算的 `angle.sin_cos()` 逐位相同。
fn rotations(d: usize) -> Rc<[(f32, f32)]> {
    ROTATIONS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some((cached, table)) = slot.as_ref()
            && *cached == d
        {
            return Rc::clone(table);
        }
        let table: Rc<[(f32, f32)]> = op::rope_f32::frequencies(d)
            .iter()
            .map(|&freq| (ROPE_POS * freq).sin_cos())
            .collect();
        *slot = Some((d, Rc::clone(&table)));
        table
    })
}

/// RoPE 反向本体：逐对把上游梯度按同一角度旋转回去；末尾不成对的元素原样透传。
fn rope_grad(x: &[f32], grad: &[f32]) -> Vec<f32> {
    let d = x.len();
    let pairs = d / 2;
    let mut dx = vec![0.0; d];
    let rotations = rotations(d);
    for (i, &(sin, cos)) in rotations.iter().enumerate() {
        dx[2 * i] = grad[2 * i] * cos + grad[2 * i + 1] * sin;
        dx[2 * i + 1] = -grad[2 * i] * sin + grad[2 * i + 1] * cos;
    }
    debug_assert_eq!(rotations.len(), pairs);
    if d % 2 == 1 {
        dx[d - 1] = grad[d - 1];
    }
    dx
}

/// 自注意力（`attention_f32` / `flash_attention_f32` 共用）：返回 `[dq, dk, dv]`。
pub(super) fn attention(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let d = op::attention::MODEL.shape.1;
    let s = divide(ctx.inputs[0].len(), d, "attention_f32")?;
    let (dq, dk, dv) = attention_grad(attention_ctx(&ctx, d), s, s);
    Ok(vec![dq, dk, dv])
}

/// 潜空间注意力（`latent_attention_f32`）：唯一差别是**写宽比读宽窄**——`dO` 与 `dV` 按压缩宽度
/// `d_c` 走，`dQ` / `dK` 仍按读宽 `W` 走。`d_c` 由上游梯度长度与查询行数反推，不额外声明。
pub(super) fn latent_attention(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let w = op::attention::latent::MODEL.shape.1;
    let q_rows = divide(ctx.inputs[0].len(), w, "latent_attention_f32")?;
    let hist = divide(ctx.inputs[1].len(), w, "latent_attention_f32")?;
    let write = ctx.incoming.len() / q_rows.max(1);
    let mut view = attention_ctx(&ctx, w);
    view.write = write;
    let (dq, dk, dv) = attention_grad(view, q_rows, hist);
    Ok(vec![dq, dk, dv])
}

/// decode 注意力（`decode_attention_f32` / `decode_flash_attention_f32` 共用）：查询行数与历史
/// 行数解耦，返回 `[dq, dk, dv]`。
pub(super) fn decode_attention(ctx: Ctx<'_>) -> Result<Vec<Vec<f32>>> {
    let d = op::decode_attention::MODEL.shape.1;
    let q_rows = divide(ctx.inputs[0].len(), d, "decode_attention_f32")?;
    let hist = divide(ctx.inputs[1].len(), d, "decode_attention_f32")?;
    let (dq, dk, dv) = attention_grad(attention_ctx(&ctx, d), q_rows, hist);
    Ok(vec![dq, dk, dv])
}

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
fn attention_grad(
    ctx: AttentionCtx<'_>,
    q_rows: usize,
    hist: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0; q_rows * ctx.d];
    let mut dk = vec![0.0; hist * ctx.d];
    let mut dv = vec![0.0; hist * ctx.write];
    for i in 0..q_rows {
        let row = attention_row(&ctx, i, hist);
        scatter(&row, &ctx, i, hist, &mut dq, &mut dk, &mut dv);
    }
    (dq, dk, dv)
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
    for slot in probs.iter_mut() {
        *slot = (*slot - max).exp();
        sum += *slot;
    }
    for slot in probs.iter_mut() {
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
    let (i_write, j_write) = (i * ctx.write, j * ctx.write);
    for h in 0..ctx.write {
        dv[j_write + h] += row.probs[j] * ctx.dout[i_write + h];
    }
    let (i_read, j_read) = (i * ctx.d, j * ctx.d);
    for h in 0..ctx.d {
        dq[i_read + h] += s * ctx.k[j_read + h];
        dk[j_read + h] += s * ctx.q[i_read + h];
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
