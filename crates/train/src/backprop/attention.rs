//! 注意力反向公式：自注意力 / 潜空间 / decode 三种前向共享的逐行反向本体。

use super::Ctx;

/// 由公式上下文拼出注意力反向的只读视野。
pub(super) fn attention_ctx<'a>(ctx: &Ctx<'a>, d: usize) -> AttentionCtx<'a> {
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
pub(super) struct AttentionCtx<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    dout: &'a [f32],
    d: usize,
    /// 写宽：`V` / `dO` / `dV` 的列宽；稠密注意力与读宽相同，潜空间注意力更窄。
    pub(super) write: usize,
    scale: f32,
}

/// 一条查询行的反向中间量：softmax 概率与分数梯度（`√d` 缩放前的口径）。
struct AttentionRow {
    probs: Vec<f32>,
    ds: Vec<f32>,
}

/// 注意力反向本体：逐查询行算概率与分数梯度，再散射回 `dq` / `dk` / `dv`。
pub(super) fn attention_grad(
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
