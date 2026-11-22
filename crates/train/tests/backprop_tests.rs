//! 反向数值对拍：自动回算的解析梯度与中心差分逐参数比对，并核对图外叶子不收梯度。

use crate::{Data, Geometry, Reference, Session, Sgd, TrainConfig};
use autograd::{Sizes, Store, Tape, Var};
use diagnostic::{Code, Result};
use nn::ParamRole;
use tensor::{DType, Shape};

use crate::support;

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
    session.bufs().compute_step("reduce_sum_f32", &[y], loss)?;
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

/// 中心差分：把 `base[idx]` 扰动 ±STEP，用闭式 `f` 重算损失。
fn central(f: impl Fn(&[f32]) -> f32, base: &[f32], idx: usize) -> f32 {
    let mut plus = base.to_vec();
    plus[idx] += STEP;
    let mut minus = base.to_vec();
    minus[idx] -= STEP;
    (f(&plus) - f(&minus)) / (2.0 * STEP)
}

/// 逐元素对拍解析梯度与中心差分：`f` 是参数向量的闭式损失，`name` 只进断言文案。
fn check_grad(grad: &[f32], base: &[f32], name: &str, f: impl Fn(&[f32]) -> f32) {
    for (idx, actual) in grad.iter().enumerate() {
        let numeric = central(&f, base, idx);
        assert!(
            close(*actual, numeric),
            "{name}[{idx}] 解析 {actual} 与差分 {numeric} 超差"
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

    let sizes = support::sizes_of(&tape, &[(a, N), (b, N), (y, N), (out, 1)]);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(a, ParamRole::Trainable, a0.to_vec())?;
    session.register_parameter(b, ParamRole::Trainable, b0.to_vec())?;
    session.bufs().compute_step(op, &[a, b], y)?;
    sum_backward(&mut session, y, out)?;

    // `loss` 借给两个闭包（按引用捕获，避免一次 move 后第二次取不到）。
    let f = &loss;
    check_grad(
        session.bufs().gradient(a).expect("a 梯度应在场"),
        a0,
        "a",
        |pa| f(pa, b0),
    );
    check_grad(
        session.bufs().gradient(b).expect("b 梯度应在场"),
        b0,
        "b",
        |pb| f(a0, pb),
    );
    Ok(())
}

/// `loss = reduce_sum(op(p))` 的单路反向夹具：建图、落参数、跑 `compute` 与 `sum_backward`，返回 `p` 的梯度。
///
/// 一元算子的两类对拍（中心差分、饱和点解析断言）与"长度会变"的算子（池化 / 插值）此前
/// 各写一份这四步；`output_len` 就是为后者准备的。
fn unary_grad(op: &str, p0: &[f32], output_len: usize) -> Result<Vec<f32>> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record(op, &[p]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let sizes = support::sizes_of(&tape, &[(p, p0.len()), (y, output_len), (out, 1)]);
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.to_vec())?;
    session.bufs().compute_step(op, &[p], y)?;
    sum_backward(&mut session, y, out)?;
    Ok(session.bufs().gradient(p).expect("p 梯度应在场").to_vec())
}

/// `loss = reduce_sum(op(p))` 的公共夹具：单路参数中心差分，`loss(p)` 给整标量。
/// `exp`/`log`/`rope` 这类"单参数、单算子、reduce_sum 收尾"的反向对拍共用。
fn check_unary_grad(op: &str, p0: &[f32], loss: impl Fn(&[f32]) -> f32) -> Result<()> {
    check_grad(&unary_grad(op, p0, p0.len())?, p0, "p", loss);
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

/// 潜空间注意力闭式：`O = softmax(QKᵀ/√W)·V`，但 `V` 与输出只有 `dc` 列（读宽 `W`、写宽 `dc`），
/// 返回全部输出的和。
fn latent_loss(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    hist: usize,
    w: usize,
    dc: usize,
) -> f32 {
    let scale = 1.0 / (w as f32).sqrt();
    let mut total = 0.0;
    for i in 0..rows {
        let scores: Vec<f32> = (0..hist).map(|j| row_dot(q, k, i, j, w) * scale).collect();
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        total += (0..dc)
            .map(|h| {
                (0..hist)
                    .map(|j| (exps[j] / sum) * v[j * dc + h])
                    .sum::<f32>()
            })
            .sum::<f32>();
    }
    total
}

/// 注意力三路对拍共用的会话装配：开裸会话后把 `q` / `k` / `v` 按初值登记成可训练参数。
fn attention_session(
    tape: Tape,
    sizes: Sizes,
    params: [Var; 3],
    values: [&[f32]; 3],
) -> Result<Session<Reference, Sgd>> {
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    for (param, value) in params.into_iter().zip(values) {
        session.register_parameter(param, ParamRole::Trainable, value.to_vec())?;
    }
    Ok(session)
}

/// 一元"长度会变"的算子：`loss = Σ op(p)`，对 `p` 逐元素中心差分。
fn check_shaped_grad(
    op: &str,
    input: &[f32],
    output: usize,
    loss: impl Fn(&[f32]) -> f32,
) -> Result<()> {
    check_grad(&unary_grad(op, input, output)?, input, "p", loss);
    Ok(())
}

/// 三元仿射归一化：`loss = Σ op(x, gamma, beta)`，三个输入各做一次中心差分。
fn check_affine_grad(
    op: &str,
    x0: &[f32],
    g0: &[f32],
    b0: &[f32],
    total: usize,
    loss: impl Fn(&[f32], &[f32], &[f32]) -> f32,
) -> Result<()> {
    check_affine(op, x0, g0, b0, total, None, loss)
}

/// 三元仿射对拍的共用骨架：声明档（`geometry = None` → `compute`）与实例几何
/// （`Some(..)` → `compute_shaped`）只差这一步，收尾（plan + backward + 中心差分）一致。
fn check_affine(
    op: &str,
    x0: &[f32],
    g0: &[f32],
    b0: &[f32],
    total: usize,
    geometry: Option<Geometry>,
    loss: impl Fn(&[f32], &[f32], &[f32]) -> f32,
) -> Result<()> {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let g = tape.parameter();
    let b = tape.parameter();
    let y = tape.record(op, &[x, g, b]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let sizes = support::sizes_of(
        &tape,
        &[
            (x, x0.len()),
            (g, g0.len()),
            (b, b0.len()),
            (y, total),
            (out, 1),
        ],
    );
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(x, ParamRole::Trainable, x0.to_vec())?;
    session.register_parameter(g, ParamRole::Trainable, g0.to_vec())?;
    session.register_parameter(b, ParamRole::Trainable, b0.to_vec())?;
    match geometry {
        Some(geometry) => session
            .bufs()
            .compute_shaped_step(op, &[x, g, b], y, geometry)?,
        None => session.bufs().compute_step(op, &[x, g, b], y)?,
    }
    sum_backward(&mut session, y, out)?;

    let f = &loss;
    check_grad(
        session.bufs().gradient(x).expect("x 梯度应在场"),
        x0,
        "x",
        |px| f(px, g0, b0),
    );
    check_grad(
        session.bufs().gradient(g).expect("gamma 梯度应在场"),
        g0,
        "gamma",
        |pg| f(x0, pg, b0),
    );
    check_grad(
        session.bufs().gradient(b).expect("beta 梯度应在场"),
        b0,
        "beta",
        |pb| f(x0, g0, pb),
    );
    Ok(())
}

/// 16 个互不相同的值：让池化的 argmax 落在可辨识的位置，也避免平手。
fn ramp(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32) * 0.37 - 2.1 + ((i % 5) as f32) * 0.11)
        .collect()
}

/// 4 个通道的仿射参数。
fn gain() -> Vec<f32> {
    vec![1.5, 0.8, 1.2, 0.6]
}

/// 4 个通道的偏置参数。
fn shift() -> Vec<f32> {
    vec![0.25, -0.4, 0.1, 0.3]
}

/// 实例几何的一元对拍：前向走 `compute_shaped`，反向按同一实例回传，再对 `p` 做中心差分。
fn check_geometry_grad(
    op: &str,
    geometry: Geometry,
    x0: &[f32],
    loss: impl Fn(&[f32]) -> f32,
) -> Result<()> {
    let total = geometry.outputs();
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record(op, &[p]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let sizes = support::sizes_of(&tape, &[(p, x0.len()), (y, total), (out, 1)]);
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, x0.to_vec())?;
    session.bufs().compute_shaped_step(op, &[p], y, geometry)?;
    sum_backward(&mut session, y, out)?;
    check_grad(
        session.bufs().gradient(p).expect("参数梯度应在场"),
        x0,
        "p",
        loss,
    );
    Ok(())
}

/// 实例几何的三元对拍：`(x, gamma, beta)` 各自中心差分。
fn check_geometry_affine(
    op: &str,
    geometry: Geometry,
    x0: &[f32],
    g0: &[f32],
    b0: &[f32],
    loss: impl Fn(&[f32], &[f32], &[f32]) -> f32,
) -> Result<()> {
    let total = geometry.outputs();
    check_affine(op, x0, g0, b0, total, Some(geometry), loss)
}

/// 5 个通道的仿射参数。
fn gain5() -> Vec<f32> {
    (0..5).map(|i| 0.6 + i as f32 * 0.2).collect()
}

/// 5 个通道的偏置参数。
fn shift5() -> Vec<f32> {
    (0..5).map(|i| 0.3 - i as f32 * 0.15).collect()
}

/// 6 个通道的仿射参数。
fn gain6() -> Vec<f32> {
    (0..6).map(|i| 0.5 + i as f32 * 0.18).collect()
}

/// 6 个通道的偏置参数。
fn shift6() -> Vec<f32> {
    (0..6).map(|i| 0.4 - i as f32 * 0.12).collect()
}

#[path = "backprop_graph_tests.rs"]
mod graph;

#[path = "backprop_norm_tests.rs"]
mod norm;

#[path = "backprop_rl_tests.rs"]
mod rl;

#[path = "backprop_pool_tests.rs"]
mod pool;

#[path = "backprop_train_tests.rs"]
mod train;

/// 暂存复用：第二次反传不再向分配器要内存（容量之和与热身后的完全一致）。
///
/// 反向每个节点都要一份"上游梯度 / 各输入前向值 / 输出前向值"的只读快照，因为这些快照与
/// `&mut bufs` 冲突；快照放在 `Bufs` 自持的 `Scratch` 里逐节点复用，于是稳态零分配。
/// 此断言盯的正是这一点：容量只增不减，故"容量不变"等价于"没有重新分配"。
#[test]
fn a_warm_backward_pass_reuses_its_scratch() -> Result<()> {
    let p0 = [0.5f32, -1.5, 2.0];
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record("silu_f32", &[p]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let sizes = support::sizes_of(&tape, &[(p, N), (y, N), (out, 1)]);
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.to_vec())?;
    session.bufs().compute_step("silu_f32", &[p], y)?;
    assert_eq!(
        session.bufs().scratch.footprint(),
        0,
        "反传前暂存应当还是空的"
    );

    sum_backward(&mut session, y, out)?;
    let warm = session.bufs().scratch.footprint();
    assert!(warm > 0, "热过一次之后暂存应当已占下容量");

    sum_backward(&mut session, y, out)?;
    assert_eq!(
        session.bufs().scratch.footprint(),
        warm,
        "第二次反传不得再增长暂存（容量不变即没有重新分配）"
    );
    Ok(())
}
