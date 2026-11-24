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

/// 单行 softmax（与 `nn::softmax_f32` 同口径：整缓冲当作一行，先减 max 再 exp）。
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
    let rms = (mean_sq + nn::rmsnorm_f32::EPS).sqrt();
    x.iter().zip(gamma).map(|(x, g)| g * x / rms).sum()
}

/// RoPE 闭式：位置固定 7，偶数下标与奇数下标成对旋转，奇数长度末尾透传；返回逐元素和。
fn rope_loss(x: &[f32]) -> f32 {
    let d = x.len();
    let pairs = d / 2;
    let mut out = vec![0.0; d];
    for i in 0..pairs {
        let freq = nn::rope_f32::BASE.powf(-(2.0 * i as f32) / d as f32);
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

/// 任意元数参数图夹具：统一建图、形状表、参数登记与声明档/实例几何前向。
fn parameter_session(
    op: &str,
    values: &[&[f32]],
    output: usize,
    geometry: Option<Geometry>,
) -> Result<(Session<Reference, Sgd>, Vec<Var>, Var, Var)> {
    let mut tape = Tape::new();
    let params: Vec<Var> = values.iter().map(|_| tape.parameter()).collect();
    let y = tape.record(op, &params).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let mut entries: Vec<_> = params
        .iter()
        .zip(values)
        .map(|(&var, value)| (var, value.len()))
        .collect();
    entries.extend([(y, output), (out, 1)]);
    let sizes = support::sizes_of(&tape, &entries);
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    for (&param, value) in params.iter().zip(values) {
        session.register_parameter(param, ParamRole::Trainable, value.to_vec())?;
    }
    match geometry {
        Some(geometry) => session.bufs().compute_shaped_step(op, &params, y, geometry)?,
        None => session.bufs().compute_step(op, &params, y)?,
    }
    Ok((session, params, y, out))
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

/// 任意几何 + 真优化器闭环：`loss = reduce_sum(batch_norm(x, gamma, beta))`。
///
/// 标准化后每通道均值恰为 0，于是 `loss = Σ_c beta_c * n * spatial`，
/// `d loss / d beta_c = n * spatial` —— 这是个可解析判据，不靠拟合：
/// 第 `t` 步后 `beta_c = -t * lr * n * spatial`，`loss = -channels * t * lr * (n * spatial)^2`。
/// 同时覆盖 `Bufs::compute_shaped` 的几何登记通道在真实训练循环里能被反向取到。
#[test]
fn an_arbitrary_geometry_trains_end_to_end_under_sgd() -> Result<()> {
    let (n, spatial, channels) = (2usize, 3usize, 4usize);
    let len = n * spatial * channels;
    let feed: Vec<f32> = (0..len).map(|i| 0.25 + (i as f32 * 0.37).sin()).collect();
    let geom = Geometry::Norm {
        n,
        spatial,
        channels,
    };

    let mut tape = Tape::new();
    let x = tape.leaf();
    let gamma = tape.parameter();
    let beta = tape.parameter();
    let y = tape
        .record("batch_norm_f32", &[x, gamma, beta])
        .expect("记录归一化");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录损失");
    let sizes = support::sizes_of(
        &tape,
        &[
            (x, len),
            (gamma, channels),
            (beta, channels),
            (y, len),
            (loss, 1),
        ],
    );
    // 总步数放大，让学习率调度在 100 步内几乎不动（默认 `max_steps = 1` 会在第 1 步后归零）。
    let mut cfg = config();
    cfg.max_steps = 10_000;
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), cfg)?;
    session.register_parameter(gamma, ParamRole::Trainable, vec![1.0; channels])?;
    session.register_parameter(beta, ParamRole::Trainable, vec![0.0; channels])?;
    session.plan(loss)?;

    // 判据与学习率调度无关：`loss = Σ_c beta_c * n * spatial`，故
    // `Δloss = n * spatial * Σ_c Δbeta_c` —— 每条字段逐位对得上就不是巧合。
    let positions = (n * spatial) as f32;
    let slope = positions * channels as f32;
    let mut last = f32::INFINITY;
    let mut last_shift = 0.0f32;
    for t in 0..100u32 {
        session.bufs().load(x, Data::F32(feed.clone()))?;
        session
            .bufs()
            .compute_shaped_step("batch_norm_f32", &[x, gamma, beta], y, geom)?;
        session.bufs().compute_step("reduce_sum_f32", &[y], loss)?;
        let now = session.record_loss(loss)?;
        let shift = session.slots()[1].params()[0];
        if t > 0 {
            assert!(now < last, "第 {t} 步损失没有下降：{last} → {now}");
            let predicted = slope * (shift - last_shift);
            assert!((now - last).abs() > 1e-3, "第 {t} 步损失没有实质变化");
            assert!(
                (now - last - predicted).abs() <= 1e-4,
                "第 {t} 步 Δloss 与 Δbeta 不成比例：{} vs {predicted}",
                now - last
            );
        }
        last = now;
        last_shift = shift;
        session.backward()?;
        session.optimize()?;
    }
    assert!(last < -13.0, "100 步后损失只有 {last}，更新量不足");
    // 各通道的 beta 梯度恒为 `n * spatial`，所以四个通道必须同步下降。
    let shift = session.slots()[1].params();
    assert!(
        shift.windows(2).all(|w| (w[0] - w[1]).abs() <= 1e-6),
        "偏置通道不同步：{shift:?}"
    );
    assert!(
        shift.iter().all(|&b| b < -0.5),
        "偏置没有实质下降：{shift:?}"
    );
    Ok(())
}

/// 跨步常驻：同一 `Bufs` 上两轮带几何的归一化，滚动统计按动量递推并可读回。
///
/// 初始 `(0, 1)`、动量 0.1，同一批数据过两次：`mean` 应为 `0.1 * μ` 再 `0.19 * μ`，
/// `var` 应为 `0.9 + 0.1 * u` 再 `0.81 + 0.19 * u`（`u` 是无偏 batch 方差）。
#[test]
fn rolling_statistics_persist_across_steps_in_the_bufs() -> Result<()> {
    let (n, spatial, channels) = (2usize, 3usize, 2usize);
    let len = n * spatial * channels;
    let count = n * spatial;
    let geom = Geometry::Norm {
        n,
        spatial,
        channels,
    };
    let feed: Vec<f32> = (0..len).map(|i| 0.2 + i as f32 * 0.13).collect();

    let mut tape = Tape::new();
    let x = tape.leaf();
    let g = tape.leaf();
    let b = tape.leaf();
    let y = tape
        .record("batch_norm_f32", &[x, g, b])
        .expect("记录归一化");
    let sizes = support::sizes_of(&tape, &[(x, len), (g, channels), (b, channels), (y, len)]);
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.bufs().load(x, Data::F32(feed.clone()))?;
    session.bufs().load(g, Data::F32(vec![1.0; channels]))?;
    session.bufs().load(b, Data::F32(vec![0.0; channels]))?;

    session
        .bufs()
        .compute_shaped_step("batch_norm_f32", &[x, g, b], y, geom)?;
    let once = session.bufs().running(y).expect("第一轮应留下统计").clone();
    session
        .bufs()
        .compute_shaped_step("batch_norm_f32", &[x, g, b], y, geom)?;
    let twice = session.bufs().running(y).expect("第二轮应继续递推").clone();

    for ci in 0..channels {
        let values: Vec<f32> = (0..count).map(|p| feed[p * channels + ci]).collect();
        let mean = values.iter().sum::<f32>() / count as f32;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / count as f32;
        let unbiased = var * count as f32 / (count - 1) as f32;
        assert!(
            (once.mean[ci] - 0.1 * mean).abs() <= 1e-5,
            "一步均值 {} vs {}",
            once.mean[ci],
            0.1 * mean
        );
        assert!(
            (twice.mean[ci] - 0.19 * mean).abs() <= 1e-5,
            "两步均值 {} vs {}",
            twice.mean[ci],
            0.19 * mean
        );
        assert!(
            (once.var[ci] - (0.9 + 0.1 * unbiased)).abs() <= 1e-5,
            "一步方差 {}",
            once.var[ci]
        );
        assert!(
            (twice.var[ci] - (0.81 + 0.19 * unbiased)).abs() <= 1e-5,
            "两步方差 {}",
            twice.var[ci]
        );
    }
    Ok(())
}

/// `loss = Σ conv2d(x, weight, bias)`：三个输入各做一次中心差分，声明档与任意步长/补零两种几何。
#[test]
fn conv2d_gradients_match_central_differences_on_any_geometry() -> Result<()> {
    let shapes = [
        Geometry::declared("conv2d_f32")?,
        Geometry::Conv2d {
            k: 3,
            s: 2,
            p: 0,
            h: 5,
            w: 5,
            cin: 2,
            cout: 2,
        },
    ];
    for geom in shapes {
        let Geometry::Conv2d {
            k, h, w, cin, cout, ..
        } = geom
        else {
            unreachable!("本测试只跑卷积几何");
        };
        let x: Vec<f32> = (0..h * w * cin)
            .map(|i| 0.2 + (i as f32 * 0.31).sin())
            .collect();
        let weight: Vec<f32> = (0..k * k * cin * cout)
            .map(|i| 0.1 + (i as f32 * 0.17).cos() * 0.2)
            .collect();
        let bias: Vec<f32> = (0..cout).map(|i| 0.3 - i as f32 * 0.1).collect();
        let loss = move |px: &[f32], pw: &[f32], pb: &[f32]| -> f32 {
            let mut y = vec![0.0f32; geom.outputs()];
            geom.forward(&[px, pw, pb], &mut y).expect("前向");
            y.iter().sum()
        };
        check_geometry_affine("conv2d_f32", geom, &x, &weight, &bias, loss)?;
    }
    Ok(())
}

/// 声明档前向：`Bufs::compute("conv2d_f32")` 的适配路径与算子自身的 reference 逐位一致。
#[test]
fn conv2d_reference_kernel_matches_the_operator_sample() -> Result<()> {
    let sample = nn::conv::conv2d::sample(0, 0);
    let x = sample.f32("x")?.to_vec();
    let weight = sample.f32("weight")?.to_vec();
    let bias = sample.f32("bias")?.to_vec();
    let want = nn::conv::conv2d::reference(&sample)?;

    let mut tape = Tape::new();
    let a = tape.leaf();
    let b = tape.leaf();
    let c = tape.leaf();
    let y = tape.record("conv2d_f32", &[a, b, c]).expect("记录");
    let sizes = support::sizes_of(
        &tape,
        &[
            (a, x.len()),
            (b, weight.len()),
            (c, bias.len()),
            (y, want.len()),
        ],
    );
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.bufs().load(a, Data::F32(x))?;
    session.bufs().load(b, Data::F32(weight))?;
    session.bufs().load(c, Data::F32(bias))?;
    session.bufs().compute_step("conv2d_f32", &[a, b, c], y)?;
    let got = session.bufs().value(y).expect("y 应在场");
    let got = got.f32().expect("y 是 f32");
    assert_eq!(got.len(), want.len());
    for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-6, "第 {idx} 个输出 {g} vs {w}");
    }
    Ok(())
}
