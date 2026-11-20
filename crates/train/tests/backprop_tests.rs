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

    let sizes = support::sizes_of(&tape, &[(p, N), (w, N), (z, N), (loss, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (y, N), (loss, 1)]);

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
    let (tape, sizes, p, x, loss) = support::add_graph(N);

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

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (loss, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(p, N), (h, 1), (loss, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(w, rows * cols), (x, cols), (y, rows), (loss, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(a, m * k), (b, k * n), (y, m * n), (loss, 1)]);

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

    let sizes = support::sizes_of(
        &tape,
        &[
            (table, rows * dim),
            (idx, tokens),
            (y, tokens * dim),
            (loss, 1),
        ],
    );

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

    let sizes = support::sizes_of(&tape, &[(p, N), (y, N)]);

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

    let sizes = support::sizes_of(&tape, &[(a, N), (b, N), (y, N), (out, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(p, N), (y, N), (out, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(x, N), (gamma, N), (y, N), (loss, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(x, N), (y, N), (loss, 1)]);

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

    let sizes = support::sizes_of(&tape, &[(q, len), (k, len), (v, len), (y, len), (loss, 1)]);

    let q0: Vec<f32> = (0..len).map(|i| 0.07 * i as f32 - 0.5).collect();
    let k0: Vec<f32> = (0..len).map(|i| -0.05 * i as f32 + 0.4).collect();
    let v0: Vec<f32> = (0..len).map(|i| 0.09 * i as f32 - 0.3).collect();

    let mut session = attention_session(tape, sizes, [q, k, v], [&q0, &k0, &v0])?;
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

    let sizes = support::sizes_of(
        &tape,
        &[(q, q_len), (k, kv_len), (v, kv_len), (y, q_len), (loss, 1)],
    );

    let q0: Vec<f32> = (0..q_len).map(|i| 0.11 * i as f32 - 0.4).collect();
    let k0: Vec<f32> = (0..kv_len).map(|i| -0.08 * i as f32 + 0.5).collect();
    let v0: Vec<f32> = (0..kv_len).map(|i| 0.12 * i as f32 - 0.2).collect();

    let mut session = attention_session(tape, sizes, [q, k, v], [&q0, &k0, &v0])?;
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

/// `loss = reduce_sum(leaky_relu(p))`：正区梯度 1、负区梯度 slope，逐元素对拍。
#[test]
fn leaky_relu_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("leaky_relu_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (loss, 1)]);
    let slope = op::leaky_relu_f32::SLOPE;
    let p0 = vec![-1.2_f32, 0.4, 1.1];
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| if x >= 0.0 { x } else { slope * x })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().compute("leaky_relu_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// `loss = reduce_sum(hard_sigmoid(p))`：线性区梯度恒为 1/6；饱和点解析梯度为 0。
#[test]
fn hard_sigmoid_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("hard_sigmoid_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (loss, 1)]);
    let p0 = vec![-1.5_f32, 0.4, 1.1];
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| (x / 6.0 + 0.5).clamp(0.0, 1.0))
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().compute("hard_sigmoid_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// 饱和点（|x| > 3）的解析梯度必须为 0，且不是差分近似能测得的小数。
#[test]
fn hard_sigmoid_is_flat_outside_the_linear_band() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("hard_sigmoid_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (loss, 1)]);
    let p0 = vec![-4.0_f32, 0.0, 4.0];
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0)?;
    session.buffers().compute("hard_sigmoid_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    let grad = session.buffers().gradient(p).expect("参数梯度应在场");
    assert_eq!(grad[0], 0.0, "下饱和梯度");
    assert!((grad[1] - 1.0 / 6.0).abs() < 1e-6, "中心梯度 {}", grad[1]);
    assert_eq!(grad[2], 0.0, "上饱和梯度");
    Ok(())
}

/// 一元"长度会变"的算子：`loss = Σ op(p)`，对 `p` 逐元素中心差分。
fn check_shaped_grad(
    op: &str,
    input: &[f32],
    output: usize,
    loss: impl Fn(&[f32]) -> f32,
) -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record(op, &[p]).expect("记录 y");
    let out = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let sizes = support::sizes_of(&tape, &[(p, input.len()), (y, output), (out, 1)]);
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, input.to_vec())?;
    session.buffers().compute(op, &[p], y)?;
    sum_backward(&mut session, y, out)?;
    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
        input,
        "p",
        loss,
    );
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
    session.buffers().compute(op, &[x, g, b], y)?;
    sum_backward(&mut session, y, out)?;

    let f = &loss;
    check_grad(
        session.buffers().gradient(x).expect("x 梯度应在场"),
        x0,
        "x",
        |px| f(px, g0, b0),
    );
    check_grad(
        session.buffers().gradient(g).expect("gamma 梯度应在场"),
        g0,
        "gamma",
        |pg| f(x0, pg, b0),
    );
    check_grad(
        session.buffers().gradient(b).expect("beta 梯度应在场"),
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

#[test]
fn max_pool1d_matches_central_differences() -> Result<()> {
    let total = op::max_pool1d_f32::L;
    let out = op::max_pool1d_f32::OL;
    let x0 = ramp(total);
    check_shaped_grad("max_pool1d_f32", &x0, out, |px| {
        let mut y = vec![0.0f32; out];
        op::max_pool1d_f32::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn max_pool2d_matches_central_differences() -> Result<()> {
    let side = op::max_pool2d_f32::H;
    let x0 = ramp(side * side);
    check_shaped_grad("max_pool2d_f32", &x0, side * side, |px| {
        let mut y = vec![0.0f32; side * side];
        op::max_pool2d_f32::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn avg_pool2d_matches_central_differences() -> Result<()> {
    let side = op::avg_pool2d_f32::H;
    let x0 = ramp(side * side);
    check_shaped_grad("avg_pool2d_f32", &x0, side * side, |px| {
        let mut y = vec![0.0f32; side * side];
        op::avg_pool2d_f32::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn adaptive_pool_matches_central_differences() -> Result<()> {
    let (side, out) = (op::adaptive_pool_f32::H, op::adaptive_pool_f32::OH);
    let x0 = ramp(side * side);
    check_shaped_grad("adaptive_pool_f32", &x0, out * out, |px| {
        let mut y = vec![0.0f32; out * out];
        op::adaptive_pool_f32::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn interpolate_matches_central_differences() -> Result<()> {
    let (h, w, out) = (
        op::interpolate_f32::H,
        op::interpolate_f32::W,
        op::interpolate_f32::OH,
    );
    let x0 = ramp(h * w);
    check_shaped_grad("interpolate_f32", &x0, out * out, |px| {
        let mut y = vec![0.0f32; out * out];
        op::interpolate_f32::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn batch_norm_matches_central_differences() -> Result<()> {
    use op::batch_norm_f32 as decl;
    let total = decl::C * decl::S * decl::N;
    let (x0, g0, b0) = (ramp(total), gain(), shift());
    check_affine_grad("batch_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        op::batch_norm_f32::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn group_norm_matches_central_differences() -> Result<()> {
    use op::group_norm_f32 as decl;
    let total = decl::C * decl::S * decl::N;
    let (x0, g0, b0) = (ramp(total), gain(), shift());
    check_affine_grad("group_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        op::group_norm_f32::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn instance_norm_matches_central_differences() -> Result<()> {
    use op::instance_norm_f32 as decl;
    let total = decl::C * decl::S * decl::N;
    let (x0, g0, b0) = (ramp(total), gain(), shift());
    check_affine_grad("instance_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        op::instance_norm_f32::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn layer_norm_matches_central_differences() -> Result<()> {
    use op::layer_norm_f32 as decl;
    let total = decl::R * decl::W;
    let x0 = ramp(total);
    let g0 = gain();
    let b0 = shift();
    check_affine_grad("layer_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        op::layer_norm_f32::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
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
    session.buffers().compute_shaped(op, &[p], y, geometry)?;
    sum_backward(&mut session, y, out)?;
    check_grad(
        session.buffers().gradient(p).expect("参数梯度应在场"),
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
    session.buffers().compute_shaped(op, &[x, g, b], y, geometry)?;
    sum_backward(&mut session, y, out)?;

    let f = &loss;
    check_grad(
        session.buffers().gradient(x).expect("x 梯度应在场"),
        x0,
        "x",
        |px| f(px, g0, b0),
    );
    check_grad(
        session.buffers().gradient(g).expect("gamma 梯度应在场"),
        g0,
        "gamma",
        |pg| f(x0, pg, b0),
    );
    check_grad(
        session.buffers().gradient(b).expect("beta 梯度应在场"),
        b0,
        "beta",
        |pb| f(x0, g0, pb),
    );
    Ok(())
}

#[test]
fn shaped_max_pool1d_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Pool1d { k: 3, s: 1, p: 1, l: 7, c: 1 };
    let x0 = ramp(7);
    check_geometry_grad("max_pool1d_f32", geometry, &x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_max_pool2d_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Pool2d { k: 3, s: 2, p: 0, h: 5, w: 5, c: 2 };
    let x0 = ramp(5 * 5 * 2);
    check_geometry_grad("max_pool2d_f32", geometry, &x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_avg_pool2d_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Avg2d {
        k: 3,
        s: 2,
        p: 1,
        h: 5,
        w: 5,
        c: 1,
        count_include_pad: false,
    };
    let x0 = ramp(5 * 5);
    check_geometry_grad("avg_pool2d_f32", geometry, &x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_adaptive_pool_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Adaptive { oh: 3, ow: 3, h: 6, w: 6, c: 1 };
    let x0 = ramp(36);
    check_geometry_grad("adaptive_pool_f32", geometry, &x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_interpolate_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Interp { oh: 5, ow: 5, h: 3, w: 3, c: 1, linear: true };
    let x0 = ramp(9);
    check_geometry_grad("interpolate_f32", geometry, &x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_batch_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Norm { n: 2, spatial: 3, channels: 5 };
    let (x0, g0, b0) = (ramp(30), gain5(), shift5());
    check_geometry_affine("batch_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_group_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Group { groups: 3, n: 2, spatial: 3, channels: 6 };
    let (x0, g0, b0) = (ramp(36), gain6(), shift6());
    check_geometry_affine("group_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_layer_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Layer { rows: 2, width: 5 };
    let x0 = ramp(10);
    let g0 = gain5();
    let b0 = shift5();
    check_geometry_affine("layer_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
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

/// 任意几何 + 真优化器闭环：`loss = reduce_sum(batch_norm(x, gamma, beta))`。
///
/// 标准化后每通道均值恰为 0，于是 `loss = Σ_c beta_c * n * spatial`，
/// `d loss / d beta_c = n * spatial` —— 这是个可解析判据，不靠拟合：
/// 第 `t` 步后 `beta_c = -t * lr * n * spatial`，`loss = -channels * t * lr * (n * spatial)^2`。
/// 同时覆盖 `Buffers::compute_shaped` 的几何登记通道在真实训练循环里能被反向取到。
#[test]
fn an_arbitrary_geometry_trains_end_to_end_under_sgd() -> Result<()> {
    let (n, spatial, channels) = (2usize, 3usize, 4usize);
    let len = n * spatial * channels;
    let feed: Vec<f32> = (0..len).map(|i| 0.25 + (i as f32 * 0.37).sin()).collect();
    let geom = Geometry::Norm { n, spatial, channels };

    let mut tape = Tape::new();
    let x = tape.leaf();
    let gamma = tape.parameter();
    let beta = tape.parameter();
    let y = tape.record("batch_norm_f32", &[x, gamma, beta]).expect("记录归一化");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录损失");
    let sizes = support::sizes_of(
        &tape,
        &[(x, len), (gamma, channels), (beta, channels), (y, len), (loss, 1)],
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
        session.buffers().load(x, Data::F32(feed.clone()))?;
        session.buffers().compute_shaped("batch_norm_f32", &[x, gamma, beta], y, geom)?;
        session.buffers().compute("reduce_sum_f32", &[y], loss)?;
        let now = session.record_loss(loss)?;
        let shift = session.slots()[1].params()[0];
        if t > 0 {
            assert!(now < last, "第 {t} 步损失没有下降：{last} → {now}");
            let predicted = slope * (shift - last_shift);
            assert!((now - last).abs() > 1e-3, "第 {t} 步损失没有实质变化");
            assert!((now - last - predicted).abs() <= 1e-4, "第 {t} 步 Δloss 与 Δbeta 不成比例：{} vs {predicted}", now - last);
        }
        last = now;
        last_shift = shift;
        session.backward()?;
        session.optimize()?;
    }
    assert!(last < -13.0, "100 步后损失只有 {last}，更新量不足");
    let mut shift = session.slots()[1].params().to_vec();
    shift.iter_mut().for_each(|b| *b += 100.0 * lr * positions);
    assert!(shift.iter().all(|b| b.abs() <= 5e-4), "偏置偏离解析式：{shift:?}");
    Ok(())
}
