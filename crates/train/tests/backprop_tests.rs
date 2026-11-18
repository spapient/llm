//! 反向数值对拍：自动回算的解析梯度与中心差分逐参数比对，并核对图外叶子不收梯度。

use crate::{Data, Reference, Session, Sgd, TrainConfig};
use autograd::{Sizes, Store, Tape, Var};
use diagnostic::{Code, Result};
use nn::ParamRole;
use tensor::{DType, Shape};

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(w, Shape::new([N]), DType::F32);
    sizes.set(z, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(h, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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
    let mut tape = Tape::new();
    let p = tape.parameter();
    let x = tape.leaf();
    let loss = tape.record("add_f32", &[p, x]).expect("记录 loss");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(h, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(h, Shape::new([1]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(w, Shape::new([rows * cols]), DType::F32);
    sizes.set(x, Shape::new([cols]), DType::F32);
    sizes.set(y, Shape::new([rows]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(a, Shape::new([m * k]), DType::F32);
    sizes.set(b, Shape::new([k * n]), DType::F32);
    sizes.set(y, Shape::new([m * n]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(table, Shape::new([rows * dim]), DType::F32);
    sizes.set(idx, Shape::new([tokens]), DType::F32);
    sizes.set(y, Shape::new([tokens * dim]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(gamma, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(q, Shape::new([len]), DType::F32);
    sizes.set(k, Shape::new([len]), DType::F32);
    sizes.set(v, Shape::new([len]), DType::F32);
    sizes.set(y, Shape::new([len]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let q0: Vec<f32> = (0..len).map(|i| 0.07 * i as f32 - 0.5).collect();
    let k0: Vec<f32> = (0..len).map(|i| -0.05 * i as f32 + 0.4).collect();
    let v0: Vec<f32> = (0..len).map(|i| 0.09 * i as f32 - 0.3).collect();

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(q, ParamRole::Trainable, q0.clone())?;
    session.register_parameter(k, ParamRole::Trainable, k0.clone())?;
    session.register_parameter(v, ParamRole::Trainable, v0.clone())?;
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

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(q, Shape::new([q_len]), DType::F32);
    sizes.set(k, Shape::new([kv_len]), DType::F32);
    sizes.set(v, Shape::new([kv_len]), DType::F32);
    sizes.set(y, Shape::new([q_len]), DType::F32);
    sizes.set(loss, Shape::new([1]), DType::F32);

    let q0: Vec<f32> = (0..q_len).map(|i| 0.11 * i as f32 - 0.4).collect();
    let k0: Vec<f32> = (0..kv_len).map(|i| -0.08 * i as f32 + 0.5).collect();
    let v0: Vec<f32> = (0..kv_len).map(|i| 0.12 * i as f32 - 0.2).collect();

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(q, ParamRole::Trainable, q0.clone())?;
    session.register_parameter(k, ParamRole::Trainable, k0.clone())?;
    session.register_parameter(v, ParamRole::Trainable, v0.clone())?;
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
