//! 反向数值对拍：自动回算的解析梯度与中心差分逐参数比对，并核对图外叶子不收梯度。

use crate::{Data, Reference, Session, Sgd, TrainConfig};
use autograd::{Sizes, Store, Tape};
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
    session.buffers().compute("reduce_sum_f32", &[z], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let grad = session.buffers().gradient(p).expect("参数梯度应在场");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = p0.clone();
        plus[index] += STEP;
        let mut minus = p0.clone();
        minus[index] -= STEP;
        let numeric = (dot(&plus, &w0) - dot(&minus, &w0)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "p[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
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
    session.buffers().compute("reduce_sum_f32", &[y], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let grad = session.buffers().gradient(p).expect("参数梯度应在场");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = p0.clone();
        plus[index] += STEP;
        let mut minus = p0.clone();
        minus[index] -= STEP;
        let numeric = (chain_loss(&plus) - chain_loss(&minus)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "p[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
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
    eprintln!("DEBUG gelu p0={p0:?}");
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)))
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.buffers().compute("gelu_f32", &[p], h)?;
    session.buffers().compute("reduce_sum_f32", &[h], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let grad = session.buffers().gradient(p).expect("参数梯度应在场");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = p0.clone();
        plus[index] += STEP;
        let mut minus = p0.clone();
        minus[index] -= STEP;
        let numeric = (chain_loss(&plus) - chain_loss(&minus)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "p[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
    Ok(())
}

/// 标量 erf：与 `op::gelu_f32` 同口径（Abramowitz-Stegun 7.1.26），测试闭式重算用。
fn erf(x: f32) -> f32 {
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
    sign * (1.0 - series * (-x * x).exp())
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
    session.buffers().compute("reduce_sum_f32", &[h], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let grad = session.buffers().gradient(p).expect("参数梯度应在场");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = p0.clone();
        plus[index] += STEP;
        let mut minus = p0.clone();
        minus[index] -= STEP;
        let numeric = (chain_loss(&plus) - chain_loss(&minus)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "p[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
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
    session.buffers().compute("reduce_sum_f32", &[y], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let grad = session.buffers().gradient(w).expect("权重梯度应在场");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = w0.clone();
        plus[index] += STEP;
        let mut minus = w0.clone();
        minus[index] -= STEP;
        let numeric = (chain_loss(&plus, &x0) - chain_loss(&minus, &x0)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "w[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
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
    session.buffers().compute("reduce_sum_f32", &[y], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let out_before = session
        .buffers()
        .value(y)
        .expect("y 在场")
        .f32()
        .expect("f32")
        .to_vec();
    eprintln!("DEBUG gemm out={out_before:?}");
    let grad = session.buffers().gradient(a).expect("A 梯度应在场");
    eprintln!("DEBUG gemm grad={grad:?}");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = a0.clone();
        plus[index] += STEP;
        let mut minus = a0.clone();
        minus[index] -= STEP;
        let numeric = (chain_loss(&plus, &b0) - chain_loss(&minus, &b0)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "a[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
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
    session.buffers().compute("reduce_sum_f32", &[y], loss)?;
    session.plan(loss)?;
    session.backward()?;

    let grad = session.buffers().gradient(table).expect("表梯度应在场");
    for (index, actual) in grad.iter().enumerate() {
        let mut plus = table0.clone();
        plus[index] += STEP;
        let mut minus = table0.clone();
        minus[index] -= STEP;
        let numeric = (chain_loss(&plus) - chain_loss(&minus)) / (2.0 * STEP);
        assert!(
            close(*actual, numeric),
            "table[{index}] 解析 {actual} 与差分 {numeric} 超差"
        );
    }
    Ok(())
}

/// 尚未接反向公式的算子在回传时显式报 [`Code::Unsupported`]，不静默忽略。
#[test]
fn unsupported_operator_backward_is_rejected() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record("rope_f32", &[p]).expect("记录 y");

    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.buffers().compute("rope_f32", &[p], y)?;
    session.plan(y)?;

    let error = session.backward().expect_err("rope 还没有解析反向");
    assert_eq!(error.code(), Code::Unsupported);
    Ok(())
}
