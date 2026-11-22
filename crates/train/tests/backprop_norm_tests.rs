use super::*;

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
    session.bufs().compute_step("rmsnorm_f32", &[x, gamma], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(x).expect("x 梯度应在场"),
        &x0,
        "x",
        |x| rmsnorm_loss(x, &gamma0),
    );
    check_grad(
        session.bufs().gradient(gamma).expect("gamma 梯度应在场"),
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
    session.bufs().compute_step("rope_f32", &[x], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(x).expect("x 梯度应在场"),
        &x0,
        "x",
        rope_loss,
    );
    Ok(())
}

/// `loss = reduce_sum(attention(q,k,v))`：`dq` / `dk` / `dv` 三路闭式对拍（头宽固定 8）。
#[test]
fn attention_matches_central_differences() -> Result<()> {
    let d = nn::conv::attention::MODEL.shape.1;
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
    session
        .bufs()
        .compute_step("attention_f32", &[q, k, v], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(q).expect("q 梯度应在场"),
        &q0,
        "q",
        |q| attention_loss(q, &k0, &v0, s, s, d),
    );
    check_grad(
        session.bufs().gradient(k).expect("k 梯度应在场"),
        &k0,
        "k",
        |k| attention_loss(&q0, k, &v0, s, s, d),
    );
    check_grad(
        session.bufs().gradient(v).expect("v 梯度应在场"),
        &v0,
        "v",
        |v| attention_loss(&q0, &k0, v, s, s, d),
    );
    Ok(())
}

/// `loss = reduce_sum(latent_attention(q,k,v))`：读宽 `W`、写宽 `dc = W/2`，三路梯度闭式对拍。
///
/// 这一条守的是"写宽比读宽窄"口径：`dV` 比 `dK` 短，`dQ`/`dK` 仍按读宽散射，任何一处把两个宽度
/// 混用都会在这里被中心差分抓到。
#[test]
fn latent_attention_matches_central_differences() -> Result<()> {
    let w = nn::conv::attention::latent::MODEL.shape.1;
    let (rows, hist, dc) = (2usize, 2usize, w / 2);
    let (q_len, kv_len, out_len) = (rows * w, hist * w, rows * dc);
    let mut tape = Tape::new();
    let q = tape.parameter();
    let k = tape.parameter();
    let v = tape.parameter();
    let y = tape
        .record("latent_attention_f32", &[q, k, v])
        .expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let sizes = support::sizes_of(
        &tape,
        &[
            (q, q_len),
            (k, kv_len),
            (v, hist * dc),
            (y, out_len),
            (loss, 1),
        ],
    );

    let q0: Vec<f32> = (0..q_len).map(|i| 0.07 * i as f32 - 0.5).collect();
    let k0: Vec<f32> = (0..kv_len).map(|i| -0.05 * i as f32 + 0.4).collect();
    let v0: Vec<f32> = (0..hist * dc).map(|i| 0.09 * i as f32 - 0.3).collect();

    let mut session = attention_session(tape, sizes, [q, k, v], [&q0, &k0, &v0])?;
    session
        .bufs()
        .compute_step("latent_attention_f32", &[q, k, v], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(q).expect("q 梯度应在场"),
        &q0,
        "q",
        |q| latent_loss(q, &k0, &v0, rows, hist, w, dc),
    );
    check_grad(
        session.bufs().gradient(k).expect("k 梯度应在场"),
        &k0,
        "k",
        |k| latent_loss(&q0, k, &v0, rows, hist, w, dc),
    );
    check_grad(
        session.bufs().gradient(v).expect("v 梯度应在场"),
        &v0,
        "v",
        |v| latent_loss(&q0, &k0, v, rows, hist, w, dc),
    );
    Ok(())
}

/// `loss = reduce_sum(decode_attention(q,k,v))`：查询一行、历史两行，三路梯度闭式对拍。
#[test]
fn decode_attention_matches_central_differences() -> Result<()> {
    let d = nn::conv::decode_attention::MODEL.shape.1;
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
        .bufs()
        .compute_step("decode_attention_f32", &[q, k, v], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(q).expect("q 梯度应在场"),
        &q0,
        "q",
        |q| attention_loss(q, &k0, &v0, q_rows, hist, d),
    );
    check_grad(
        session.bufs().gradient(k).expect("k 梯度应在场"),
        &k0,
        "k",
        |k| attention_loss(&q0, k, &v0, q_rows, hist, d),
    );
    check_grad(
        session.bufs().gradient(v).expect("v 梯度应在场"),
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
