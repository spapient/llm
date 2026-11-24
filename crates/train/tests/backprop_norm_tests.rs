use super::*;

/// `loss = reduce_sum(rmsnorm(x, gamma))`：`dx` 与 `dgamma` 都闭式对拍。
#[test]
fn rmsnorm_matches_central_differences() -> Result<()> {
    check_binary_grad(
        "rmsnorm_f32",
        &[0.4, -0.9, 1.3],
        &[0.7, 1.1, 0.5],
        rmsnorm_loss,
    )
}

/// `loss = reduce_sum(rope(x))`：旋转正交，反向即转置；奇数长度末尾透传。
#[test]
fn rope_matches_central_differences() -> Result<()> {
    check_unary_grad("rope_f32", &[0.5, -0.7, 1.1], rope_loss)
}

/// `loss = reduce_sum(attention(q,k,v))`：`dq` / `dk` / `dv` 三路闭式对拍（头宽固定 8）。
#[test]
fn attention_matches_central_differences() -> Result<()> {
    let (d, s) = (nn::conv::attention::MODEL.shape.1, 2);
    let len = s * d;
    let q0 = (0..len).map(|i| 0.07 * i as f32 - 0.5).collect();
    let k0 = (0..len).map(|i| -0.05 * i as f32 + 0.4).collect();
    let v0 = (0..len).map(|i| 0.09 * i as f32 - 0.3).collect();
    check_attention_grad(
        "attention_f32",
        [len, len, len, len, 1],
        q0,
        k0,
        v0,
        |q, k, v| attention_loss(q, k, v, s, s, d),
    )
}

/// `latent_attention` 的写宽为 `dc = W/2`，三路梯度仍分别做中心差分。
#[test]
fn latent_attention_matches_central_differences() -> Result<()> {
    let w = nn::conv::attention::latent::MODEL.shape.1;
    let (rows, hist, dc) = (2, 2, w / 2);
    let (q_len, kv_len, out_len) = (rows * w, hist * w, rows * dc);
    let q0 = (0..q_len).map(|i| 0.07 * i as f32 - 0.5).collect();
    let k0 = (0..kv_len).map(|i| -0.05 * i as f32 + 0.4).collect();
    let v0 = (0..hist * dc).map(|i| 0.09 * i as f32 - 0.3).collect();
    check_attention_grad(
        "latent_attention_f32",
        [q_len, kv_len, hist * dc, out_len, 1],
        q0,
        k0,
        v0,
        |q, k, v| latent_loss(q, k, v, rows, hist, w, dc),
    )
}

/// `decode_attention` 查询一行、历史两行，三路梯度闭式对拍。
#[test]
fn decode_attention_matches_central_differences() -> Result<()> {
    let d = nn::conv::decode_attention::MODEL.shape.1;
    let (q_rows, hist) = (1, 2);
    let (q_len, kv_len) = (q_rows * d, hist * d);
    let q0 = (0..q_len).map(|i| 0.11 * i as f32 - 0.4).collect();
    let k0 = (0..kv_len).map(|i| -0.08 * i as f32 + 0.5).collect();
    let v0 = (0..kv_len).map(|i| 0.12 * i as f32 - 0.2).collect();
    check_attention_grad(
        "decode_attention_f32",
        [q_len, kv_len, kv_len, q_len, 1],
        q0,
        k0,
        v0,
        |q, k, v| attention_loss(q, k, v, q_rows, hist, d),
    )
}

/// 三分支注意力共享建图、装配、反传与逐参数中心差分；`lengths` 顺序为 q/k/v/y/loss。
fn check_attention_grad(
    op: &str,
    lengths: [usize; 5],
    q0: Vec<f32>,
    k0: Vec<f32>,
    v0: Vec<f32>,
    objective: impl Fn(&[f32], &[f32], &[f32]) -> f32,
) -> Result<()> {
    let mut tape = Tape::new();
    let (q, k, v) = (tape.parameter(), tape.parameter(), tape.parameter());
    let y = tape.record(op, &[q, k, v]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");
    let sizes = support::sizes_of(
        &tape,
        &[
            (q, lengths[0]),
            (k, lengths[1]),
            (v, lengths[2]),
            (y, lengths[3]),
            (loss, 1),
        ],
    );
    let mut session = attention_session(tape, sizes, [q, k, v], [&q0, &k0, &v0])?;
    session.bufs().compute_step(op, &[q, k, v], y)?;
    sum_backward(&mut session, y, loss)?;
    let f = &objective;
    check_grad(
        session.bufs().gradient(q).expect("q 梯度"),
        &q0,
        "q",
        |x| f(x, &k0, &v0),
    );
    check_grad(
        session.bufs().gradient(k).expect("k 梯度"),
        &k0,
        "k",
        |x| f(&q0, x, &v0),
    );
    check_grad(
        session.bufs().gradient(v).expect("v 梯度"),
        &v0,
        "v",
        |x| f(&q0, &k0, x),
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

/// 几何反向对拍：前向取元素和，交给 `check_geometry_grad` 做中心差分。
fn sum_grad(op: &str, geometry: Geometry, x0: &[f32]) -> Result<()> {
    check_geometry_grad(op, geometry, x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

/// 声明档一元前向的中心差分测试由同一张参数表生成，测试名与覆盖点保持独立。
macro_rules! declared_sum_grad {
    ($( $name:ident => ($module:ident, $input:expr, $output:expr) ),* $(,)?) => {$(
        #[test]
        fn $name() -> Result<()> {
            let (input, output) = ($input, $output);
            check_shaped_grad(concat!(stringify!($module), "_f32"), &ramp(input), output, |px| {
                let mut y = vec![0.0f32; output];
                nn::conv::$module::forward(tensor::View::one(px), &mut y).expect("前向");
                y.iter().sum()
            })
        }
    )*};
}

declared_sum_grad! {
    max_pool1d_matches_central_differences => (max_pool1d, nn::conv::max_pool1d::L, nn::conv::max_pool1d::OL),
    max_pool2d_matches_central_differences => (max_pool2d, nn::conv::max_pool2d::H * nn::conv::max_pool2d::W, nn::conv::max_pool2d::H * nn::conv::max_pool2d::W),
    avg_pool2d_matches_central_differences => (avg_pool2d, nn::conv::avg_pool2d::H * nn::conv::avg_pool2d::W, nn::conv::avg_pool2d::H * nn::conv::avg_pool2d::W),
    adaptive_pool_matches_central_differences => (adaptive_pool, nn::conv::adaptive_pool::H * nn::conv::adaptive_pool::W, nn::conv::adaptive_pool::OH * nn::conv::adaptive_pool::OW),
    interpolate_matches_central_differences => (interpolate, nn::conv::interpolate::H * nn::conv::interpolate::W, nn::conv::interpolate::OH * nn::conv::interpolate::OW),
}

/// 声明档三元输入共用同一中心差分骨架；每条仍生成独立命名的测试。
macro_rules! declared_affine_grad {
    ($( $name:ident => ($module:ident, $total:expr) ),* $(,)?) => {$(
        #[test]
        fn $name() -> Result<()> {
            let total = $total;
            let (x0, g0, b0) = (ramp(total), gain(), shift());
            check_affine_grad(concat!(stringify!($module), "_f32"), &x0, &g0, &b0, total, |px, pg, pb| {
                let mut y = vec![0.0f32; total];
                nn::conv::$module::forward(
                    tensor::View::one(px),
                    tensor::View::one(pg),
                    tensor::View::one(pb),
                    &mut y,
                )
                .expect("前向");
                y.iter().sum()
            })
        }
    )*};
}

declared_affine_grad! {
    batch_norm_matches_central_differences => (batch_norm, nn::conv::batch_norm::C * nn::conv::batch_norm::S * nn::conv::batch_norm::N),
    group_norm_matches_central_differences => (group_norm, nn::conv::group_norm::C * nn::conv::group_norm::S * nn::conv::group_norm::N),
    instance_norm_matches_central_differences => (instance_norm, nn::conv::instance_norm::C * nn::conv::instance_norm::S * nn::conv::instance_norm::N),
    layer_norm_matches_central_differences => (layer_norm, nn::conv::layer_norm::R * nn::conv::layer_norm::W),
}

/// 运行时几何的一元反向对拍只声明几何、算子和输入长度。
macro_rules! shaped_sum_grad {
    ($( $name:ident => ($op:literal, $geometry:expr, $len:expr) ),* $(,)?) => {$(
        #[test]
        fn $name() -> Result<()> {
            sum_grad($op, $geometry, &ramp($len))
        }
    )*};
}

shaped_sum_grad! {
    shaped_max_pool1d_matches_central_differences => ("max_pool1d_f32", Geometry::Pool1d { k: 3, s: 1, p: 1, l: 7, c: 1 }, 7),
    shaped_max_pool2d_matches_central_differences => ("max_pool2d_f32", Geometry::Pool2d { k: 3, s: 2, p: 0, h: 5, w: 5, c: 2 }, 5 * 5 * 2),
    shaped_avg_pool2d_matches_central_differences => ("avg_pool2d_f32", Geometry::Avg2d { k: 3, s: 2, p: 1, h: 5, w: 5, c: 1, count_include_pad: false }, 5 * 5),
    shaped_adaptive_pool_matches_central_differences => ("adaptive_pool_f32", Geometry::Adaptive { oh: 3, ow: 3, h: 6, w: 6, c: 1 }, 36),
    shaped_interpolate_matches_central_differences => ("interpolate_f32", Geometry::Interp { oh: 5, ow: 5, h: 3, w: 3, c: 1, linear: true }, 9),
}

#[test]
fn shaped_batch_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Norm {
        n: 2,
        spatial: 3,
        channels: 5,
    };
    let (x0, g0, b0) = (ramp(30), gain5(), shift5());
    check_geometry_affine("batch_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_group_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Group {
        groups: 3,
        n: 2,
        spatial: 3,
        channels: 6,
    };
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
