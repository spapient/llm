use super::*;

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
