use super::*;
use diagnostic::Code;

/// 声明档样本的输入切片：卷积是 `(x, weight, bias)`，其余按一元/三元口径。
fn declared_inputs(sample: &nn::model::Sample, g: Geometry) -> Vec<&[f32]> {
    match g {
        Geometry::Conv2d { .. } => vec![
            sample.f32("x").expect("x"),
            sample.f32("weight").expect("weight"),
            sample.f32("bias").expect("bias"),
        ],
        _ if g.arity() == 1 => vec![sample.f32("x").expect("x")],
        _ => vec![
            sample.f32("x").expect("x"),
            sample.f32("gamma").expect("gamma"),
            sample.f32("beta").expect("beta"),
        ],
    }
}

/// 十种声明档算子及其样本，供形状对齐与前向对拍两个测试共用。
fn declared_samples() -> [(&'static str, nn::model::Sample); 10] {
    [
        ("max_pool1d_f32", nn::conv::max_pool1d::sample(1, 1)),
        ("max_pool2d_f32", nn::conv::max_pool2d::sample(1, 1)),
        ("avg_pool2d_f32", nn::conv::avg_pool2d::sample(1, 1)),
        ("adaptive_pool_f32", nn::conv::adaptive_pool::sample(1, 1)),
        ("interpolate_f32", nn::conv::interpolate::sample(1, 1)),
        ("batch_norm_f32", nn::conv::batch_norm::sample(1, 1)),
        ("conv2d_f32", nn::conv::conv2d::sample(1, 1)),
        ("group_norm_f32", nn::conv::group_norm::sample(1, 1)),
        ("instance_norm_f32", nn::conv::instance_norm::sample(1, 1)),
        ("layer_norm_f32", nn::conv::layer_norm::sample(1, 1)),
    ]
}

/// 声明档实例与 `nn::conv::classic` 的声明档样本长度逐一对齐：前向缓冲与样本同长。
#[test]
fn declared_geometries_match_the_sample_lengths() {
    for (op, sample) in declared_samples() {
        let g = Geometry::declared(op).expect("声明档实例");
        let y = sample.f32("y").expect("y");
        let inputs = declared_inputs(&sample, g);
        let lens: Vec<usize> = inputs.iter().map(|input| input.len()).collect();
        g.check_shape(&lens, y.len()).expect("声明档形状自洽");
    }
}

/// 前向：实例几何的结果必须与 `nn::conv::classic` 的 reference 逐位一致（同一份算术的两次执行）。
#[test]
fn declared_forward_matches_the_reference() {
    for (op, sample) in declared_samples() {
        let g = Geometry::declared(op).expect("声明档实例");
        let mut out = vec![0.0f32; g.outputs()];
        g.forward(&declared_inputs(&sample, g), &mut out)
            .expect("前向");
        let want = (nn::registry::model(op).expect("已注册").reference)(&sample).expect("参考");
        let want = want.as_f32().expect("参考档为 f32");
        assert_eq!(out.as_slice(), want, "{op} 实例前向与声明档参考不一致");
    }
}

/// 任意几何也要自洽：非声明档的窗口/分组必须能前向，且形状校验拦住不成立的几何。
/// 滚动统计按动量递推，同时不改变训练前向的输出（训练语义与数值参考同源）。
/// 推理用滚动统计归一化：`(x - m) / sqrt(v + eps) * gamma + beta`，与手算逐位一致。
#[test]
fn evaluation_normalizes_with_the_given_rolling_statistics() {
    let (n, spatial, channels) = (1usize, 2usize, 2usize);
    let len = n * spatial * channels;
    let geom = Geometry::Norm {
        n,
        spatial,
        channels,
    };
    let x: Vec<f32> = (0..len).map(|i| 0.4 + i as f32 * 0.3).collect();
    let gain = vec![1.5f32, 0.5];
    let shift = vec![0.2f32, -0.1];
    let running = Running {
        mean: vec![0.3, -0.2],
        var: vec![2.0, 0.5],
    };

    let mut out = vec![0.0f32; len];
    geom.forward_eval(&[&x, &gain, &shift], &mut out, &running)
        .expect("推理前向");
    for (idx, got) in out.iter().enumerate() {
        let ci = idx % channels;
        let want =
            (x[idx] - running.mean[ci]) / (running.var[ci] + EPS).sqrt() * gain[ci] + shift[ci];
        assert!(
            (got - want).abs() <= 1e-6,
            "第 {idx} 个输出 {got} vs {want}"
        );
    }
    // 通道数不匹配要在形状层报错，不静默套用错位统计。
    let wrong = Running::fresh(channels + 1);
    let error = geom
        .forward_eval(&[&x, &gain, &shift], &mut out, &wrong)
        .expect_err("通道数不符");
    assert_eq!(error.code(), Code::Invalid);
}

#[test]
fn running_statistics_update_by_momentum_without_changing_the_output() {
    let (n, spatial, channels) = (2usize, 3usize, 2usize);
    let len = n * spatial * channels;
    let geom = Geometry::Norm {
        n,
        spatial,
        channels,
    };
    let gain = vec![1.0f32; channels];
    let shift = vec![0.0f32; channels];
    let first: Vec<f32> = (0..len).map(|i| 0.3 + i as f32 * 0.1).collect();
    let second: Vec<f32> = (0..len).map(|i| 1.4 - i as f32 * 0.06).collect();

    let mut stats = Running::fresh(channels);
    for batch in [&first, &second] {
        let mut out = vec![0.0f32; len];
        geom.forward_with(&[batch, &gain, &shift], &mut out, Some(&mut stats))
            .expect("带统计的前向");
        let mut plain = vec![0.0f32; len];
        geom.forward(&[batch, &gain, &shift], &mut plain)
            .expect("无统计的前向");
        for (idx, (a, b)) in out.iter().zip(&plain).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "第 {idx} 个输出被统计影响：{a} vs {b}"
            );
        }
    }

    // 手动复算两批的滑动口径：`m = 0.9 * m + 0.1 * batch_mean`，方差用无偏值。
    let count = n * spatial;
    let mut mean = vec![0.0f32; channels];
    let mut var = vec![1.0f32; channels];
    for batch in [&first, &second] {
        for ci in 0..channels {
            let values: Vec<f32> = (0..count).map(|p| batch[p * channels + ci]).collect();
            let batch_mean = values.iter().sum::<f32>() / count as f32;
            let batch_var =
                values.iter().map(|v| (v - batch_mean).powi(2)).sum::<f32>() / count as f32;
            let unbiased = batch_var * count as f32 / (count - 1) as f32;
            mean[ci] = 0.9 * mean[ci] + 0.1 * batch_mean;
            var[ci] = 0.9 * var[ci] + 0.1 * unbiased;
        }
    }
    for ci in 0..channels {
        assert!(
            (stats.mean[ci] - mean[ci]).abs() <= 1e-5,
            "通道 {ci} 均值 {} vs {}",
            stats.mean[ci],
            mean[ci]
        );
        assert!(
            (stats.var[ci] - var[ci]).abs() <= 1e-5,
            "通道 {ci} 方差 {} vs {}",
            stats.var[ci],
            var[ci]
        );
    }
}

#[test]
fn arbitrary_geometries_forward_and_reject_bad_shapes() {
    let pool = Geometry::Pool2d {
        k: 3,
        s: 2,
        p: 0,
        h: 5,
        w: 5,
        c: 2,
    };
    assert_eq!(pool.outputs(), 2 * 2 * 2);
    let mut out = vec![0.0f32; pool.outputs()];
    let x: Vec<f32> = (0..5 * 5 * 2).map(|i| i as f32 - 12.0).collect();
    pool.forward(&[&x], &mut out).expect("前向");
    assert!(out.iter().all(|v| v.is_finite()));

    // 窗口比输入还大：显式拒绝，不出 0 长度。
    let bad = Geometry::Pool1d {
        k: 5,
        s: 1,
        p: 0,
        l: 3,
        c: 1,
    };
    assert!(bad.check_shape(&[3], bad.outputs()).is_err());
}
