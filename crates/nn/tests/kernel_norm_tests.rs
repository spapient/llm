use super::*;

/// 逐通道的均值与方差（有偏），测试用独立口径，避免复用被测代码。
fn stats(x: &[f32], c: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f64; c];
    let mut sq = vec![0.0f64; c];
    for p in 0..count {
        for ci in 0..c {
            let v = f64::from(x[p * c + ci]);
            mean[ci] += v;
            sq[ci] += v * v;
        }
    }
    let mean: Vec<f32> = mean.iter().map(|v| (v / count as f64) as f32).collect();
    let var: Vec<f32> = sq
        .iter()
        .enumerate()
        .map(|(ci, v)| ((v / count as f64) as f32 - mean[ci] * mean[ci]).max(0.0))
        .collect();
    (mean, var)
}

testkit::test_cases! {
fn batchnorm_train_normalizes_per_channel() {
    let bn = BatchNorm2d {
        channels: 3,
        eps: 1e-5,
        momentum: 0.1,
    };
    let (n, h, w, c) = (2, 2, 2, 3);
    let count = n * h * w;
    let x = testkit::sample(count * c, 1);
    let gamma = vec![1.0f32; c];
    let beta = vec![0.0f32; c];
    let mut rm = vec![0.0f32; c];
    let mut rv = vec![1.0f32; c];
    let mut out = vec![0.0f32; count * c];
    bn.forward_train(&x, &gamma, &beta, n, h, w, &mut rm, &mut rv, &mut out)
        .unwrap();

    let (mean, var) = stats(&x, c, count);
    for ci in 0..c {
        let want: Vec<f32> = (0..count)
            .map(|p| (x[p * c + ci] - mean[ci]) / (var[ci] + bn.eps).sqrt())
            .collect();
        let got: Vec<f32> = (0..count).map(|p| out[p * c + ci]).collect();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "通道 {ci}: {g} vs {w}");
        }
    }
}

fn batchnorm_train_updates_running_stats() {
    let bn = BatchNorm2d {
        channels: 1,
        eps: 1e-5,
        momentum: 0.5,
    };
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32];
    let beta = [0.0f32];
    let mut rm = [0.0f32];
    let mut rv = [1.0f32];
    let mut out = [0.0f32; 4];
    bn.forward_train(&x, &gamma, &beta, 1, 1, 4, &mut rm, &mut rv, &mut out)
        .unwrap();
    // batch mean = 2.5，动量 0.5 → 滑动均值 1.25。
    assert!((rm[0] - 1.25).abs() < 1e-6, "running_mean {}", rm[0]);
    // 有偏方差 1.25，无偏 5/3≈1.6667 → 滑动方差 (1+1.6667)/2。
    assert!((rv[0] - 1.333_333).abs() < 1e-4, "running_var {}", rv[0]);
}

fn batchnorm_eval_uses_running_stats_only() {
    let bn = BatchNorm2d {
        channels: 2,
        eps: 1e-5,
        momentum: 0.1,
    };
    let x = testkit::sample(4 * 2, 2);
    let gamma = [1.0f32, 2.0];
    let beta = [0.5f32, -0.5];
    let rm = [0.25f32, -0.75];
    let rv = [4.0f32, 16.0];
    let mut out = vec![0.0f32; 8];
    bn.forward_eval(&x, &gamma, &beta, 1, 1, 4, &rm, &rv, &mut out)
        .unwrap();
    for p in 0..4 {
        for ci in 0..2 {
            let want = (x[p * 2 + ci] - rm[ci]) / (rv[ci] + bn.eps).sqrt() * gamma[ci] + beta[ci];
            assert!((out[p * 2 + ci] - want).abs() < 1e-5);
        }
    }
}

fn groupnorm_yields_zero_mean_unit_variance() {
    let gn = GroupNorm {
        groups: 2,
        channels: 4,
        eps: 1e-5,
    };
    let (n, spatial, c) = (2, 5, 4);
    let x = testkit::sample(n * spatial * c, 3);
    let gamma = vec![1.0f32; c];
    let beta = vec![0.0f32; c];
    let mut out = vec![0.0f32; n * spatial * c];
    gn.forward(&x, &gamma, &beta, n, spatial, &mut out).unwrap();

    let per = c / gn.groups;
    for bi in 0..n {
        for g in 0..gn.groups {
            let mut sum = 0.0f32;
            let mut sq = 0.0f32;
            let len = (per * spatial) as f32;
            for s in 0..spatial {
                for ci in g * per..g * per + per {
                    let v = out[(bi * spatial + s) * c + ci];
                    sum += v;
                    sq += v * v;
                }
            }
            assert!(sum.abs() / len < 1e-5, "组均值未归零");
            assert!((sq / len - 1.0).abs() < 1e-4, "组方差未归一");
        }
    }
}

fn instancenorm_equals_groupnorm_with_c_groups() {
    let (n, spatial, c) = (1, 4, 3);
    let x = testkit::sample(n * spatial * c, 4);
    let gamma = vec![1.0f32; c];
    let beta = vec![0.0f32; c];
    let mut a = vec![0.0f32; n * spatial * c];
    let mut b = vec![0.0f32; n * spatial * c];
    InstanceNorm {
        channels: c,
        eps: 1e-5,
    }
    .forward(&x, &gamma, &beta, n, spatial, &mut a)
    .unwrap();
    GroupNorm {
        groups: c,
        channels: c,
        eps: 1e-5,
    }
    .forward(&x, &gamma, &beta, n, spatial, &mut b)
    .unwrap();
    assert_eq!(a, b);
}

fn layernorm_normalizes_each_row() {
    let ln = LayerNorm {
        width: 5,
        eps: 1e-5,
    };
    let x = testkit::sample(15, 5);
    let gamma = vec![1.0f32; 5];
    let beta = vec![0.0f32; 5];
    let mut out = vec![0.0f32; 15];
    ln.forward(&x, &gamma, &beta, &mut out).unwrap();
    for row in 0..3 {
        let src = &out[row * 5..(row + 1) * 5];
        let mean: f64 = src.iter().map(|v| f64::from(*v)).sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-6, "行 {row} 均值 {mean}");
        // 输入的方差为 var 时，归一化后方差是 var/(var+eps)——eps 会把它压到 1 以下。
        let xin = &x[row * 5..(row + 1) * 5];
        let im: f64 = xin.iter().map(|v| f64::from(*v)).sum::<f64>() / 5.0;
        let var: f64 = xin
            .iter()
            .map(|v| (f64::from(*v) - im).powi(2))
            .sum::<f64>()
            / 5.0;
        let want = (var / (var + f64::from(ln.eps))) as f32;
        let got: f32 = src.iter().map(|v| (v - mean as f32).powi(2)).sum::<f32>() / 5.0;
        assert!(
            (got - want).abs() < 1e-5,
            "行 {row} 方差 {got}，期望 {want}"
        );
    }
}

fn rejects_bad_geometry() {
    let gn = GroupNorm {
        groups: 3,
        channels: 4,
        eps: 1e-5,
    };
    let x = vec![0.0f32; 8];
    let mut out = vec![0.0f32; 8];
    assert!(
        gn.forward(&x, &[1.0; 4], &[0.0; 4], 1, 2, &mut out)
            .is_err()
    );
    let ln = LayerNorm {
        width: 0,
        eps: 1e-5,
    };
    assert!(ln.forward(&x, &[], &[], &mut out).is_err());
}
}
