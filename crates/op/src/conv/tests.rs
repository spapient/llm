//! 职责：卷积/池化/插值算子测试（从 conv/mod.rs 拆出，见父模块职责）。

use super::*;

fn sample(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((x >> 9) as f32) / 512.0 - 4.0
        })
        .collect()
}

fn sample_i8(len: usize, salt: u32) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((x % 31) as i32 - 15) as i8
        })
        .collect()
}

/// 朴素二维卷积：四重循环直算，作为唯一参考口径。
fn naive2d(
    x: &[f32],
    wgt: &[f32],
    bias: &[f32],
    conv: &Conv2d,
    n: usize,
    h: usize,
    ww: usize,
    out: &mut [f32],
) {
    let (oh, ow) = conv.shape(h, ww).unwrap();
    let cin = conv.cin;
    let cout = conv.cout;
    let g = conv.geom;
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                for co in 0..cout {
                    let mut acc = bias[co];
                    for ki in 0..g.kh {
                        let ii = (oi * g.sh + ki) as isize - g.ph as isize;
                        if ii < 0 || ii as usize >= h {
                            continue;
                        }
                        for kj in 0..g.kw {
                            let jj = (oj * g.sw + kj) as isize - g.pw as isize;
                            if jj < 0 || jj as usize >= ww {
                                continue;
                            }
                            for ci in 0..cin {
                                let xv = x[((ni * h + ii as usize) * ww + jj as usize) * cin + ci];
                                let wv = wgt[((ki * g.kw + kj) * cin + ci) * cout + co];
                                acc += xv * wv;
                            }
                        }
                    }
                    out[((ni * oh + oi) * ow + oj) * cout + co] = acc;
                }
            }
        }
    }
}

fn close(a: &[f32], b: &[f32]) {
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let tol = 1e-4 + 1e-3 * y.abs();
        assert!((x - y).abs() <= tol, "第 {i} 个元素 {x} vs {y}");
    }
}

#[test]
fn conv2d_matches_naive() {
    let cases = [
        (
            (1, 8, 8, 3, 4),
            ConvGeom {
                kh: 3,
                kw: 3,
                sh: 1,
                sw: 1,
                ph: 1,
                pw: 1,
            },
        ),
        (
            (2, 7, 9, 2, 5),
            ConvGeom {
                kh: 3,
                kw: 3,
                sh: 2,
                sw: 2,
                ph: 0,
                pw: 0,
            },
        ),
        (
            (1, 5, 5, 1, 1),
            ConvGeom {
                kh: 5,
                kw: 5,
                sh: 1,
                sw: 1,
                ph: 0,
                pw: 0,
            },
        ),
        (
            (1, 6, 6, 4, 3),
            ConvGeom {
                kh: 2,
                kw: 4,
                sh: 3,
                sw: 1,
                ph: 2,
                pw: 0,
            },
        ),
    ];
    for ((n, h, w, cin, cout), geom) in cases {
        let conv = Conv2d::new(cin, cout, geom);
        let (oh, ow) = conv.shape(h, w).unwrap();
        let x = sample(n * h * w * cin, 1);
        let wgt = sample(geom.patch(cin) * cout, 2);
        let bias = sample(cout, 3);
        let mut got = vec![0.0f32; n * oh * ow * cout];
        let mut want = vec![0.0f32; n * oh * ow * cout];
        conv.forward_f32(&x, &wgt, &bias, n, (h, w), &mut got)
            .unwrap();
        naive2d(&x, &wgt, &bias, &conv, n, h, w, &mut want);
        close(&got, &want);
    }
}

#[test]
fn conv1d_matches_naive() {
    let (n, l, cin, cout, kw, stride, pad) = (2, 9, 3, 4, 3, 2, 1);
    let conv = Conv1d::new(cin, cout, kw, stride, pad);
    let ol = conv.len(l).unwrap();
    let x = sample(n * l * cin, 1);
    let wgt = sample(kw * cin * cout, 2);
    let bias = sample(cout, 3);
    let mut got = vec![0.0f32; n * ol * cout];
    conv.forward_f32(&x, &wgt, &bias, n, l, &mut got).unwrap();

    // 参考：1d 就是 `kh=1` 的 2d。
    let conv2 = Conv2d::new(cin, cout, conv.geom);
    let mut want = vec![0.0f32; n * ol * cout];
    naive2d(&x, &wgt, &bias, &conv2, n, 1, l, &mut want);
    close(&got, &want);
}

#[test]
fn conv2d_i8_matches_exact() {
    let (n, h, w, cin, cout) = (1, 6, 6, 3, 5);
    let geom = ConvGeom {
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
    };
    let conv = Conv2d::new(cin, cout, geom);
    let (oh, ow) = conv.shape(h, w).unwrap();
    let x = sample_i8(n * h * w * cin, 1);
    let wgt = sample_i8(geom.patch(cin) * cout, 2);
    let bias = vec![0i32; cout];
    let mut got = vec![0i32; n * oh * ow * cout];
    conv.forward_i8(&x, &wgt, &bias, n, (h, w), &mut got).unwrap();

    let mut want = vec![0i32; n * oh * ow * cout];
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                for co in 0..cout {
                    let mut acc = 0i32;
                    for ki in 0..geom.kh {
                        let ii = (oi * geom.sh + ki) as isize - geom.ph as isize;
                        if ii < 0 || ii as usize >= h {
                            continue;
                        }
                        for kj in 0..geom.kw {
                            let jj = (oj * geom.sw + kj) as isize - geom.pw as isize;
                            if jj < 0 || jj as usize >= w {
                                continue;
                            }
                            for ci in 0..cin {
                                let xv = x[((ni * h + ii as usize) * w + jj as usize) * cin + ci];
                                let wv = wgt[((ki * geom.kw + kj) * cin + ci) * cout + co];
                                acc += i32::from(xv) * i32::from(wv);
                            }
                        }
                    }
                    want[((ni * oh + oi) * ow + oj) * cout + co] = acc;
                }
            }
        }
    }
    assert_eq!(got, want, "量化卷积与整数直算逐位相同");
}

#[test]
fn max_pool_matches_naive() {
    let pool = MaxPool2d {
        geom: ConvGeom {
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
            ph: 0,
            pw: 0,
        },
    };
    let (n, h, w, c) = (1, 4, 4, 2);
    let (oh, ow) = pool.shape(h, w).unwrap();
    let x = sample(n * h * w * c, 1);
    let mut got = vec![0.0f32; n * oh * ow * c];
    pool.forward(&x, n, h, w, c, &mut got).unwrap();
    let mut want = vec![0.0f32; n * oh * ow * c];
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                for ci in 0..c {
                    let mut best = f32::NEG_INFINITY;
                    for ki in 0..2 {
                        for kj in 0..2 {
                            let v = x[((ni * h + oi * 2 + ki) * w + oj * 2 + kj) * c + ci];
                            best = best.max(v);
                        }
                    }
                    want[((ni * oh + oi) * ow + oj) * c + ci] = best;
                }
            }
        }
    }
    close(&got, &want);
}

#[test]
fn avg_pool_counts_only_real_cells() {
    // 3×3 输入、2×2..1 补零、步长 2 → 输出 2×2；左上窗口只有 (0,0) 一个真实单元。
    let pool = AvgPool2d {
        geom: ConvGeom {
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
            ph: 1,
            pw: 1,
        },
        count_include_pad: false,
    };
    let (n, h, w, c) = (1, 3, 3, 1);
    let (oh, ow) = pool.shape(h, w).unwrap();
    let x = sample(n * h * w * c, 1);
    let mut got = vec![0.0f32; n * oh * ow * c];
    pool.forward(&x, n, h, w, c, &mut got).unwrap();
    assert_eq!((oh, ow), (2, 2));
    assert!((got[0] - x[0]).abs() < 1e-6, "剔除补零后只剩左上单元");
}

/// 卷积端到端吞吐（含 im2col + 分配）：`cargo test --release -p llm-op --lib -- --ignored --nocapture bench_conv`
#[test]
#[ignore = "卷积吞吐基线，手动跑"]
fn bench_conv() {
    use std::time::Instant;
    let shapes = [
        (1usize, 128usize, 128usize, 256usize, 256usize, 3usize),
        (1, 64, 64, 256, 256, 3),
        (1, 32, 32, 512, 512, 3),
    ];
    for (n, h, w, cin, cout, k) in shapes {
        let geom = ConvGeom {
            kh: k,
            kw: k,
            sh: 1,
            sw: 1,
            ph: k / 2,
            pw: k / 2,
        };
        let conv = Conv2d::new(cin, cout, geom);
        let (oh, ow) = conv.shape(h, w).unwrap();
        let rows = n * oh * ow;
        let patch = geom.patch(cin);
        let flops = 2.0 * rows as f64 * cout as f64 * patch as f64;

        let x8 = sample_i8(n * h * w * cin, 1);
        let w8 = sample_i8(patch * cout, 2);
        let b8 = vec![0i32; cout];
        let mut o8 = vec![0i32; rows * cout];
        conv.forward_i8(&x8, &w8, &b8, n, (h, w), &mut o8).unwrap();
        let mut best = f64::INFINITY;
        let mut bestcol = f64::INFINITY;
        let mut buf = vec![0i8; rows * patch];
        for _ in 0..5 {
            let t = Instant::now();
            conv.forward_i8(&x8, &w8, &b8, n, (h, w), &mut o8).unwrap();
            best = best.min(t.elapsed().as_nanos() as f64);
            let t = Instant::now();
            conv.forward_i8_with(&x8, &w8, &b8, n, (h, w), &mut o8, &mut buf)
                .unwrap();
            bestcol = bestcol.min(t.elapsed().as_nanos() as f64);
        }
        // 拆两段看谁在吃时间。
        let mut tcol = f64::INFINITY;
        let mut tgemm = f64::INFINITY;
        for col in 0..5 {
            let _ = col;
            let t = Instant::now();
            im2col2d(&x8, &mut buf, n, (h, w), &conv, (oh, ow));
            tcol = tcol.min(t.elapsed().as_nanos() as f64);
            let t = Instant::now();
            fast::gemm_i8(&buf, &w8, rows, cout, patch, &mut o8);
            tgemm = tgemm.min(t.elapsed().as_nanos() as f64);
        }
        println!(
            "  conv2d_i8 {n}x{h}x{w} cin{cin} cout{cout} k{k}: 新分配 {:.1} / 复用 {:.1} GF/s \
             (im2col {:.2} ms, gemm {:.2} ms)",
            flops / best,
            flops / bestcol,
            tcol / 1e6,
            tgemm / 1e6
        );

        let x32 = sample(n * h * w * cin, 1);
        let w32 = sample(patch * cout, 2);
        let b32 = sample(cout, 3);
        let mut o32 = vec![0.0f32; rows * cout];
        conv.forward_f32(&x32, &w32, &b32, n, (h, w), &mut o32)
            .unwrap();
        let mut best32 = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            conv.forward_f32(&x32, &w32, &b32, n, (h, w), &mut o32)
                .unwrap();
            best32 = best32.min(t.elapsed().as_nanos() as f64);
        }
        println!(
            "  conv2d_f32 {n}x{h}x{w} cin{cin} cout{cout} k{k}: {:.1} GF/s",
            flops / best32
        );
    }
}

#[test]
fn maxpool1d_matches_naive() {
    // NHWC，l=6、c=1，窗口 3 步长 2 补零 1 → 输出长 4。
    let x = [1.0f32, 5.0, 2.0, 8.0, 3.0, 0.5];
    let pool = MaxPool1d { k: 3, s: 2, p: 1 };
    assert_eq!(pool.len(6).unwrap(), 3);
    let mut out = [0.0f32; 3];
    pool.forward(&x, 1, 6, 1, &mut out).unwrap();
    // 窗口（输入坐标）：[-1,0,1]→5；[1,2,3]→8；[3,4,5]→8。
    assert_eq!(out, [5.0, 8.0, 8.0]);
    // 多通道：各通道独立，不串味。
    let x = [1.0f32, 9.0, 3.0, 2.0, 0.0, 4.0];
    let pool = MaxPool1d { k: 2, s: 1, p: 0 };
    let mut out = [0.0f32; 4];
    pool.forward(&x, 1, 3, 2, &mut out).unwrap();
    assert_eq!(out, [3.0, 9.0, 3.0, 4.0]);
    assert!(MaxPool1d { k: 0, s: 1, p: 0 }.len(4).is_err());
}

#[test]
fn adaptive_avg_pool_matches_block_means() {
    // 4×4 单通道压到 2×2：每格正好 2×2 块。
    let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let pool = AdaptiveAvgPool2d { oh: 2, ow: 2 };
    let mut out = [0.0f32; 4];
    pool.forward(&x, 1, 4, 4, 1, &mut out).unwrap();
    assert_eq!(out, [2.5, 4.5, 10.5, 12.5]);
    // 5×5 压到 2×2：窗口按 floor/ceil 划分，末格覆盖 [2,5)×[2,5)。
    let x25: Vec<f32> = (0..25).map(|v| v as f32).collect();
    let pool = AdaptiveAvgPool2d { oh: 2, ow: 2 };
    let mut out = [0.0f32; 4];
    pool.forward(&x25, 1, 5, 5, 1, &mut out).unwrap();
    let mut want = [0.0f32; 4];
    for oi in 0usize..2 {
        let h0 = oi * 5 / 2;
        let h1 = ((oi + 1) * 5).div_ceil(2);
        for oj in 0usize..2 {
            let w0 = oj * 5 / 2;
            let w1 = ((oj + 1) * 5).div_ceil(2);
            let area = (h1 - h0) * (w1 - w0);
            let mut sum = 0.0f32;
            for ii in h0..h1 {
                for jj in w0..w1 {
                    sum += (ii * 5 + jj) as f32;
                }
            }
            want[oi * 2 + oj] = sum / area as f32;
        }
    }
    for (got, exp) in out.iter().zip(want.iter()) {
        assert!((got - exp).abs() < 1e-5, "{got} vs {exp}");
    }
    assert!(
        AdaptiveAvgPool2d { oh: 0, ow: 1 }
            .forward(&x, 1, 4, 4, 1, &mut out)
            .is_err()
    );
}

#[test]
fn interpolate_nearest_and_bilinear() {
    // 2×2 单通道升到 4×4：最近邻严格复制左上角。
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let up = Interpolate {
        sampler: Sampler::Nearest,
        oh: 4,
        ow: 4,
    };
    let mut out = [0.0f32; 16];
    up.forward(&x, 1, 2, 2, 1, &mut out).unwrap();
    for oi in 0..4 {
        for oj in 0..4 {
            assert_eq!(out[oi * 4 + oj], x[(oi * 2 / 4) * 2 + (oj * 2 / 4)]);
        }
    }
    // 双线性在 4×4 上：中心附近的点应是四邻的加权，边界被夹住。
    let up = Interpolate {
        sampler: Sampler::Linear,
        oh: 4,
        ow: 4,
    };
    up.forward(&x, 1, 2, 2, 1, &mut out).unwrap();
    // 左上角 (0,0)：源坐标 (−0.25,−0.25) → 夹到 (0,0) → 值 1.0。
    assert!((out[0] - 1.0).abs() < 1e-6);
    // 右下角 (3,3)：源坐标 (1.25,1.25) → 夹到 (1,1) → 值 4.0。
    assert!((out[15] - 4.0).abs() < 1e-6);
    // 内部 (1,1)：源 (0.25,0.25) → 0.25/0.25 加权四邻 = 1.75。
    assert!((out[5] - 1.75).abs() < 1e-6, "{}", out[5]);
    // 恒等（同尺寸）时两种取样都必须逐元素一致。
    let id = Interpolate {
        sampler: Sampler::Linear,
        oh: 2,
        ow: 2,
    };
    let mut same = [0.0f32; 4];
    id.forward(&x, 1, 2, 2, 1, &mut same).unwrap();
    assert_eq!(same, x);
    assert!(
        Interpolate {
            sampler: Sampler::Nearest,
            oh: 0,
            ow: 2
        }
        .forward(&x, 1, 2, 2, 1, &mut out)
        .is_err()
    );
}
