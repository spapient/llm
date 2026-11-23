//! 职责：卷积/池化/插值算子测试（从 conv/mod.rs 拆出，见父模块职责）。

use crate::kernel::fast;

use super::support::*;
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
#[allow(clippy::too_many_arguments)]
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
        testkit::assert_close(&got, &want, 1e-4, 1e-3);
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
    testkit::assert_close(&got, &want, 1e-4, 1e-3);
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
    conv.forward_i8(&x, &wgt, &bias, n, (h, w), &mut got)
        .unwrap();

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

/// 池化用例的公共跑法：按输出形状造确定性样本与同形输出，前向由调用点注入。
///
/// 三个用例（最大池 / 平均池 / 平均池补零）此前各自重复"推形状、造样本、备缓冲、跑前向"
/// 四步；收进一处后，用例只剩各自的断言口径。
fn pooled(
    (oh, ow): (usize, usize),
    (n, h, w, c): (usize, usize, usize, usize),
    forward: impl FnOnce(&[f32], &mut [f32]),
) -> (Vec<f32>, Vec<f32>) {
    let x = sample(n * h * w * c, 1);
    let mut got = vec![0.0f32; n * oh * ow * c];
    forward(&x, &mut got);
    (x, got)
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
    let (x, got) = pooled((oh, ow), (n, h, w, c), |x, out| {
        pool.forward(x, n, h, w, c, out).unwrap();
    });
    let mut want = vec![0.0f32; got.len()];
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
    testkit::assert_close(&got, &want, 1e-4, 1e-3);
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
    assert_eq!((oh, ow), (2, 2), "3×3 / 2×2 / 补零 1 / 步长 2 的输出形状");
    let (x, got) = pooled((oh, ow), (n, h, w, c), |x, out| {
        pool.forward(x, n, h, w, c, out).unwrap();
    });
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

// conv 数值内核的**定律**测试：每条性质都从算子的数学定义推出，不复述实现。
//
// 挑选原则与 `gemm/tests/prop_numeric.rs` 一致——按"能有多强就用多强"分档：
//
// - **精确相等**：`w = 0` 时输出就是 `bias`、`conv` 的奇对称、最大池化的"取值必被取到"。
// 这些地方用容差只会掩盖错误。
// - **相对误差**：仿射性、线性、两个独立实现的差分（`flash` 对 `naive`）。
// - **结构性**：注意力的输出必须落在 `v` 的凸包内、平均/插值输出必须落在窗口界内。
// 这类性质不依赖任何实现细节，只依赖算子的定义，且能抓住"权重没归一化"这类错。

use crate::conv::attention;
use crate::conv::classic::{
    adaptive_pool, avg_pool2d, conv2d, interpolate, max_pool1d, max_pool2d,
};
use proptest::prelude::*;
use testkit::prop;

/// 定长浮点向量：几何是声明档常量，不能由生成器决定长度。
fn fixed(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(prop::finite(), len)
}

/// `k = 3, s = 1, p = 1` 的 same 输出共用窗口：返回该输出位置**有效**（非填充）元素的下标。
fn same_window(idx: usize, h: usize, w: usize) -> Vec<usize> {
    let (oh, ow) = ((idx / w) as i32, (idx % w) as i32);
    let mut idx_set = Vec::with_capacity(9);
    for dh in -1..=1 {
        for dw in -1..=1 {
            let (ih, iw) = (oh + dh, ow + dw);
            if (0..h as i32).contains(&ih) && (0..w as i32).contains(&iw) {
                idx_set.push(ih as usize * w + iw as usize);
            }
        }
    }
    idx_set
}

/// 自适应池化 `4 → 2` 的分块：输出格 `(oh, ow)` 覆盖 `2×2` 输入块。
fn adaptive_window(idx: usize, oh: usize, ow: usize, h: usize, w: usize) -> Vec<usize> {
    let (bh, bw) = (h / oh, w / ow);
    let (row, col) = (idx / ow, idx % ow);
    let mut idx_set = Vec::with_capacity(bh * bw);
    for dh in 0..bh {
        for dw in 0..bw {
            idx_set.push((row * bh + dh) * w + col * bw + dw);
        }
    }
    idx_set
}

/// 把下标集合映射成值的上下界（`min`/`max` 对有限输入是全域定义的）。
fn bounds(values: &[f32], idx_set: &[usize]) -> (f32, f32) {
    idx_set
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &i| {
            (low.min(values[i]), high.max(values[i]))
        })
}

fn conv2d_run(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; conv2d::OH * conv2d::OW * conv2d::COUT];
    conv2d::forward(x, weight, bias, &mut out).expect("定长缓冲");
    out
}

/// 一维窗口族与插值的统一调用口径：输入定长、输出定长、无参数。
fn window_run(
    forward: fn(&[f32], &mut [f32]) -> diagnostic::Result<()>,
    x: &[f32],
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    forward(x, &mut out).expect("定长缓冲");
    out
}

fn attention_run(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; s * d];
    attention::forward(q, k, v, &mut out, s, d).expect("等长");
    out
}

fn flash_run(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; s * d];
    attention::flash_forward(q, k, v, &mut out, s, d).expect("等长");
    out
}

proptest! {
    #![proptest_config(prop::config())]

    // ---------------------------------------------------------------- 注意力

    /// 注意力的定义：输出第 `i` 行是 `v` 各行按行和为 1 的非负权重的凸组合。
    /// 于是每个分量必须落在 `v` 该列的最小值与最大值之间——这条律不依赖 softmax、
    /// 不依赖 `1/√d`、不依赖是否分块，只依赖"凸组合"这一个定义。
    #[test]
    fn attention_output_lies_in_the_convex_hull_of_values(
        q in fixed(4 * 3), k in fixed(4 * 3), v in fixed(4 * 3)
    ) {
        let (s, d) = (4, 3);
        let out = attention_run(&q, &k, &v, s, d);
        prop_assert!(prop::all_finite(&out), "输出出现非有限值");
        for row in 0..s {
            for col in 0..d {
                let column: Vec<f32> = (0..s).map(|j| v[j * d + col]).collect();
                let (low, high) = bounds(&column, &(0..s).collect::<Vec<_>>());
                let got = out[row * d + col];
                prop_assert!(
                    got >= low - 1e-6 && got <= high + 1e-6,
                    "第 {row} 行第 {col} 列越出凸包：[{low}, {high}] 之外得 {got}"
                );
            }
        }
    }

    /// 对 `v` 的线性：注意力权重只由 `q`、`k` 决定，所以 `v` 方向是线性的。
    #[test]
    fn attention_is_linear_in_values(
        q in fixed(3 * 2), k in fixed(3 * 2), v1 in fixed(3 * 2), v2 in fixed(3 * 2),
        alpha in prop::finite(), beta in prop::finite()
    ) {
        let (s, d) = (3, 2);
        let mixed: Vec<f32> = v1.iter().zip(&v2).map(|(a, b)| alpha * a + beta * b).collect();
        let got = attention_run(&q, &k, &mixed, s, d);
        let left = attention_run(&q, &k, &v1, s, d);
        let right = attention_run(&q, &k, &v2, s, d);
        let want: Vec<f32> = left.iter().zip(&right).map(|(a, b)| alpha * a + beta * b).collect();
        prop::assert_within_rel(&got, &want, 1e-5, "注意力对 v 的线性");
    }

    /// softmax 的参数不变性：给**所有** `k` 行加同一个向量，等价于给每行 logit 加同一个常数
    /// `q_i · c`，行内相对大小不变，因此注意力输出不变。这条律盯的是"是否真的按行做 softmax"。
    #[test]
    fn attention_ignores_a_common_shift_of_keys(
        q in fixed(3 * 2), k in fixed(3 * 2), v in fixed(3 * 2), shift in fixed(2)
    ) {
        let (s, d) = (3, 2);
        let shifted: Vec<f32> = k
            .iter()
            .enumerate()
            .map(|(i, value)| value + shift[i % d])
            .collect();
        prop::assert_within_rel(
            &attention_run(&q, &shifted, &v, s, d),
            &attention_run(&q, &k, &v, s, d),
            1e-5,
            "所有 k 行同加一个向量不改变注意力",
        );
    }

    /// `v` 各行相同时，凸组合只能是那个值本身：权重是否归一化在这里无处可藏。
    #[test]
    fn attention_on_constant_values_reproduces_them(
        q in fixed(3 * 2), k in fixed(3 * 2), value in fixed(2)
    ) {
        let (s, d) = (3, 2);
        let v: Vec<f32> = (0..s).flat_map(|_| value.clone()).collect();
        let out = attention_run(&q, &k, &v, s, d);
        let want: Vec<f32> = (0..s).flat_map(|_| value.clone()).collect();
        prop::assert_within_rel(&out, &want, 1e-5, "常量 v 必须原样穿过注意力");
    }

    /// 差分律：`flash`（online softmax）与 `naive`（先求全行再归一化）是两条独立路径，
    /// 数学上同值。分块与不分块、单遍与两遍的差别只能落在 `f32` 舍入上。
    #[test]
    fn flash_and_naive_attention_agree(
        q in fixed(5 * 3), k in fixed(5 * 3), v in fixed(5 * 3)
    ) {
        let (s, d) = (5, 3);
        prop::assert_within_rel(
            &flash_run(&q, &k, &v, s, d),
            &attention_run(&q, &k, &v, s, d),
            1e-5,
            "flash 与 naive 注意力",
        );
    }

    // ---------------------------------------------------------------- 二维卷积

    /// `w = 0` 时卷积核没有任何贡献，输出必须**恰好**是 `bias` 的广播。
    /// 用精确相等：零累加后再加 `bias` 不引入任何舍入。
    #[test]
    fn conv_with_zero_weights_is_the_bias(x in fixed(4 * 4 * 2), bias in fixed(3)) {
        let weight = vec![0.0f32; 3 * 3 * 2 * 3];
        let out = conv2d_run(&x, &weight, &bias);
        let want: Vec<f32> = (0..conv2d::OH * conv2d::OW)
            .flat_map(|_| bias.clone())
            .collect();
        prop::assert_exact("w = 0 的卷积", &out, &want);
    }

    /// 输入全零时同理：输出必须**恰好**是 `bias`。
    #[test]
    fn conv_of_zero_input_is_the_bias(weight in fixed(3 * 3 * 2 * 3), bias in fixed(3)) {
        let x = vec![0.0f32; 4 * 4 * 2];
        let out = conv2d_run(&x, &weight, &bias);
        let want: Vec<f32> = (0..conv2d::OH * conv2d::OW)
            .flat_map(|_| bias.clone())
            .collect();
        prop::assert_exact("x = 0 的卷积", &out, &want);
    }

    /// 对输入的仿射性：`conv(x₁ + x₂) = conv(x₁) + conv(x₂) − bias`。
    /// 卷积是线性算子，"减掉多加的那一份 bias"是它区别于纯线性的全部内容。
    #[test]
    fn conv_is_affine_in_its_input(
        x1 in fixed(4 * 4 * 2), x2 in fixed(4 * 4 * 2),
        weight in fixed(3 * 3 * 2 * 3), bias in fixed(3)
    ) {
        let sum: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| a + b).collect();
        let got = conv2d_run(&sum, &weight, &bias);
        let want: Vec<f32> = conv2d_run(&x1, &weight, &bias)
            .iter()
            .zip(conv2d_run(&x2, &weight, &bias))
            .zip((0..conv2d::OH * conv2d::OW).flat_map(|_| bias.clone()))
            .map(|((a, b), c)| a + b - c)
            .collect();
        prop::assert_within_rel(&got, &want, 1e-5, "卷积对输入的仿射性");
    }

    /// 奇对称**逐位**成立：`conv(−x, w, −b) = −conv(x, w, b)`。
    /// 每个乘积取反是精确的、累加顺序不变，所以左右两侧的舍入完全一致；这里用精确断言。
    #[test]
    fn conv_is_odd_when_input_and_bias_both_flip(
        x in fixed(4 * 4 * 2), weight in fixed(3 * 3 * 2 * 3), bias in fixed(3)
    ) {
        let negated: Vec<f32> = x.iter().map(|v| -v).collect();
        let flipped_bias: Vec<f32> = bias.iter().map(|v| -v).collect();
        let want: Vec<f32> = conv2d_run(&x, &weight, &bias).iter().map(|v| -v).collect();
        prop::assert_exact("conv(−x, w, −b) = −conv(x, w, b)", &conv2d_run(&negated, &weight, &flipped_bias), &want);
    }

    // ---------------------------------------------------------------- 池化

    /// 最大池化 `3×3, s=1, p=1` 含中心点，所以输出每个位置必**不小于**输入同位置。
    /// 这条律直接盯住"填充值取 0"这类错误：全负输入下它会立刻破防。
    #[test]
    fn max_pool2d_never_goes_below_its_input(x in fixed(4 * 4)) {
        let out = window_run(max_pool2d::forward, &x, max_pool2d::H * max_pool2d::W);
        for (i, (got, source)) in out.iter().zip(&x).enumerate() {
            prop_assert!(got >= source, "第 {i} 个位置：池化结果 {got} 小于输入 {source}");
        }
    }

    /// 最大值必被**取到**：每个输出都恰好等于其有效窗口里的某个输入元素（逐位相等）。
    /// 这比"不小于窗口内所有元素"更强——它同时排除了"多加了什么"。
    #[test]
    fn max_pool2d_output_is_attained_in_its_window(x in fixed(4 * 4)) {
        let (h, w) = (max_pool2d::H, max_pool2d::W);
        let out = window_run(max_pool2d::forward, &x, h * w);
        for (i, got) in out.iter().enumerate() {
            let window = same_window(i, h, w);
            prop_assert!(
                window.iter().any(|&j| x[j].to_bits() == got.to_bits()),
                "第 {i} 个输出 {got} 不是其窗口中任何元素的原值"
            );
        }
    }

    /// 常量场上最大池化原样返回（逐位相等）：窗口内所有元素相同，最大值就是它。
    #[test]
    fn max_pool2d_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; max_pool2d::H * max_pool2d::W];
        let out = window_run(max_pool2d::forward, &x, x.len());
        prop::assert_exact("常量场的最大池化", &out, &x);
    }

    /// 一维最大池化 `k=2, s=2` 同样"取值必被取到"：每个输出必等于其两个输入之一。
    #[test]
    fn max_pool1d_output_is_one_of_its_pair(x in fixed(16)) {
        let out = window_run(max_pool1d::forward, &x, max_pool1d::OL);
        for (i, got) in out.iter().enumerate() {
            prop_assert!(
                x[2 * i].to_bits() == got.to_bits() || x[2 * i + 1].to_bits() == got.to_bits(),
                "第 {i} 个输出 {got} 既不是 x[{}] 也不是 x[{}]",
                2 * i,
                2 * i + 1
            );
        }
    }

    /// 平均池化 `3×3, s=1, p=1`、`count_include_pad = false`：常量场必须原样穿过。
    /// 若填充被计入分母，角上的输出会变成 `4c/9`——这条律对那种错误是致命的。
    #[test]
    fn avg_pool2d_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; avg_pool2d::H * avg_pool2d::W];
        let out = window_run(avg_pool2d::forward, &x, x.len());
        let want = vec![value; x.len()];
        prop::assert_within_rel(&out, &want, 1e-6, "常量场的平均池化");
    }

    /// 平均池化输出必须落在其有效窗口的上下界之内（平均值不会越界）。
    #[test]
    fn avg_pool2d_stays_within_its_window_bounds(x in fixed(4 * 4)) {
        let (h, w) = (avg_pool2d::H, avg_pool2d::W);
        let out = window_run(avg_pool2d::forward, &x, h * w);
        for (i, got) in out.iter().enumerate() {
            let (low, high) = bounds(&x, &same_window(i, h, w));
            prop_assert!(
                *got >= low - 1e-6 && *got <= high + 1e-6,
                "第 {i} 个输出 {got} 越出窗口界 [{low}, {high}]"
            );
        }
    }

    /// 自适应池化 `4 → 2`：每个输出是其 `2×2` 分块的均值，常量场必须原样穿过。
    #[test]
    fn adaptive_pool_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; adaptive_pool::H * adaptive_pool::W];
        let out = window_run(adaptive_pool::forward, &x, adaptive_pool::OH * adaptive_pool::OW);
        let want = vec![value; out.len()];
        prop::assert_within_rel(&out, &want, 1e-6, "常量场的自适应池化");
    }

    /// 自适应池化输出必须落在其分块的上下界之内。
    #[test]
    fn adaptive_pool_stays_within_its_block_bounds(x in fixed(4 * 4)) {
        let (h, w) = (adaptive_pool::H, adaptive_pool::W);
        let (oh, ow) = (adaptive_pool::OH, adaptive_pool::OW);
        let out = window_run(adaptive_pool::forward, &x, oh * ow);
        for (i, got) in out.iter().enumerate() {
            let (low, high) = bounds(&x, &adaptive_window(i, oh, ow, h, w));
            prop_assert!(
                *got >= low - 1e-6 && *got <= high + 1e-6,
                "第 {i} 个输出 {got} 越出分块界 [{low}, {high}]"
            );
        }
    }

    // ---------------------------------------------------------------- 双线性插值

    /// 双线性插值的权重是单位分解（该点两侧权重和为 1），所以常量场必须原样穿过。
    /// 这条律盯的是"权重和是否为 1"——`align_corners`、边界钳制都会在这里露出来。
    #[test]
    fn interpolation_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; interpolate::H * interpolate::W];
        let out = window_run(interpolate::forward, &x, interpolate::OH * interpolate::OW);
        let want = vec![value; out.len()];
        prop::assert_within_rel(&out, &want, 1e-6, "常量场的双线性插值");
    }

    /// 双线性插值是线性算子：`interp(α·x₁ + β·x₂) = α·interp(x₁) + β·interp(x₂)`。
    #[test]
    fn interpolation_is_linear(
        x1 in fixed(4), x2 in fixed(4), alpha in prop::finite(), beta in prop::finite()
    ) {
        let len = interpolate::OH * interpolate::OW;
        let mixed: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| alpha * a + beta * b).collect();
        let got = window_run(interpolate::forward, &mixed, len);
        let left = window_run(interpolate::forward, &x1, len);
        let right = window_run(interpolate::forward, &x2, len);
        let want: Vec<f32> = left.iter().zip(&right).map(|(a, b)| alpha * a + beta * b).collect();
        prop::assert_within_rel(&got, &want, 1e-5, "双线性插值的线性");
    }

    /// 平均池化同样是线性算子，且线性律与几何无关（不像"窗口均值"那样复述实现）。
    #[test]
    fn avg_pool2d_is_linear(
        x1 in fixed(16), x2 in fixed(16), alpha in prop::finite(), beta in prop::finite()
    ) {
        let len = avg_pool2d::H * avg_pool2d::W;
        let mixed: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| alpha * a + beta * b).collect();
        let got = window_run(avg_pool2d::forward, &mixed, len);
        let left = window_run(avg_pool2d::forward, &x1, len);
        let right = window_run(avg_pool2d::forward, &x2, len);
        let want: Vec<f32> = left.iter().zip(&right).map(|(a, b)| alpha * a + beta * b).collect();
        prop::assert_within_rel(&got, &want, 1e-5, "平均池化的线性");
    }
}
