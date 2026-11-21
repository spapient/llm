use super::*;

fn sample_i8(len: usize, salt: u32) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let z = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((z % 15) as i32 - 7) as i8
        })
        .collect()
}

/// 朴素转置卷积：逐输入位置、逐核元素散射累加。
#[allow(clippy::too_many_arguments)]
fn naive(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    de: &ConvTranspose2d,
    n: usize,
    h: usize,
    w: usize,
    out: &mut [f32],
) {
    let g = de.geom;
    let (cin, cout) = (de.cin, de.cout);
    let oh = de.oh(h).unwrap();
    let ow = de.ow(w).unwrap();
    out.fill(0.0);
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ki in 0..g.kh {
                    let ii = (hi * g.sh + ki) as isize - g.ph as isize;
                    if ii < 0 || ii as usize >= oh {
                        continue;
                    }
                    for kj in 0..g.kw {
                        let jj = (wi * g.sw + kj) as isize - g.pw as isize;
                        if jj < 0 || jj as usize >= ow {
                            continue;
                        }
                        for ci in 0..cin {
                            let xv = x[((ni * h + hi) * w + wi) * cin + ci];
                            for co in 0..cout {
                                let wv = weight[((ci * cout + co) * g.kh + ki) * g.kw + kj];
                                out[((ni * oh + ii as usize) * ow + jj as usize) * cout + co] +=
                                    xv * wv;
                            }
                        }
                    }
                }
            }
        }
    }
    for p in 0..n * oh * ow {
        for co in 0..cout {
            out[p * cout + co] += bias[co];
        }
    }
}

#[test]
fn matches_naive_across_geometries() {
    let cases = [
        (
            (1, 4, 4, 2, 3),
            ConvGeom {
                kh: 3,
                kw: 3,
                sh: 2,
                sw: 2,
                ph: 1,
                pw: 1,
            },
        ),
        (
            (2, 3, 5, 3, 2),
            ConvGeom {
                kh: 2,
                kw: 2,
                sh: 2,
                sw: 1,
                ph: 0,
                pw: 0,
            },
        ),
        (
            (1, 5, 5, 1, 1),
            ConvGeom {
                kh: 4,
                kw: 4,
                sh: 3,
                sw: 3,
                ph: 1,
                pw: 2,
            },
        ),
    ];
    for ((n, h, w, cin, cout), geom) in cases {
        let de = ConvTranspose2d::new(cin, cout, geom);
        let oh = de.oh(h).unwrap();
        let ow = de.ow(w).unwrap();
        let x = testkit::sample(n * h * w * cin, 1);
        let wgt = testkit::sample(cin * cout * geom.kh * geom.kw, 2);
        let bias = testkit::sample(cout, 3);
        let mut got = vec![0.0f32; n * oh * ow * cout];
        let mut want = vec![0.0f32; n * oh * ow * cout];
        de.forward_f32(&x, &wgt, &bias, n, h, w, &mut got).unwrap();
        naive(&x, &wgt, &bias, &de, n, h, w, &mut want);
        testkit::assert_close(&got, &want, 1e-4, 1e-3);
    }
}

#[test]
fn transposed_conv_inverts_spatial_downsampling() {
    // stride=2、k=2、无补零：输出恰好是输入的 2 倍边长。
    let de = ConvTranspose2d::new(
        1,
        1,
        ConvGeom {
            kh: 2,
            kw: 2,
            sh: 2,
            sw: 2,
            ph: 0,
            pw: 0,
        },
    );
    assert_eq!(de.oh(4).unwrap(), 8);
    assert_eq!(de.ow(4).unwrap(), 8);
}

#[test]
fn i8_matches_integer_naive() {
    let de = ConvTranspose2d::new(
        2,
        3,
        ConvGeom {
            kh: 3,
            kw: 3,
            sh: 2,
            sw: 2,
            ph: 1,
            pw: 1,
        },
    );
    let (n, h, w) = (1, 3, 3);
    let oh = de.oh(h).unwrap();
    let ow = de.ow(w).unwrap();
    let x = sample_i8(n * h * w * de.cin, 1);
    let wgt = sample_i8(de.cin * de.cout * 9, 2);
    let bias = vec![0i32; de.cout];
    let mut got = vec![0i32; n * oh * ow * de.cout];
    de.forward_i8(&x, &wgt, &bias, n, h, w, &mut got).unwrap();

    // i32 直算参考。
    let mut want = vec![0i32; n * oh * ow * de.cout];
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ki in 0..3 {
                    let ii = (hi * 2 + ki) as isize - 1;
                    if ii < 0 || ii as usize >= oh {
                        continue;
                    }
                    for kj in 0..3 {
                        let jj = (wi * 2 + kj) as isize - 1;
                        if jj < 0 || jj as usize >= ow {
                            continue;
                        }
                        for ci in 0..de.cin {
                            let xv = x[((ni * h + hi) * w + wi) * de.cin + ci];
                            for co in 0..de.cout {
                                let wv = wgt[((ci * de.cout + co) * 3 + ki) * 3 + kj];
                                want[((ni * oh + ii as usize) * ow + jj as usize) * de.cout
                                    + co] += i32::from(xv) * i32::from(wv);
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(got, want);
}

#[test]
fn rejects_underflowing_geometry() {
    let de = ConvTranspose2d::new(
        1,
        1,
        ConvGeom {
            kh: 1,
            kw: 1,
            sh: 1,
            sw: 1,
            ph: 3,
            pw: 3,
        },
    );
    assert!(de.oh(1).is_err());
}

/// 一维朴素转置卷积：逐输入位置、逐核元素散射累加后加偏置。
#[allow(clippy::too_many_arguments)]
fn naive1(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    de: &ConvTranspose1d,
    n: usize,
    l: usize,
    out: &mut [f32],
) {
    let ol = de.len(l).unwrap();
    out.fill(0.0);
    for ni in 0..n {
        for i in 0..l {
            for ki in 0..de.k {
                let dst = (i * de.s + ki) as isize - de.p as isize;
                if dst < 0 || dst as usize >= ol {
                    continue;
                }
                for ci in 0..de.cin {
                    let xv = x[(ni * l + i) * de.cin + ci];
                    for co in 0..de.cout {
                        let wv = weight[(ci * de.cout + co) * de.k + ki];
                        out[((ni * ol + dst as usize) * de.cout) + co] += xv * wv;
                    }
                }
            }
        }
    }
    for pos in 0..n * ol {
        for co in 0..de.cout {
            out[pos * de.cout + co] += bias[co];
        }
    }
}

#[test]
fn conv_transpose1d_matches_naive() {
    // 三组几何：same（k3/s1/p1）、放大（k4/s2/p1）、不合补零（k2/s2/p0）。
    let cases = [
        (3usize, 1usize, 1usize, 4usize, 5usize, 2usize),
        (4, 2, 1, 3, 6, 3),
        (2, 2, 0, 5, 4, 2),
    ];
    for (index, (k, s, p, cin, cout, l)) in cases.iter().enumerate() {
        let de = ConvTranspose1d::new(*cin, *cout, *k, *s, *p);
        let ol = de.len(*l).unwrap();
        let x = testkit::sample(*l * *cin, 1 + index as u32);
        let w = testkit::sample(*cin * *cout * *k, 7 + index as u32);
        let b = testkit::sample(*cout, 3);
        let mut got = vec![0.0f32; ol * *cout];
        de.forward_f32(&x, &w, &b, 1, *l, &mut got).unwrap();
        let mut want = vec![0.0f32; ol * *cout];
        naive1(&x, &w, &b, &de, 1, *l, &mut want);
        for (a, e) in got.iter().zip(want.iter()) {
            assert!((a - e).abs() < 1e-4, "case {index}: {a} vs {e}");
        }
    }
}

#[test]
fn conv_transpose1d_stride_doubles_length() {
    let de = ConvTranspose1d::new(1, 1, 2, 2, 0);
    assert_eq!(de.len(4).unwrap(), 8);
    assert_eq!(de.len(1).unwrap(), 2);
    // 上采样后的每个输入样本落在偶数位，奇数位是相邻样本的核重叠。
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let w = [1.0f32, 0.0];
    let mut out = [0.0f32; 8];
    de.forward_f32(&x, &w, &[0.0], 1, 4, &mut out).unwrap();
    assert_eq!(out, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
}

#[test]
fn conv_transpose1d_i8_matches_integer_reference() {
    let de = ConvTranspose1d::new(3, 2, 3, 2, 1);
    let l = 5;
    let ol = de.len(l).unwrap();
    let x = sample_i8(l * 3, 2);
    let w = sample_i8(3 * 2 * 3, 5);
    let b = [3i32, -2];
    let mut got = vec![0i32; ol * 2];
    de.forward_i8(&x, &w, &b, 1, l, &mut got).unwrap();
    // 整数直算：等价于把 f32 朴素式在 i32 上重跑。
    let mut want = vec![0i32; ol * 2];
    for i in 0..l {
        for ki in 0..de.k {
            let dst = i * de.s + ki;
            if dst < de.p || dst - de.p >= ol {
                continue;
            }
            for ci in 0..3 {
                for co in 0..2 {
                    want[((dst - de.p) * 2) + co] +=
                        i32::from(x[i * 3 + ci]) * i32::from(w[(ci * 2 + co) * 3 + ki]);
                }
            }
        }
    }
    for pos in 0..ol {
        for co in 0..2 {
            want[pos * 2 + co] += b[co];
        }
    }
    assert_eq!(got, want);
}

#[test]
fn conv_transpose1d_rejects_bad_geometry() {
    let de = ConvTranspose1d::new(1, 1, 1, 0, 0);
    assert!(de.len(2).is_err());
    let de = ConvTranspose1d::new(1, 1, 1, 1, 3);
    assert!(de.len(1).is_err());
}
