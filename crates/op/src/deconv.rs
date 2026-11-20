//! 转置卷积 `ConvTranspose2d`（解卷积/上采样），NHWC，权重布局对齐 PyTorch
//! `[Cin, Cout, kh, kw]`。
//!
//! 前向 = `x · Wᵀ`（一次 GEMM）再把结果按核位置**散射累加**回输出网格（`col2im`）。
//! 不写逐输出位置的双重循环：散射累加的算术与普通卷积同构，只是方向相反。

use diagnostic::{Code, Error, Result};

use crate::conv::ConvGeom;
use crate::fast;

/// 二维转置卷积：输入 `[n,h,w,cin]`，权重 `[cin,cout,kh,kw]`，输出 `[n,oh,ow,cout]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2d {
    /// 输入通道。
    pub cin: usize,
    /// 输出通道。
    pub cout: usize,
    /// 核、步长、补零（与正向卷积同一套语义）。
    pub geom: ConvGeom,
}

impl ConvTranspose2d {
    /// 由通道与几何构造。
    pub fn new(cin: usize, cout: usize, geom: ConvGeom) -> Self {
        Self { cin, cout, geom }
    }

    /// 输出高：`(h−1)·sh − 2·ph + kh`。
    ///
    /// # Errors
    ///
    /// 步长为 0 或结果下溢时返回说明。
    pub fn oh(&self, h: usize) -> Result<usize> {
        self.out_len(h, self.geom.sh, self.geom.kh, self.geom.ph, "h")
    }

    /// 输出宽；口径同 [`ConvTranspose2d::oh`]。
    ///
    /// # Errors
    ///
    /// 步长为 0 或结果下溢时返回说明。
    pub fn ow(&self, w: usize) -> Result<usize> {
        self.out_len(w, self.geom.sw, self.geom.kw, self.geom.pw, "w")
    }

    fn out_len(&self, inp: usize, s: usize, k: usize, p: usize, axis: &str) -> Result<usize> {
        if s == 0 {
            return Err(Error::new(Code::Invalid, format!("{axis}: 步长必须非零")));
        }
        let padded = (inp.saturating_sub(1)) * s;
        if padded + k < 2 * p {
            return Err(Error::new(
                Code::Invalid,
                format!("{axis}: 输出尺寸下溢（{inp}×{s}+{k} 小于 2·{p}）"),
            ));
        }
        Ok(padded + k - 2 * p)
    }

    /// f32 前向：GEMM 得列矩阵，再散射累加到输出。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    pub fn forward_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n: usize,
        h: usize,
        w: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let oh = self.oh(h)?;
        let ow = self.ow(w)?;
        let (cin, cout) = (self.cin, self.cout);
        let karea = self.geom.kh * self.geom.kw;
        check(x.len(), n * h * w * cin, "x")?;
        check(weight.len(), cin * cout * karea, "weight")?;
        check(bias.len(), cout, "bias")?;
        check(out.len(), n * oh * ow * cout, "out")?;

        let rows = n * h * w;
        let cols = cout * karea;
        // 权重转置：`[cin, cout, kh, kw]` → `[cin, cout·kh·kw]`。
        let mut wt = vec![0.0f32; cin * cols];
        for ci in 0..cin {
            for co in 0..cout {
                for ki in 0..self.geom.kh {
                    for kj in 0..self.geom.kw {
                        let src = ((ci * cout + co) * self.geom.kh + ki) * self.geom.kw + kj;
                        wt[ci * cols + (ki * self.geom.kw + kj) * cout + co] = weight[src];
                    }
                }
            }
        }
        let mut col = vec![0.0f32; rows * cols];
        fast::gemm_f32(x, &wt, rows, cols, cin, &mut col);

        out[..n * oh * ow * cout].fill(0.0);
        scatter(&col, out, n, (h, w), (oh, ow), self, cout);
        for p in 0..n * oh * ow {
            for co in 0..cout {
                out[p * cout + co] += bias[co];
            }
        }
        Ok(())
    }

    /// i8 前向（对称量化）：GEMM 走多槽池，散射累加在 i32 上。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_i8(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        h: usize,
        w: usize,
        out: &mut [i32],
    ) -> Result<()> {
        let oh = self.oh(h)?;
        let ow = self.ow(w)?;
        let (cin, cout) = (self.cin, self.cout);
        let karea = self.geom.kh * self.geom.kw;
        check(x.len(), n * h * w * cin, "x")?;
        check(weight.len(), cin * cout * karea, "weight")?;
        check(bias.len(), cout, "bias")?;
        check(out.len(), n * oh * ow * cout, "out")?;

        let rows = n * h * w;
        let cols = cout * karea;
        let mut wt = vec![0i8; cin * cols];
        for ci in 0..cin {
            for co in 0..cout {
                for ki in 0..self.geom.kh {
                    for kj in 0..self.geom.kw {
                        let src = ((ci * cout + co) * self.geom.kh + ki) * self.geom.kw + kj;
                        wt[ci * cols + (ki * self.geom.kw + kj) * cout + co] = weight[src];
                    }
                }
            }
        }
        let mut col = vec![0i32; rows * cols];
        fast::gemm_i8(x, &wt, rows, cols, cin, &mut col);

        out[..n * oh * ow * cout].fill(0);
        scatter_i8(&col, out, n, (h, w), (oh, ow), self, cout);
        for p in 0..n * oh * ow {
            for co in 0..cout {
                out[p * cout + co] += bias[co];
            }
        }
        Ok(())
    }
}

/// f32 散射：把列矩阵 `[n·h·w, cout·kh·kw]` 的第 `(ki,kj,co)` 项累加到 `(hi·sh−ph+ki, wi·sw−pw+kj)`。
fn scatter(
    col: &[f32],
    out: &mut [f32],
    n: usize,
    (h, w): (usize, usize),
    (oh, ow): (usize, usize),
    conv: &ConvTranspose2d,
    cout: usize,
) {
    let g = conv.geom;
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let row = &col[((ni * h + hi) * w + wi) * (cout * g.kh * g.kw)..][..cout * g.kh * g.kw];
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
                        let dst = ((ni * oh + ii as usize) * ow + jj as usize) * cout;
                        let src = (ki * g.kw + kj) * cout;
                        for co in 0..cout {
                            out[dst + co] += row[src + co];
                        }
                    }
                }
            }
        }
    }
}

/// i8 散射：口径与 [`scatter`] 一致，累加在 i32。
fn scatter_i8(
    col: &[i32],
    out: &mut [i32],
    n: usize,
    (h, w): (usize, usize),
    (oh, ow): (usize, usize),
    conv: &ConvTranspose2d,
    cout: usize,
) {
    let g = conv.geom;
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let row = &col[((ni * h + hi) * w + wi) * (cout * g.kh * g.kw)..][..cout * g.kh * g.kw];
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
                        let dst = ((ni * oh + ii as usize) * ow + jj as usize) * cout;
                        let src = (ki * g.kw + kj) * cout;
                        for co in 0..cout {
                            out[dst + co] += row[src + co];
                        }
                    }
                }
            }
        }
    }
}

/// 缓冲长度校验。
fn check(actual: usize, required: usize, what: &str) -> Result<()> {
    if actual < required {
        return Err(Error::new(
            Code::Invalid,
            format!("{what}: 缓冲 {actual}，需要 {required}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let z = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((z >> 8) % 2048) as f32 / 1024.0 - 1.0
            })
            .collect()
    }

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

    fn close(a: &[f32], b: &[f32]) {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let tol = 1e-4 + 1e-3 * y.abs();
            assert!((x - y).abs() <= tol, "第 {i} 个元素 {x} vs {y}");
        }
    }

    #[test]
    fn matches_naive_across_geometries() {
        let cases = [
            ((1, 4, 4, 2, 3), ConvGeom { kh: 3, kw: 3, sh: 2, sw: 2, ph: 1, pw: 1 }),
            ((2, 3, 5, 3, 2), ConvGeom { kh: 2, kw: 2, sh: 2, sw: 1, ph: 0, pw: 0 }),
            ((1, 5, 5, 1, 1), ConvGeom { kh: 4, kw: 4, sh: 3, sw: 3, ph: 1, pw: 2 }),
        ];
        for ((n, h, w, cin, cout), geom) in cases {
            let de = ConvTranspose2d::new(cin, cout, geom);
            let oh = de.oh(h).unwrap();
            let ow = de.ow(w).unwrap();
            let x = sample(n * h * w * cin, 1);
            let wgt = sample(cin * cout * geom.kh * geom.kw, 2);
            let bias = sample(cout, 3);
            let mut got = vec![0.0f32; n * oh * ow * cout];
            let mut want = vec![0.0f32; n * oh * ow * cout];
            de.forward_f32(&x, &wgt, &bias, n, h, w, &mut got).unwrap();
            naive(&x, &wgt, &bias, &de, n, h, w, &mut want);
            close(&got, &want);
        }
    }

    #[test]
    fn transposed_conv_inverts_spatial_downsampling() {
        // stride=2、k=2、无补零：输出恰好是输入的 2 倍边长。
        let de = ConvTranspose2d::new(
            1,
            1,
            ConvGeom { kh: 2, kw: 2, sh: 2, sw: 2, ph: 0, pw: 0 },
        );
        assert_eq!(de.oh(4).unwrap(), 8);
        assert_eq!(de.ow(4).unwrap(), 8);
    }

    #[test]
    fn i8_matches_integer_naive() {
        let de = ConvTranspose2d::new(
            2,
            3,
            ConvGeom { kh: 3, kw: 3, sh: 2, sw: 2, ph: 1, pw: 1 },
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
                                    want[((ni * oh + ii as usize) * ow + jj as usize) * de.cout + co] +=
                                        i32::from(xv) * i32::from(wv);
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
            ConvGeom { kh: 1, kw: 1, sh: 1, sw: 1, ph: 3, pw: 3 },
        );
        assert!(de.oh(1).is_err());
    }
}
