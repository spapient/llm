//! 转置卷积 `ConvTranspose2d`（解卷积/上采样），NHWC，权重布局对齐 PyTorch
//! `[Cin, Cout, kh, kw]`。
//!
//! 前向 = `x · Wᵀ`（一次 GEMM）再把结果按核位置**散射累加**回输出网格（`col2im`）。
//! 不写逐输出位置的双重循环：散射累加的算术与普通卷积同构，只是方向相反。

use std::ops::AddAssign;

use diagnostic::{Code, Error, Result};
use crate::guard::check_all;

use crate::conv::ConvGeom;
use crate::fast;

/// 前向标量缝：f32 与对称 i8（i32 累加）共用同一套前向/散射算法，
/// 只有 GEMM 后端与零值随类型切换。换后端只需新增一个 impl，调用方不变。
trait DeconvScalar: Copy + Default {
    /// 累加/输出类型：f32 → f32，i8 → i32。
    type Acc: Copy + Default + AddAssign;

    /// 列矩阵 GEMM：`col = x · wt`，收缩维为 `inner`。
    fn gemm(x: &[Self], wt: &[Self], rows: usize, cols: usize, inner: usize, col: &mut [Self::Acc]);
}

impl DeconvScalar for f32 {
    type Acc = f32;

    fn gemm(x: &[f32], wt: &[f32], rows: usize, cols: usize, inner: usize, col: &mut [f32]) {
        fast::gemm_f32(x, wt, rows, cols, inner, col);
    }
}

impl DeconvScalar for i8 {
    type Acc = i32;

    fn gemm(x: &[i8], wt: &[i8], rows: usize, cols: usize, inner: usize, col: &mut [i32]) {
        fast::gemm_i8(x, wt, rows, cols, inner, col);
    }
}

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
        forward(self, x, weight, bias, n, h, w, out)
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
        forward(self, x, weight, bias, n, h, w, out)
    }
}

/// 二维前向核心：标量类型经 [`DeconvScalar`] 参数化，f32/i8 共用同一份算法。
#[allow(clippy::too_many_arguments)]
fn forward<S: DeconvScalar>(
    conv: &ConvTranspose2d,
    x: &[S],
    weight: &[S],
    bias: &[S::Acc],
    n: usize,
    h: usize,
    w: usize,
    out: &mut [S::Acc],
) -> Result<()> {
    let oh = conv.oh(h)?;
    let ow = conv.ow(w)?;
    let (cin, cout) = (conv.cin, conv.cout);
    let karea = conv.geom.kh * conv.geom.kw;
    check_all(&[
        (x.len(), n * h * w * cin, "x"),
        (weight.len(), cin * cout * karea, "weight"),
        (bias.len(), cout, "bias"),
        (out.len(), n * oh * ow * cout, "out"),
    ])?;

    let rows = n * h * w;
    let cols = cout * karea;
    let mut wt = vec![S::default(); cin * cols];
    transpose_weight_2d(weight, &mut wt, cin, cout, conv.geom.kh, conv.geom.kw);
    let mut col = vec![S::Acc::default(); rows * cols];
    S::gemm(x, &wt, rows, cols, cin, &mut col);

    out[..n * oh * ow * cout].fill(S::Acc::default());
    scatter(&col, out, n, (h, w), (oh, ow), conv, cout);
    add_bias_2d(out, bias, n, oh, ow, cout);
    Ok(())
}

/// 散射：把列矩阵 `[n·h·w, cout·kh·kw]` 的第 `(ki,kj,co)` 项累加到 `(hi·sh−ph+ki, wi·sw−pw+kj)`。
fn scatter<T: Copy + AddAssign>(
    col: &[T],
    out: &mut [T],
    n: usize,
    (h, w): (usize, usize),
    (oh, ow): (usize, usize),
    conv: &ConvTranspose2d,
    cout: usize,
) {
    let g = conv.geom;
    let karea = cout * g.kh * g.kw;
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                let row = &col[((ni * h + hi) * w + wi) * karea..][..karea];
                scatter_row_2d(row, out, g, oh, ow, cout, ni, hi, wi);
            }
        }
    }
}



/// 一维反卷积：NHWC，权重布局对齐 PyTorch `[Cin, Cout, k]`。
///
/// 输出长度 `(l − 1)·s − 2p + k`；每个输入位置的 `cin` 向量乘上 `Cout×k` 的权重块，
/// 再按 `i·s − p + ki` 散射累加。GEMM 与二维版同一支（列矩阵 `[l, cout·k]`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1d {
    /// 输入通道。
    pub cin: usize,
    /// 输出通道。
    pub cout: usize,
    /// 窗口长。
    pub k: usize,
    /// 步长。
    pub s: usize,
    /// 两端补零（等价于输出两侧裁掉的量）。
    pub p: usize,
}

impl ConvTranspose1d {
    /// 构造：通道、窗口与步长都必须非零。
    #[must_use]
    pub fn new(cin: usize, cout: usize, k: usize, s: usize, p: usize) -> Self {
        Self { cin, cout, k, s, p }
    }

    /// 输出长度 `(l − 1)·s − 2p + k`。
    ///
    /// # Errors
    ///
    /// 步长/窗口为 0，或 `(l − 1)·s < 2p − k + 1`（输出长度非正）时返回说明。
    pub fn len(&self, l: usize) -> Result<usize> {
        if self.s == 0 || self.k == 0 {
            return Err(Error::new(Code::Invalid, "反卷积窗口与步长必须大于 0"));
        }
        let base = l
            .checked_sub(1)
            .and_then(|v| v.checked_mul(self.s))
            .ok_or(Error::new(Code::Invalid, "反卷积输出长度溢出"))?;
        let span = base + self.k;
        let pad = 2 * self.p;
        // 输出必须严格为正：`span − pad ≥ 1`。
        if span < pad + 1 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "反卷积输出长度非正（l={l}, s={}, p={}, k={}）",
                    self.s, self.p, self.k
                ),
            ));
        }
        Ok(span - pad)
    }

    /// f32 前向：GEMM 得列矩阵，再散射累加。
    ///
    /// # Errors
    ///
    /// 缓冲长度与几何不符时返回说明。
    pub fn forward_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n: usize,
        l: usize,
        out: &mut [f32],
    ) -> Result<()> {
        forward_1d(self, x, weight, bias, n, l, out)
    }

    /// i8 前向：GEMM 走多槽池，散射累加在 i32 上。
    ///
    /// # Errors
    ///
    /// 缓冲长度与几何不符时返回说明。
    pub fn forward_i8(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        l: usize,
        out: &mut [i32],
    ) -> Result<()> {
        forward_1d(self, x, weight, bias, n, l, out)
    }
}

/// 一维前向核心：标量类型经 [`DeconvScalar`] 参数化，f32/i8 共用同一份算法。
fn forward_1d<S: DeconvScalar>(
    conv: &ConvTranspose1d,
    x: &[S],
    weight: &[S],
    bias: &[S::Acc],
    n: usize,
    l: usize,
    out: &mut [S::Acc],
) -> Result<()> {
    let ol = conv.len(l)?;
    let (cin, cout, k) = (conv.cin, conv.cout, conv.k);
    let cols = cout * k;
    check_all(&[
        (x.len(), n * l * cin, "x"),
        (weight.len(), cin * cols, "weight"),
        (bias.len(), cout, "bias"),
        (out.len(), n * ol * cout, "out"),
    ])?;

    let mut wt = vec![S::default(); cin * cols];
    transpose_weight_1d(weight, &mut wt, cin, cout, k);
    let mut col = vec![S::Acc::default(); n * l * cols];
    S::gemm(x, &wt, n * l, cols, cin, &mut col);

    out[..n * ol * cout].fill(S::Acc::default());
    scatter1(&col, out, n, l, ol, conv, cout);
    add_bias_1d(out, bias, n, ol, cout);
    Ok(())
}

/// 列矩阵 `[n·l, cout·k]` 散射累加到 `[n, ol, cout]`：越界位置裁掉。
fn scatter1<T: Copy + AddAssign>(
    col: &[T],
    out: &mut [T],
    n: usize,
    l: usize,
    ol: usize,
    conv: &ConvTranspose1d,
    cout: usize,
) {
    for ni in 0..n {
        for i in 0..l {
            let src = (ni * l + i) * cout * conv.k;
            scatter1_at(col, out, ol, conv, cout, ni, i, src);
        }
    }
}

/// `0 <= idx < n`：散射越界裁剪共用的边界谓词。
fn in_range(idx: isize, n: usize) -> bool {
    idx >= 0 && idx < n as isize
}

/// 二维权重转置 `[cin,cout,kh,kw]` → `[cin, cout·kh·kw]`。
fn transpose_weight_2d<T: Copy>(
    weight: &[T],
    wt: &mut [T],
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
) {
    let cols = cout * kh * kw;
    for ci in 0..cin {
        for co in 0..cout {
            for ki in 0..kh {
                for kj in 0..kw {
                    let src = ((ci * cout + co) * kh + ki) * kw + kj;
                    wt[ci * cols + (ki * kw + kj) * cout + co] = weight[src];
                }
            }
        }
    }
}

/// 二维偏置累加：`out[n,oh,ow,cout] += bias[cout]`。
fn add_bias_2d<T: Copy + AddAssign>(
    out: &mut [T],
    bias: &[T],
    n: usize,
    oh: usize,
    ow: usize,
    cout: usize,
) {
    for p in 0..n * oh * ow {
        for co in 0..cout {
            out[p * cout + co] += bias[co];
        }
    }
}

/// 单个输入行的散射：沿核高 `kh` 散射，越界行裁掉。
#[allow(clippy::too_many_arguments)]
fn scatter_row_2d<T: Copy + AddAssign>(
    row: &[T],
    out: &mut [T],
    g: ConvGeom,
    oh: usize,
    ow: usize,
    cout: usize,
    ni: usize,
    hi: usize,
    wi: usize,
) {
    for ki in 0..g.kh {
        let ii = (hi * g.sh + ki) as isize - g.ph as isize;
        if !in_range(ii, oh) {
            continue;
        }
        scatter_kernel_2d(row, out, g, oh, ow, cout, ni, ii as usize, wi, ki);
    }
}

/// 单个核位置的散射：沿核宽 `kw` 散射，越界列裁掉，`cout` 通道逐个累加。
#[allow(clippy::too_many_arguments)]
fn scatter_kernel_2d<T: Copy + AddAssign>(
    row: &[T],
    out: &mut [T],
    g: ConvGeom,
    oh: usize,
    ow: usize,
    cout: usize,
    ni: usize,
    ii: usize,
    wi: usize,
    ki: usize,
) {
    for kj in 0..g.kw {
        let jj = (wi * g.sw + kj) as isize - g.pw as isize;
        if !in_range(jj, ow) {
            continue;
        }
        let dst = ((ni * oh + ii) * ow + jj as usize) * cout;
        let src = (ki * g.kw + kj) * cout;
        for co in 0..cout {
            out[dst + co] += row[src + co];
        }
    }
}

/// 一维权重转置 `[cin,cout,k]` → `[cin, cout·k]`。
fn transpose_weight_1d<T: Copy>(weight: &[T], wt: &mut [T], cin: usize, cout: usize, k: usize) {
    let cols = cout * k;
    for ci in 0..cin {
        for co in 0..cout {
            for ki in 0..k {
                wt[ci * cols + ki * cout + co] = weight[(ci * cout + co) * k + ki];
            }
        }
    }
}

/// 一维偏置累加：`out[n,ol,cout] += bias[cout]`。
fn add_bias_1d<T: Copy + AddAssign>(out: &mut [T], bias: &[T], n: usize, ol: usize, cout: usize) {
    for pos in 0..n * ol {
        for co in 0..cout {
            out[pos * cout + co] += bias[co];
        }
    }
}

/// 一维单个输入位置的散射：沿窗口 `k` 散射，越界位置裁掉。
#[allow(clippy::too_many_arguments)]
fn scatter1_at<T: Copy + AddAssign>(
    col: &[T],
    out: &mut [T],
    ol: usize,
    conv: &ConvTranspose1d,
    cout: usize,
    ni: usize,
    i: usize,
    src: usize,
) {
    for ki in 0..conv.k {
        let dst = (i * conv.s + ki) as isize - conv.p as isize;
        if !in_range(dst, ol) {
            continue;
        }
        let d = (ni * ol + dst as usize) * cout;
        let src_k = src + ki * cout;
        for co in 0..cout {
            out[d + co] += col[src_k + co];
        }
    }
}

#[cfg(test)]
mod tests {
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
                                    out[((ni * oh + ii as usize) * ow + jj as usize) * cout
                                        + co] += xv * wv;
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
}
