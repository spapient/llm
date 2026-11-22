//! 转置卷积 `ConvTranspose2d`（解卷积/上采样），NHWC，权重布局对齐 PyTorch
//! `[Cin, Cout, kh, kw]`。

use std::ops::AddAssign;

use crate::guard::check_all;
use diagnostic::{Code, Error, Result};

use crate::conv::ConvGeom;
use crate::kernel::fast;

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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
#[path = "../../tests/conv/deconv_tests.rs"]
mod tests;
