//! 职责：池化：max/avg 窗口归约与输出体量。
use diagnostic::{Code, Error, Result};

use crate::Scalar;

use super::support::*;

use super::geom::ConvGeom;

/// 二维最大池化：补零位置不参与比较（与框架语义一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool2d {
    /// 池化核、步长、补零（补零位置不参与比较）。
    pub geom: ConvGeom,
}

/// 二维平均池化：口径同 [`MaxPool2d`]，`count_include_pad` 决定补零是否计入分母。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvgPool2d {
    /// 池化核、步长、补零。
    pub geom: ConvGeom,
    /// 是否把补零位置计入平均的分母。
    pub count_include_pad: bool,
}

impl MaxPool2d {
    /// 输出形状。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        out_shape(&self.geom, h, w)
    }

    /// 前向：逐窗口取最大，补零位不参与。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        run(&self.geom, x, n, (h, w), c, out, || {
            MaxFold(S::NEG_INFINITY)
        })
    }
}

impl AvgPool2d {
    /// 输出形状。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        out_shape(&self.geom, h, w)
    }

    /// 前向：窗口求和后按口径取分母；`count_include_pad=false` 时只除有效位置。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        let fold = || AvgFold {
            sum: S::ZERO,
            count: 0,
            area: self.geom.kh * self.geom.kw,
            include_pad: self.count_include_pad,
        };
        run(&self.geom, x, n, (h, w), c, out, fold)
    }
}

/// 池化窗口的归约口径：最大 / 平均。窗口遍历（补零裁剪、下标推导）全仓只有一份，
/// 两条口径只在累加器与收尾不同；泛型化后各自单态展开，内层循环里没有分支。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool1d {
    /// 窗口长。
    pub k: usize,
    /// 步长。
    pub s: usize,
    /// 两端补零。
    pub p: usize,
}

impl MaxPool1d {
    /// 输出长度 `(l + 2p − k) / s + 1`。
    ///
    /// # Errors
    ///
    /// 窗口/步长为 0，或窗口比补零后的输入还长时返回说明。
    pub fn len(&self, l: usize) -> Result<usize> {
        if self.k == 0 || self.s == 0 {
            return Err(Error::new(Code::Invalid, "池化窗口与步长必须大于 0"));
        }
        if l + 2 * self.p < self.k {
            return Err(Error::new(Code::Invalid, "池化窗口大于补零后的输入"));
        }
        Ok((l + 2 * self.p - self.k) / self.s + 1)
    }

    /// 前向：逐窗口取最大。
    ///
    /// # Errors
    ///
    /// 缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        l: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        let ol = self.len(l)?;
        check_shapes(&[
            (x.len(), n * l * c, "输入"),
            (out.len(), n * ol * c, "输出"),
        ])?;
        self.pool_max(x, out, n, l, c, ol);
        Ok(())
    }

    fn pool_max<S: Scalar>(&self, x: &[S], out: &mut [S], n: usize, l: usize, c: usize, ol: usize) {
        for ni in 0..n {
            for oi in 0..ol {
                for ci in 0..c {
                    out[(ni * ol + oi) * c + ci] = self.window_max(x, ni, oi, ci, l, c);
                }
            }
        }
    }

    fn window_max<S: Scalar>(
        &self,
        x: &[S],
        ni: usize,
        oi: usize,
        ci: usize,
        l: usize,
        c: usize,
    ) -> S {
        let base = oi * self.s;
        let mut best = S::NEG_INFINITY;
        for ki in 0..self.k {
            let src = base + ki;
            if src < self.p || src - self.p >= l {
                continue;
            }
            best = best.max(x[(ni * l + src - self.p) * c + ci]);
        }
        // 整窗都落在补零上（只会在极端几何下出现）按 0 处理。
        if best == S::NEG_INFINITY {
            S::ZERO
        } else {
            best
        }
    }
}

/// 二维自适应平均池化：任意输入尺寸都压到 `(oh, ow)`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveAvgPool2d {
    /// 输出高。
    pub oh: usize,
    /// 输出宽。
    pub ow: usize,
}

impl AdaptiveAvgPool2d {
    /// 前向：每个输出格取覆盖窗口的平均。
    ///
    /// # Errors
    ///
    /// 输出尺寸为 0 或缓冲长度不符时返回说明。
    pub fn forward<S: Scalar>(
        &self,
        x: &[S],
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        out: &mut [S],
    ) -> Result<()> {
        if self.oh == 0 || self.ow == 0 {
            return Err(Error::new(Code::Invalid, "自适应池化输出尺寸必须大于 0"));
        }
        check_shapes(&[
            (x.len(), n * h * w * c, "输入"),
            (out.len(), n * self.oh * self.ow * c, "输出"),
        ])?;
        self.pool_avg(x, out, n, (h, w), c);
        Ok(())
    }

    fn pool_avg<S: Scalar>(
        &self,
        x: &[S],
        out: &mut [S],
        n: usize,
        (h, w): (usize, usize),
        c: usize,
    ) {
        for ni in 0..n {
            for oi in 0..self.oh {
                let (h0, h1) = adaptive_span(oi, h, self.oh);
                for oj in 0..self.ow {
                    let (w0, w1) = adaptive_span(oj, w, self.ow);
                    let area = S::from_usize((h1 - h0) * (w1 - w0));
                    for ci in 0..c {
                        out[((ni * self.oh + oi) * self.ow + oj) * c + ci] =
                            cell_sum(x, (ni, ci), (h0, h1), (w0, w1), (h, w), c) / area;
                    }
                }
            }
        }
    }
}

/// 自适应池化的连续输出区间 `[lo, hi)`。
fn adaptive_span(i: usize, len: usize, out_len: usize) -> (usize, usize) {
    let lo = i * len / out_len;
    let hi = ((i + 1) * len).div_ceil(out_len).min(len);
    (lo, hi)
}

/// 求和 `x` 上矩形区域 `[h0,h1)×[w0,w1)` 的通道 `ci`。
fn cell_sum<S: Scalar>(
    x: &[S],
    (ni, ci): (usize, usize),
    (h0, h1): (usize, usize),
    (w0, w1): (usize, usize),
    (h, w): (usize, usize),
    c: usize,
) -> S {
    let mut sum = S::ZERO;
    for ii in h0..h1 {
        for jj in w0..w1 {
            sum = sum + x[((ni * h + ii) * w + jj) * c + ci];
        }
    }
    sum
}
