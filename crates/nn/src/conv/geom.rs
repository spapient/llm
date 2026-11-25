//! 职责：卷积几何：输入输出体量与切片步幅计算。
use diagnostic::{Code, Error, Result};

use crate::kernel::fast;

use crate::conv::support::*;

/// 卷积几何参数：核、步长、补零。1d 层把 `kh`/`sh` 固定为 1 以复用 2d 的展开。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvGeom {
    /// 卷积核高（1d 恒为 1）。
    pub kh: usize,
    /// 卷积核宽。
    pub kw: usize,
    /// 步长高（1d 恒为 1）。
    pub sh: usize,
    /// 步长宽。
    pub sw: usize,
    /// 补零高。
    pub ph: usize,
    /// 补零宽。
    pub pw: usize,
}

impl ConvGeom {
    /// 输出高：`floor((h + 2·ph - kh) / sh) + 1`，下溢即形状不合法。
    ///
    /// # Errors
    ///
    /// `h + 2·ph < kh` 或 `sh == 0` 时返回错误。
    pub fn oh(&self, h: usize) -> Result<usize> {
        self.out_len(h, self.ph, self.kh, self.sh, "h")
    }

    /// 输出宽；口径同 [`ConvGeom::oh`]。
    pub fn ow(&self, w: usize) -> Result<usize> {
        self.out_len(w, self.pw, self.kw, self.sw, "w")
    }

    fn out_len(&self, input: usize, pad: usize, k: usize, s: usize, axis: &str) -> Result<usize> {
        if s == 0 || k == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{axis}: 核与步长必须非零"),
            ));
        }
        let padded = padded_len(input, pad, axis)?;
        if padded < k {
            return Err(Error::new(
                Code::Invalid,
                format!("{axis}: 补零后 {padded} 小于核 {k}，没有合法输出"),
            ));
        }
        Ok((padded - k) / s + 1)
    }

    /// 一行 im2col 的饱和估算长度；执行入口使用 [`Self::checked_patch`] 拒绝溢出。
    pub fn patch(&self, cin: usize) -> usize {
        self.kh.saturating_mul(self.kw).saturating_mul(cin)
    }

    /// 一行 im2col 的受检长度，任一乘法溢出时返回诊断。
    pub fn checked_patch(&self, cin: usize) -> Result<usize> {
        self.kh
            .checked_mul(self.kw)
            .and_then(|kernel| kernel.checked_mul(cin))
            .ok_or_else(|| Error::new(Code::Invalid, "卷积核展开长度溢出"))
    }
}

/// 补零后的轴长 `input + 2·pad`：受检乘法与加法，任何溢出都在输出长度计算前被拒绝。
fn padded_len(input: usize, pad: usize, axis: &str) -> Result<usize> {
    pad.checked_mul(2)
        .and_then(|padding| input.checked_add(padding))
        .ok_or_else(|| Error::new(Code::Invalid, format!("{axis}: 补零尺寸溢出")))
}

/// 一维卷积：输入 `[n, l, cin]`、权重 `[kw, cin, cout]`、输出 `[n, ol, cout]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conv1d {
    /// 输入/输出通道与核宽、步长、补零。
    pub cin: usize,
    /// 输出通道。
    pub cout: usize,
    /// 核、步长、补零（`kh`/`sh`/`ph` 恒为 1）。
    pub geom: ConvGeom,
}

impl Conv1d {
    /// 按核宽与步长构造；补零默认 0。
    pub fn new(cin: usize, cout: usize, kw: usize, stride: usize, pad: usize) -> Self {
        Self {
            cin,
            cout,
            geom: ConvGeom {
                kh: 1,
                kw,
                sh: 1,
                sw: stride,
                ph: 0,
                pw: pad,
            },
        }
    }

    /// 输出长度。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn len(&self, l: usize) -> Result<usize> {
        self.geom.ow(l)
    }

    /// 输出长度是否为 0（clippy 要求与 `len` 成对出现）。
    pub fn is_empty(&self, l: usize) -> Result<bool> {
        Ok(self.len(l)? == 0)
    }

    /// f32 前向：im2col → GEMM，再按需加偏置。
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
        l: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let ol = self.len(l)?;
        let cin = self.cin;
        let cout = self.cout;
        let patch = self.geom.checked_patch(cin)?;
        check_shapes(&[
            (x.len(), n * l * cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (bias.len(), cout, "偏置"),
            (out.len(), n * ol * cout, "输出"),
        ])?;

        let mut col = vec![0.0f32; n * ol * patch];
        im2col1d(x, &mut col, n, l, ol, self.geom, cin);
        gemm(&col, weight, n * ol, cout, patch, out);
        add_bias(out, bias, cout);
        Ok(())
    }
}

/// 二维卷积：输入 `[n, h, w, cin]`、权重 `[kh, kw, cin, cout]`、输出 `[n, oh, ow, cout]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conv2d {
    /// 输入通道。
    pub cin: usize,
    /// 输出通道。
    pub cout: usize,
    /// 核、步长、补零。
    pub geom: ConvGeom,
}

impl Conv2d {
    /// 按核、步长、补零构造。
    pub fn new(cin: usize, cout: usize, geom: ConvGeom) -> Self {
        Self { cin, cout, geom }
    }

    /// 输出形状 `(oh, ow)`。
    ///
    /// # Errors
    ///
    /// 形状不合法时返回说明。
    pub fn shape(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        Ok((self.geom.oh(h)?, self.geom.ow(w)?))
    }

    /// f32 前向：im2col → GEMM → 偏置。
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
        (h, w): (usize, usize),
        out: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cin = self.cin;
        let cout = self.cout;
        let patch = self.geom.checked_patch(cin)?;
        check_shapes(&[
            (x.len(), n * h * w * cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (bias.len(), cout, "偏置"),
            (out.len(), n * oh * ow * cout, "输出"),
        ])?;

        let rows = n * oh * ow;
        let mut col = vec![0.0f32; rows * patch];
        im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
        gemm(&col, weight, rows, cout, patch, out);
        add_bias(out, bias, cout);
        Ok(())
    }

    /// f32 反向：上游梯度 `dy` → `dx / dweight / dbias`。
    ///
    /// 与前向共用同一份 `im2col` 与卷积几何，三个输出缓冲**先清零再累加**（多上游可叠加）。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    #[allow(clippy::too_many_arguments)]
    pub fn backward_f32(
        &self,
        x: &[f32],
        weight: &[f32],
        dy: &[f32],
        n: usize,
        h: usize,
        w: usize,
        dx: &mut [f32],
        dweight: &mut [f32],
        dbias: &mut [f32],
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let (cin, cout) = (self.cin, self.cout);
        let patch = self.geom.checked_patch(cin)?;
        let rows = n * oh * ow;
        check_shapes(&[
            (x.len(), n * h * w * cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (dy.len(), rows * cout, "上游梯度"),
            (dx.len(), n * h * w * cin, "输入梯度"),
            (dweight.len(), patch * cout, "权重梯度"),
            (dbias.len(), cout, "偏置梯度"),
        ])?;

        let mut col = vec![0.0f32; rows * patch];
        im2col2d(x, &mut col, n, (h, w), self, (oh, ow));
        dx.fill(0.0);
        dweight.fill(0.0);
        dbias.fill(0.0);
        accumulate_weight_grad(dy, &col, dbias, dweight, rows, cout, patch);
        let mut dcol = vec![0.0f32; rows * patch];
        compute_dcol(dy, weight, &mut dcol, rows, cout, patch);
        col2im2d(&dcol, dx, (h, w), self, (oh, ow));
        Ok(())
    }

    /// i8 前向（对称量化）：输入与权重已量化，输出 i32 累加，`bias` 为同尺度整数偏置。
    ///
    /// 量化卷积的全部算力都落在 `fast::gemm_i8` 的多槽池上；`bias` 传全 0 可省。
    ///
    /// # Errors
    ///
    /// 缓冲长度与形状不符时返回说明。
    /// 复用外部 `col` 缓冲的 i8 卷积：大形状下省掉每轮 37.7MB 的零页分配与缺页。
    ///
    /// `col` 只需足够长（`rows * patch`）；不足时按需扩容，内容由 im2col 全量覆盖。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_i8_with(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        (h, w): (usize, usize),
        out: &mut [i32],
        col: &mut Vec<i8>,
    ) -> Result<()> {
        let (oh, ow) = self.shape(h, w)?;
        let cout = self.cout;
        let patch = self.geom.checked_patch(self.cin)?;
        check_shapes(&[
            (x.len(), n * h * w * self.cin, "输入"),
            (weight.len(), patch * cout, "权重"),
            (bias.len(), cout, "偏置"),
            (out.len(), n * oh * ow * cout, "输出"),
        ])?;

        let rows = n * oh * ow;
        col.resize(rows * patch, 0);
        im2col2d(x, col, n, (h, w), self, (oh, ow));
        fast::gemm_i8(col, weight, rows, cout, patch, out);
        add_bias(out, bias, cout);
        Ok(())
    }

    /// i8 卷积：`col` 内部现分配，单次调用；批量场景用 [`Self::forward_i8_with`] 复用缓冲。
    pub fn forward_i8(
        &self,
        x: &[i8],
        weight: &[i8],
        bias: &[i32],
        n: usize,
        (h, w): (usize, usize),
        out: &mut [i32],
    ) -> Result<()> {
        with_scratch_col(|col| self.forward_i8_with(x, weight, bias, n, (h, w), out, col))
    }
}

pub mod interp {
    //! 双线性插值采样与坐标重映射。
    use diagnostic::{Code, Error, Result};

    use crate::Scalar;
    use crate::conv::support::*;

    /// 插值取样方式：最近邻或双线性。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Sampler {
        /// 取左上角样本，源坐标 `floor(i·h/oh)`。
        Nearest,
        /// 双线性，源坐标 `(i+0.5)·h/oh − 0.5` 后夹到边界。
        Linear,
    }

    /// 二维上/下采样，NHWC。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Interpolate {
        /// 取样方式。
        pub sampler: Sampler,
        /// 目标高。
        pub oh: usize,
        /// 目标宽。
        pub ow: usize,
    }

    impl Interpolate {
        /// 前向。
        ///
        /// # Errors
        ///
        /// 目标尺寸为 0 或缓冲长度不符时返回说明。
        pub fn forward<S: Scalar>(
            &self,
            x: &[S],
            n: usize,
            h: usize,
            w: usize,
            c: usize,
            out: &mut [S],
        ) -> Result<()> {
            self.ensure_nonzero(h, w)?;
            check_shapes(&[
                (x.len(), n * h * w * c, "输入"),
                (out.len(), n * self.oh * self.ow * c, "输出"),
            ])?;
            match self.sampler {
                Sampler::Nearest => self.forward_nearest(x, n, h, w, c, out),
                Sampler::Linear => self.forward_linear(x, n, h, w, c, out),
            }
            Ok(())
        }

        fn ensure_nonzero(&self, h: usize, w: usize) -> Result<()> {
            if self.oh == 0 || self.ow == 0 {
                return Err(Error::new(Code::Invalid, "插值目标尺寸必须大于 0"));
            }
            if h == 0 || w == 0 {
                return Err(Error::new(Code::Invalid, "插值输入尺寸必须大于 0"));
            }
            Ok(())
        }

        fn forward_nearest<S: Scalar>(
            &self,
            x: &[S],
            n: usize,
            h: usize,
            w: usize,
            c: usize,
            out: &mut [S],
        ) {
            for ni in 0..n {
                for oi in 0..self.oh {
                    let ii = (oi * h / self.oh).min(h - 1);
                    for oj in 0..self.ow {
                        let jj = (oj * w / self.ow).min(w - 1);
                        let dst = ((ni * self.oh + oi) * self.ow + oj) * c;
                        let src = ((ni * h + ii) * w + jj) * c;
                        out[dst..dst + c].copy_from_slice(&x[src..src + c]);
                    }
                }
            }
        }

        fn forward_linear<S: Scalar>(
            &self,
            x: &[S],
            n: usize,
            h: usize,
            w: usize,
            c: usize,
            out: &mut [S],
        ) {
            for ni in 0..n {
                for oi in 0..self.oh {
                    let (y0, y1, ly) = axis_samples::<S>(oi, h, self.oh);
                    for oj in 0..self.ow {
                        let (x0, x1, lx) = axis_samples::<S>(oj, w, self.ow);
                        let dst = ((ni * self.oh + oi) * self.ow + oj) * c;
                        for ci in 0..c {
                            let a = x[((ni * h + y0) * w + x0) * c + ci];
                            let b = x[((ni * h + y0) * w + x1) * c + ci];
                            let d = x[((ni * h + y1) * w + x0) * c + ci];
                            let e = x[((ni * h + y1) * w + x1) * c + ci];
                            let top = a + (b - a) * lx;
                            let bot = d + (e - d) * lx;
                            out[dst + ci] = top + (bot - top) * ly;
                        }
                    }
                }
            }
        }
    }

    fn axis_samples<S: Scalar>(i: usize, len: usize, out_len: usize) -> (usize, usize, S) {
        let f = ((S::from_usize(i) + S::from_f64(0.5)) * S::from_usize(len)
            / S::from_usize(out_len)
            - S::from_f64(0.5))
        .max(S::ZERO);
        let lo = f.floor().min(S::from_usize(len - 1)).to_usize();
        let hi = (lo + 1).min(len - 1);
        let lam = if lo == hi {
            S::ZERO
        } else {
            f - S::from_usize(lo)
        };
        (lo, hi, lam)
    }
}

pub mod pool {
    //! 职责：池化：max/avg 窗口归约与输出体量。
    use diagnostic::{Code, Error, Result};

    use crate::Scalar;

    use crate::conv::support::*;

    use super::ConvGeom;

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

        fn pool_max<S: Scalar>(
            &self,
            x: &[S],
            out: &mut [S],
            n: usize,
            l: usize,
            c: usize,
            ol: usize,
        ) {
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
}
