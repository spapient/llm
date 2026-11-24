//! 神经算子：卷积 / 反卷积（`geom` / `pool` / `interp` / `deconv`）、注意力（`attention`）、
//! 池化与范数（`classic`）、扫描骨架（`scan`）。

pub mod attention;
pub mod classic;
pub mod deconv;
mod geom {
    //! 职责：卷积几何：输入输出体量与切片步幅计算。
    use diagnostic::{Code, Error, Result};

    use crate::kernel::fast;

    use super::support::*;

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
            self.out_len(h + 2 * self.ph, self.kh, self.sh, "h")
        }

        /// 输出宽；口径同 [`ConvGeom::oh`]。
        pub fn ow(&self, w: usize) -> Result<usize> {
            self.out_len(w + 2 * self.pw, self.kw, self.sw, "w")
        }

        fn out_len(&self, padded: usize, k: usize, s: usize, axis: &str) -> Result<usize> {
            if s == 0 || k == 0 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{axis}: 核与步长必须非零"),
                ));
            }
            if padded < k {
                return Err(Error::new(
                    Code::Invalid,
                    format!("{axis}: 补零后 {padded} 小于核 {k}，没有合法输出"),
                ));
            }
            Ok((padded - k) / s + 1)
        }

        /// 一行 im2col 的长度：`kh·kw·cin`。
        pub fn patch(&self, cin: usize) -> usize {
            self.kh * self.kw * cin
        }
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
            let patch = self.geom.patch(cin);
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
            let patch = self.geom.patch(cin);
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
            let patch = self.geom.patch(cin);
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
            let patch = self.geom.patch(self.cin);
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
}
mod interp {
    //! 双线性插值采样与坐标重映射。
    use diagnostic::{Code, Error, Result};

    use super::support::*;
    use crate::Scalar;

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
mod pool;
pub mod scan;
mod support;

pub use attention::decode_attention;
pub use classic::{
    adaptive_pool, avg_pool2d, batch_norm, conv2d, group_norm, instance_norm, interpolate,
    layer_norm, max_pool1d, max_pool2d,
};
pub use geom::{Conv1d, Conv2d, ConvGeom};
pub use interp::{Interpolate, Sampler};
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d};
pub use scan::f32 as scan_f32;

#[cfg(test)]
#[path = "../tests/conv_tests.rs"]
mod conv_tests;

#[cfg(test)]
#[path = "../tests/conv_prop_tests.rs"]
mod conv_prop_tests;
