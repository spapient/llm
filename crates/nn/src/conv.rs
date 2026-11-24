//! 神经算子：卷积 / 反卷积（`geom` / `pool` / `interp` / `deconv`）、注意力（`attention`）、
//! 池化与范数（`classic`）、扫描骨架（`scan`）。

pub mod attention;
pub mod classic;
pub mod deconv;
mod geom;
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
