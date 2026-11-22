//! 职责：`Geometry` 的方法分派——形状核对交给 [`shape`](super::shape)、逐变体数值交给
//! [`kernel`](super::kernel)，本文件只做路由，不重复实现两者。

use diagnostic::Result;
use nn::conv::{AdaptiveAvgPool2d, AvgPool2d, Conv2d, Interpolate, MaxPool1d, MaxPool2d, Sampler};
use nn::kernel::norm::{BatchNorm2d, GroupNorm, LayerNorm};

use super::kernel::*;
use super::shape::*;
use super::{EPS, Geometry, MOMENTUM, Running};

impl Geometry {
    /// 输入个数：池化与插值是一元，三种归一化是 `(x, gamma, beta)`。
    pub fn arity(self) -> usize {
        match self {
            Self::Pool1d { .. }
            | Self::Pool2d { .. }
            | Self::Avg2d { .. }
            | Self::Adaptive { .. }
            | Self::Interp { .. } => 1,
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } | Self::Conv2d { .. } => 3,
        }
    }

    /// 输出元素数。前向缓冲长度必须与它相等。
    pub fn outputs(self) -> usize {
        match self {
            Self::Pool1d { k, s, p, l, c } => pool_len(l, k, s, p).map_or(0, |ol| ol * c),
            Self::Pool2d { .. } | Self::Avg2d { .. } | Self::Conv2d { .. } => spatial_outputs(self),
            Self::Adaptive { oh, ow, c, .. } | Self::Interp { oh, ow, c, .. } => oh * ow * c,
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } => norm_outputs(self),
        }
    }

    /// 校验一次调用的形状：输入个数、各输入长度、输出长度都必须与实例自洽。
    ///
    /// # Errors
    ///
    /// 任一处不符返回 [`Code::Invalid`](diagnostic::Code::Invalid)。
    pub fn check_shape(self, lens: &[usize], out: usize) -> Result<()> {
        if lens.len() != self.arity() {
            return Err(invalid(format!(
                "输入个数是 {}，实例要 {}",
                lens.len(),
                self.arity()
            )));
        }
        match self {
            Self::Pool1d { .. } | Self::Pool2d { .. } | Self::Avg2d { .. } => {
                self.check_pool(lens, out)
            }
            Self::Adaptive { .. } | Self::Interp { .. } => self.check_reshape(lens, out),
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } => {
                self.check_norm_family(lens, out)
            }
            Self::Conv2d { .. } => self.check_conv(lens, out),
        }
    }

    fn check_pool(self, lens: &[usize], out: usize) -> Result<()> {
        match self {
            Self::Pool1d { k, s, p, l, c } => check_pool1d(lens, out, k, s, p, l, c),
            Self::Pool2d { k, s, p, h, w, c } => {
                check_spatial_pool(lens, out, Window { k, s, p }, (h, w, c), "二维池化")
            }
            Self::Avg2d {
                k, s, p, h, w, c, ..
            } => check_spatial_pool(lens, out, Window { k, s, p }, (h, w, c), "二维平均池化"),
            _ => Ok(()),
        }
    }

    fn check_reshape(self, lens: &[usize], out: usize) -> Result<()> {
        match self {
            Self::Adaptive { oh, ow, h, w, c } => {
                check_spatial_reshape(lens, out, (oh, ow), (h, w, c), "自适应池化")
            }
            Self::Interp {
                oh, ow, h, w, c, ..
            } => check_spatial_reshape(lens, out, (oh, ow), (h, w, c), "插值"),
            _ => Ok(()),
        }
    }

    fn check_norm_family(self, lens: &[usize], out: usize) -> Result<()> {
        match self {
            Self::Norm {
                n,
                spatial,
                channels,
            } => check_norm(lens, out, n, spatial, channels),
            Self::Group {
                groups,
                n,
                spatial,
                channels,
            } => check_group(lens, out, groups, n, spatial, channels),
            Self::Layer { rows, width } => check_layer(lens, out, rows, width),
            _ => Ok(()),
        }
    }

    fn check_conv(self, lens: &[usize], out: usize) -> Result<()> {
        let Self::Conv2d {
            k,
            s,
            p,
            h,
            w,
            cin,
            cout,
        } = self
        else {
            return Ok(());
        };
        check_conv2d_shape(lens, out, Window { k, s, p }, (h, w, cin, cout))
    }

    /// 按实例几何前向：算术全部复用 `nn::conv` / `nn::kernel::norm` 的真实现。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、或底层算子报错时返回说明。
    pub fn forward(self, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        self.forward_with(inputs, out, None)
    }

    /// 带滚动统计的前向：只有 [`Geometry::Norm`] 消费 `running`，其余变体忽略。
    ///
    /// 传 `None` 等价于每次前向从"均值 0、方差 1"起算，输出与 [`Geometry::forward`] 逐位一致。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、或底层算子报错时返回说明。
    pub fn forward_with(
        self,
        inputs: &[&[f32]],
        out: &mut [f32],
        running: Option<&mut Running>,
    ) -> Result<()> {
        self.check_shape(
            &inputs.iter().map(|x| x.len()).collect::<Vec<_>>(),
            out.len(),
        )?;
        match self {
            Self::Pool1d { .. } | Self::Pool2d { .. } | Self::Avg2d { .. } => {
                self.forward_pool(inputs, out)
            }
            Self::Adaptive { .. } | Self::Interp { .. } => self.forward_reshape(inputs, out),
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } => {
                self.forward_norm(inputs, out, running)
            }
            Self::Conv2d { .. } => self.forward_conv(inputs, out),
        }
    }

    fn forward_pool(self, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Pool1d { k, s, p, l, c } => {
                let n = inputs[0].len() / (l * c);
                MaxPool1d { k, s, p }.forward(inputs[0], n, l, c, out)
            }
            Self::Pool2d { k, s, p, h, w, c } => {
                let n = inputs[0].len() / (h * w * c);
                MaxPool2d {
                    geom: square(k, s, p),
                }
                .forward(inputs[0], n, h, w, c, out)
            }
            Self::Avg2d {
                k,
                s,
                p,
                h,
                w,
                c,
                count_include_pad,
            } => {
                let n = inputs[0].len() / (h * w * c);
                AvgPool2d {
                    geom: square(k, s, p),
                    count_include_pad,
                }
                .forward(inputs[0], n, h, w, c, out)
            }
            _ => Ok(()),
        }
    }

    fn forward_reshape(self, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Adaptive { oh, ow, h, w, c } => {
                let n = inputs[0].len() / (h * w * c);
                AdaptiveAvgPool2d { oh, ow }.forward(inputs[0], n, h, w, c, out)
            }
            Self::Interp {
                oh,
                ow,
                h,
                w,
                c,
                linear,
            } => {
                let n = inputs[0].len() / (h * w * c);
                let sampler = if linear {
                    Sampler::Linear
                } else {
                    Sampler::Nearest
                };
                Interpolate { sampler, oh, ow }.forward(inputs[0], n, h, w, c, out)
            }
            _ => Ok(()),
        }
    }

    fn forward_norm(
        self,
        inputs: &[&[f32]],
        out: &mut [f32],
        running: Option<&mut Running>,
    ) -> Result<()> {
        match self {
            Self::Norm {
                n,
                spatial,
                channels,
            } => forward_batch_norm(inputs, out, running, n, spatial, channels),
            Self::Group {
                groups,
                n,
                spatial,
                channels,
            } => GroupNorm {
                groups,
                channels,
                eps: EPS,
            }
            .forward(inputs[0], inputs[1], inputs[2], n, spatial, out),
            Self::Layer { width, .. } => {
                LayerNorm { width, eps: EPS }.forward(inputs[0], inputs[1], inputs[2], out)
            }
            _ => Ok(()),
        }
    }

    fn forward_conv(self, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        let Self::Conv2d {
            k,
            s,
            p,
            h,
            w,
            cin,
            cout,
        } = self
        else {
            return Ok(());
        };
        Conv2d::new(cin, cout, square(k, s, p)).forward_f32(
            inputs[0],
            inputs[1],
            inputs[2],
            1,
            (h, w),
            out,
        )
    }

    /// 推理前向：`Norm` 用给定的滚动统计归一化、不改状态；其余变体与训练一致。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、统计通道数不匹配、或底层算子报错时返回说明。
    pub fn forward_eval(self, inputs: &[&[f32]], out: &mut [f32], running: &Running) -> Result<()> {
        let Self::Norm {
            n,
            spatial,
            channels,
        } = self
        else {
            return self.forward(inputs, out);
        };
        self.check_shape(
            &inputs.iter().map(|x| x.len()).collect::<Vec<_>>(),
            out.len(),
        )?;
        if running.mean.len() != channels || running.var.len() != channels {
            return Err(invalid(format!(
                "滚动统计有 {} 个均值 / {} 个方差，实例几何要 {channels} 个通道",
                running.mean.len(),
                running.var.len()
            )));
        }
        BatchNorm2d {
            channels,
            eps: EPS,
            momentum: MOMENTUM,
        }
        .forward_eval(
            inputs[0],
            inputs[1],
            inputs[2],
            n,
            1,
            spatial,
            &running.mean,
            &running.var,
            out,
        )
    }

    /// 按实例几何反向：贡献顺序与输入对齐（一元给 `dx`，三种归一化给 `dx/dgamma/dbeta`）。
    ///
    /// # Errors
    ///
    /// 形状与实例不符时返回 [`Code::Invalid`](diagnostic::Code::Invalid)。
    pub fn backward(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.check_shape(
            &inputs.iter().map(Vec::len).collect::<Vec<_>>(),
            incoming.len(),
        )?;
        match self {
            Self::Pool1d { .. } | Self::Pool2d { .. } | Self::Avg2d { .. } => {
                self.backward_pool(inputs, incoming)
            }
            Self::Adaptive { .. } | Self::Interp { .. } => self.backward_reshape(inputs, incoming),
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } => {
                self.backward_norm_family(inputs, incoming)
            }
            Self::Conv2d { .. } => self.backward_conv(inputs, incoming),
        }
    }

    fn backward_pool(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Pool1d { k, s, p, l, c } => backward_pool1d(inputs, incoming, k, s, p, l, c),
            Self::Pool2d { k, s, p, h, w, c } => {
                backward_pool2d(inputs, incoming, Window { k, s, p }, (h, w, c))
            }
            Self::Avg2d {
                k,
                s,
                p,
                h,
                w,
                c,
                count_include_pad,
            } => backward_avg2d(
                inputs,
                incoming,
                Window { k, s, p },
                (h, w, c),
                count_include_pad,
            ),
            _ => Ok(vec![]),
        }
    }

    fn backward_reshape(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Adaptive { oh, ow, h, w, c } => {
                backward_adaptive(inputs, incoming, oh, ow, h, w, c)
            }
            Self::Interp {
                oh,
                ow,
                h,
                w,
                c,
                linear,
            } => backward_interp(inputs, incoming, (oh, ow), (h, w, c), linear),
            _ => Ok(vec![]),
        }
    }

    fn backward_norm_family(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Norm {
                n,
                spatial,
                channels,
            } => backward_norm(inputs, incoming, n, spatial, channels),
            Self::Group {
                groups,
                n,
                spatial,
                channels,
            } => backward_group(inputs, incoming, groups, n, spatial, channels),
            Self::Layer { rows, width } => backward_layer(inputs, incoming, rows, width),
            _ => Ok(vec![]),
        }
    }

    fn backward_conv(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        let Self::Conv2d {
            k,
            s,
            p,
            h,
            w,
            cin,
            cout,
        } = self
        else {
            return Ok(vec![]);
        };
        backward_conv2d(inputs, incoming, Window { k, s, p }, (h, w, cin, cout))
    }
}
