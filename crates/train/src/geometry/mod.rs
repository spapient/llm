//! 算子实例几何：把窗口、步长、分组、批空间这些**运行时参数**从算子名里拆出来。
//!
//! `op::classic` 的声明档（固定的 `k/s/p`、`4×4 → 2×2`、`C=4, spatial=2, n=2`）在这里是
//! [`Geometry::declared`] 的一个实例：对拍基准仍然是它，但训练侧可以按任意几何前向，反向也
//! 按**同一个实例**回传。前向与反向共用这一个枚举，是"两套几何静默错配"的根治办法——算术只有
//! 一份，声明档只是它的一个取值。
//!
//! 布局沿用 `op::conv` / `op::norm` 的口径：池化与归一化都是 NHWC（通道在最内维），
//! 层归一化是行主序 `rows × width`。

use diagnostic::{Code, Error, Result};
use op::conv::{AdaptiveAvgPool2d, AvgPool2d, Conv2d, Interpolate, MaxPool1d, MaxPool2d, Sampler};
use op::norm::{BatchNorm2d, GroupNorm, LayerNorm};

mod kernel;
use kernel::*;

/// 归一化 eps：与 `op::classic` 四个声明档模块的 `EPS` 同值（算术来源留在 op 侧）。
pub const EPS: f32 = op::batch_norm::EPS;

/// BatchNorm 的滑动统计动量：只用于更新 running，不参与输出。
const MOMENTUM: f32 = 0.1;

/// BatchNorm 的滚动统计：训练期按 batch 统计与动量递推，推理期直接用这一份。
///
/// 更新口径与 `op::norm::BatchNorm2d::forward_train` 一致：方差按无偏值进滑动统计，
/// 归一化本身仍用有偏方差。
#[derive(Debug, Clone, PartialEq)]
pub struct Running {
    /// 每通道滚动均值。
    pub mean: Vec<f32>,
    /// 每通道滚动方差（无偏口径）。
    pub var: Vec<f32>,
}

impl Running {
    /// 初始口径：均值 0、方差 1。
    pub fn fresh(channels: usize) -> Self {
        Self {
            mean: vec![0.0; channels],
            var: vec![1.0; channels],
        }
    }
}

/// 一次算子调用的实例化几何。字段即前向/反向所需的全部形状信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// 一维最大池化：`(n, l, c)` 布局，窗口 `k`、步长 `s`、补零 `p`。
    Pool1d {
        /// 窗口长度。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入长度。
        l: usize,
        /// 通道数。
        c: usize,
    },
    /// 二维最大池化：`(n, h, w, c)` 布局，方形窗口 `k×k`。
    Pool2d {
        /// 窗口边长。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
    },
    /// 二维平均池化；`count_include_pad` 决定补零位是否计入分母。
    Avg2d {
        /// 窗口边长。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
        /// 补零位是否计入分母。
        count_include_pad: bool,
    },
    /// 自适应平均池化：任意 `h×w` 压到 `oh×ow`。
    Adaptive {
        /// 输出高。
        oh: usize,
        /// 输出宽。
        ow: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
    },
    /// 插值：任意 `h×w` 到 `oh×ow`；`linear = false` 走最近邻。
    Interp {
        /// 输出高。
        oh: usize,
        /// 输出宽。
        ow: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 通道数。
        c: usize,
        /// 是否双线性。
        linear: bool,
    },
    /// 批归一化：统计域是"同通道跨批"，共 `n·spatial` 个元素。
    Norm {
        /// 批大小。
        n: usize,
        /// 每样本的空间元素数。
        spatial: usize,
        /// 通道数。
        channels: usize,
    },
    /// 组归一化：统计域是每个 `(样本, 组)` 的 `C/G·spatial` 个元素；
    /// 实例归一化即 `groups = channels`。
    Group {
        /// 组数。
        groups: usize,
        /// 批大小。
        n: usize,
        /// 每样本的空间元素数。
        spatial: usize,
        /// 通道数。
        channels: usize,
    },
    /// 二维卷积：NHWC 激活、`[(ki·k+kj)·cin+ci, cout]` 行主序权重、`same` 式补零（单样本）。
    Conv2d {
        /// 核边长。
        k: usize,
        /// 步长。
        s: usize,
        /// 两侧补零。
        p: usize,
        /// 输入高。
        h: usize,
        /// 输入宽。
        w: usize,
        /// 输入通道。
        cin: usize,
        /// 输出通道。
        cout: usize,
    },
    /// 层归一化：逐行在最后一维统计。
    Layer {
        /// 行数。
        rows: usize,
        /// 行宽。
        width: usize,
    },
}

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

    /// 声明档实例：与 `op::classic` 里的固定几何逐字段对齐（梯度对拍的基准）。
    ///
    /// # Errors
    ///
    /// `op` 不在十个经典算子之列时返回 [`Code::Unsupported`]。
    pub fn declared(op: &str) -> Result<Self> {
        DECLARED
            .iter()
            .find(|(name, _)| *name == op)
            .map(|(_, geom)| *geom)
            .ok_or_else(|| {
                Error::new(Code::Unsupported, format!("`{op}` 没有声明档实例几何"))
                    .with("train::Geometry::declared")
            })
    }

    /// 校验一次调用的形状：输入个数、各输入长度、输出长度都必须与实例自洽。
    ///
    /// # Errors
    ///
    /// 任一处不符返回 [`Code::Invalid`]。
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

    /// 按实例几何前向：算术全部复用 `op::conv` / `op::norm` 的真实现。
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
    /// 形状与实例不符时返回 [`Code::Invalid`]。
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
/// 十个经典算子的声明档实例：`declared` 的事实来源，一份表、一个数据源。
///
/// 值全部取自 `op::*` 的 `pub const`，不在训练侧另写一份算术常量。
const DECLARED: &[(&str, Geometry)] = &[
    (
        "max_pool1d_f32",
        Geometry::Pool1d {
            k: 2,
            s: 2,
            p: 0,
            l: op::max_pool1d::L,
            c: 1,
        },
    ),
    (
        "max_pool2d_f32",
        Geometry::Pool2d {
            k: 3,
            s: 1,
            p: 1,
            h: op::max_pool2d::H,
            w: op::max_pool2d::W,
            c: 1,
        },
    ),
    (
        "avg_pool2d_f32",
        Geometry::Avg2d {
            k: 3,
            s: 1,
            p: 1,
            h: op::avg_pool2d::H,
            w: op::avg_pool2d::W,
            c: 1,
            count_include_pad: false,
        },
    ),
    (
        "adaptive_pool_f32",
        Geometry::Adaptive {
            oh: op::adaptive_pool::OH,
            ow: op::adaptive_pool::OW,
            h: op::adaptive_pool::H,
            w: op::adaptive_pool::W,
            c: 1,
        },
    ),
    (
        "interpolate_f32",
        Geometry::Interp {
            oh: op::interpolate::OH,
            ow: op::interpolate::OW,
            h: op::interpolate::H,
            w: op::interpolate::W,
            c: 1,
            linear: true,
        },
    ),
    (
        "conv2d_f32",
        Geometry::Conv2d {
            k: op::conv2d::K,
            s: op::conv2d::S,
            p: op::conv2d::P,
            h: op::conv2d::H,
            w: op::conv2d::W,
            cin: op::conv2d::CIN,
            cout: op::conv2d::COUT,
        },
    ),
    (
        "batch_norm_f32",
        Geometry::Norm {
            n: op::batch_norm::N,
            spatial: op::batch_norm::S,
            channels: op::batch_norm::C,
        },
    ),
    (
        "group_norm_f32",
        Geometry::Group {
            groups: op::group_norm::G,
            n: op::group_norm::N,
            spatial: op::group_norm::S,
            channels: op::group_norm::C,
        },
    ),
    (
        "instance_norm_f32",
        Geometry::Group {
            groups: op::instance_norm::C,
            n: op::instance_norm::N,
            spatial: op::instance_norm::S,
            channels: op::instance_norm::C,
        },
    ),
    (
        "layer_norm_f32",
        Geometry::Layer {
            rows: op::layer_norm::R,
            width: op::layer_norm::W,
        },
    ),
];
#[cfg(test)]
#[path = "../../tests/geometry_tests.rs"]
mod tests;
