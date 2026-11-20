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
use op::conv::{AdaptiveAvgPool2d, AvgPool2d, ConvGeom, Interpolate, MaxPool1d, MaxPool2d, Sampler};
use op::norm::{BatchNorm2d, GroupNorm, LayerNorm};

/// 归一化 eps：与 `op::classic` 四个声明档模块的 `EPS` 同值（算术来源留在 op 侧）。
pub const EPS: f32 = op::batch_norm_f32::EPS;

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
            Self::Norm { .. } | Self::Group { .. } | Self::Layer { .. } => 3,
        }
    }

    /// 输出元素数。前向缓冲长度必须与它相等。
    pub fn outputs(self) -> usize {
        match self {
            Self::Pool1d { k, s, p, l, c } => pool_len(l, k, s, p).map_or(0, |ol| ol * c),
            Self::Pool2d { k, s, p, h, w, c } => pool_len(h, k, s, p)
                .and_then(|oh| Ok(oh * pool_len(w, k, s, p)? * c))
                .unwrap_or(0),
            Self::Avg2d { k, s, p, h, w, c, .. } => pool_len(h, k, s, p)
                .and_then(|oh| Ok(oh * pool_len(w, k, s, p)? * c))
                .unwrap_or(0),
            Self::Adaptive { oh, ow, c, .. } => oh * ow * c,
            Self::Interp { oh, ow, c, .. } => oh * ow * c,
            Self::Norm { n, spatial, channels } | Self::Group { n, spatial, channels, .. } => {
                n * spatial * channels
            }
            Self::Layer { rows, width } => rows * width,
        }
    }

    /// 声明档实例：与 `op::classic` 里的固定几何逐字段对齐（梯度对拍的基准）。
    ///
    /// # Errors
    ///
    /// `op` 不在九个经典算子之列时返回 [`Code::Unsupported`]。
    pub fn declared(op: &str) -> Result<Self> {
        match op {
            "max_pool1d_f32" => Ok(Self::Pool1d {
                k: 2,
                s: 2,
                p: 0,
                l: op::max_pool1d_f32::L,
                c: 1,
            }),
            "max_pool2d_f32" => Ok(Self::Pool2d {
                k: 3,
                s: 1,
                p: 1,
                h: op::max_pool2d_f32::H,
                w: op::max_pool2d_f32::W,
                c: 1,
            }),
            "avg_pool2d_f32" => Ok(Self::Avg2d {
                k: 3,
                s: 1,
                p: 1,
                h: op::avg_pool2d_f32::H,
                w: op::avg_pool2d_f32::W,
                c: 1,
                count_include_pad: false,
            }),
            "adaptive_pool_f32" => Ok(Self::Adaptive {
                oh: op::adaptive_pool_f32::OH,
                ow: op::adaptive_pool_f32::OW,
                h: op::adaptive_pool_f32::H,
                w: op::adaptive_pool_f32::W,
                c: 1,
            }),
            "interpolate_f32" => Ok(Self::Interp {
                oh: op::interpolate_f32::OH,
                ow: op::interpolate_f32::OW,
                h: op::interpolate_f32::H,
                w: op::interpolate_f32::W,
                c: 1,
                linear: true,
            }),
            "batch_norm_f32" => Ok(Self::Norm {
                n: op::batch_norm_f32::N,
                spatial: op::batch_norm_f32::S,
                channels: op::batch_norm_f32::C,
            }),
            "group_norm_f32" => Ok(Self::Group {
                groups: op::group_norm_f32::G,
                n: op::group_norm_f32::N,
                spatial: op::group_norm_f32::S,
                channels: op::group_norm_f32::C,
            }),
            // 实例归一化在 op 侧就是 `GroupNorm(groups = channels)`，这里不再单开变体。
            "instance_norm_f32" => Ok(Self::Group {
                groups: op::instance_norm_f32::C,
                n: op::instance_norm_f32::N,
                spatial: op::instance_norm_f32::S,
                channels: op::instance_norm_f32::C,
            }),
            "layer_norm_f32" => Ok(Self::Layer {
                rows: op::layer_norm_f32::R,
                width: op::layer_norm_f32::W,
            }),
            other => Err(Error::new(
                Code::Unsupported,
                format!("`{other}` 没有声明档实例几何"),
            )
            .with("train::Geometry::declared")),
        }
    }

    /// 校验一次调用的形状：输入个数、各输入长度、输出长度都必须与实例自洽。
    ///
    /// # Errors
    ///
    /// 任一处不符返回 [`Code::Invalid`]。
    pub fn expect(self, lens: &[usize], out: usize) -> Result<()> {
        if lens.len() != self.arity() {
            return Err(invalid(format!(
                "输入个数是 {}，实例要 {}",
                lens.len(),
                self.arity()
            )));
        }
        let site = "train::Geometry::expect";
        match self {
            Self::Pool1d { k, s, p, l, c } => {
                let per = l.checked_mul(c).filter(|&v| v > 0).ok_or_else(|| invalid("池化输入为空"))?;
                let n = lens[0] / per;
                let want = n * pool_len(l, k, s, p)? * c;
                check(lens[0] == n * per && out == want, lens, out, "一维池化", site)?;
            }
            Self::Pool2d { k, s, p, h, w, c } => {
                let (oh, ow) = (pool_len(h, k, s, p)?, pool_len(w, k, s, p)?);
                let per = h * w * c;
                let n = lens[0] / per;
                check(
                    lens[0] == n * per && out == n * oh * ow * c,
                    lens,
                    out,
                    "二维池化",
                    site,
                )?;
            }
            Self::Avg2d { k, s, p, h, w, c, .. } => {
                let (oh, ow) = (pool_len(h, k, s, p)?, pool_len(w, k, s, p)?);
                let per = h * w * c;
                let n = lens[0] / per;
                check(
                    lens[0] == n * per && out == n * oh * ow * c,
                    lens,
                    out,
                    "二维平均池化",
                    site,
                )?;
            }
            Self::Adaptive { oh, ow, h, w, c } => {
                let per = h * w * c;
                let n = lens[0] / per;
                check(
                    lens[0] == n * per && out == n * oh * ow * c,
                    lens,
                    out,
                    "自适应池化",
                    site,
                )?;
            }
            Self::Interp { oh, ow, h, w, c, .. } => {
                let per = h * w * c;
                let n = lens[0] / per;
                check(
                    lens[0] == n * per && out == n * oh * ow * c,
                    lens,
                    out,
                    "插值",
                    site,
                )?;
            }
            Self::Norm { n, spatial, channels } => {
                let total = n * spatial * channels;
                check(
                    lens[0] == total && out == total && lens[1] == channels && lens[2] == channels,
                    lens,
                    out,
                    "批归一化",
                    site,
                )?;
            }
            Self::Group { groups, n, spatial, channels } => {
                let total = n * spatial * channels;
                check(
                    groups > 0 && channels % groups == 0 && lens[0] == total && out == total,
                    lens,
                    out,
                    "组归一化",
                    site,
                )?;
                check(
                    lens[1] == channels && lens[2] == channels,
                    lens,
                    out,
                    "组归一化",
                    site,
                )?;
            }
            Self::Layer { rows, width } => {
                let total = rows * width;
                check(
                    lens[0] == total && out == total && lens[1] == width && lens[2] == width,
                    lens,
                    out,
                    "层归一化",
                    site,
                )?;
            }
        }
        Ok(())
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
        self.expect(&inputs.iter().map(|x| x.len()).collect::<Vec<_>>(), out.len())?;
        match self {
            Self::Pool1d { k, s, p, l, c } => {
                let n = inputs[0].len() / (l * c);
                MaxPool1d { k, s, p }.forward(inputs[0], n, l, c, out)
            }
            Self::Pool2d { k, s, p, h, w, c } => {
                let n = inputs[0].len() / (h * w * c);
                MaxPool2d { geom: square(k, s, p) }.forward(inputs[0], n, h, w, c, out)
            }
            Self::Avg2d { k, s, p, h, w, c, count_include_pad } => {
                let n = inputs[0].len() / (h * w * c);
                AvgPool2d { geom: square(k, s, p), count_include_pad }.forward(inputs[0], n, h, w, c, out)
            }
            Self::Adaptive { oh, ow, h, w, c } => {
                let n = inputs[0].len() / (h * w * c);
                AdaptiveAvgPool2d { oh, ow }.forward(inputs[0], n, h, w, c, out)
            }
            Self::Interp { oh, ow, h, w, c, linear } => {
                let n = inputs[0].len() / (h * w * c);
                let sampler = if linear { Sampler::Linear } else { Sampler::Nearest };
                Interpolate { sampler, oh, ow }.forward(inputs[0], n, h, w, c, out)
            }
            Self::Norm { n, spatial, channels } => {
                // 有滚动统计就原地递推（训练语义）；没有就用一次性局部统计，不改外部状态。
                let mut local = (vec![0.0f32; channels], vec![1.0f32; channels]);
                let (mean, var) = match running {
                    Some(stats) if stats.mean.len() == channels && stats.var.len() == channels => {
                        (&mut stats.mean, &mut stats.var)
                    }
                    _ => (&mut local.0, &mut local.1),
                };
                BatchNorm2d { channels, eps: EPS, momentum: MOMENTUM }
                    .forward_train(inputs[0], inputs[1], inputs[2], n, 1, spatial, mean, var, out)
            }
            Self::Group { groups, n, spatial, channels } => GroupNorm { groups, channels, eps: EPS }
                .forward(inputs[0], inputs[1], inputs[2], n, spatial, out),
            Self::Layer { width, .. } => {
                LayerNorm { width, eps: EPS }.forward(inputs[0], inputs[1], inputs[2], out)
            }
        }
    }

    /// 推理前向：`Norm` 用给定的滚动统计归一化、不改状态；其余变体与训练一致。
    ///
    /// # Errors
    ///
    /// 形状与实例不符、统计通道数不匹配、或底层算子报错时返回说明。
    pub fn forward_eval(self, inputs: &[&[f32]], out: &mut [f32], running: &Running) -> Result<()> {
        let Self::Norm { n, spatial, channels } = self else {
            return self.forward(inputs, out);
        };
        self.expect(&inputs.iter().map(|x| x.len()).collect::<Vec<_>>(), out.len())?;
        if running.mean.len() != channels || running.var.len() != channels {
            return Err(invalid(format!(
                "滚动统计有 {} 个均值 / {} 个方差，实例几何要 {channels} 个通道",
                running.mean.len(),
                running.var.len()
            )));
        }
        BatchNorm2d { channels, eps: EPS, momentum: MOMENTUM }.forward_eval(
            inputs[0], inputs[1], inputs[2], n, 1, spatial, &running.mean, &running.var, out,
        )
    }

    /// 按实例几何反向：贡献顺序与输入对齐（一元给 `dx`，三种归一化给 `dx/dgamma/dbeta`）。
    ///
    /// # Errors
    ///
    /// 形状与实例不符时返回 [`Code::Invalid`]。
    pub fn backward(self, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.expect(&inputs.iter().map(Vec::len).collect::<Vec<_>>(), incoming.len())?;
        match self {
            Self::Pool1d { k, s, p, l, c } => {
                let ol = pool_len(l, k, s, p)?;
                let n = inputs[0].len() / (l * c);
                let win = Window { k, s, p };
                let mut dx = vec![0.0f32; inputs[0].len()];
                for ni in 0..n {
                    for oi in 0..ol {
                        for ci in 0..c {
                            let g = incoming[(ni * ol + oi) * c + ci];
                            let pick = window1d(&inputs[0], ni * l * c, l, oi, win, ci);
                            dx[pick] += g;
                        }
                    }
                }
                Ok(vec![dx])
            }
            Self::Pool2d { k, s, p, h, w, c } => {
                let (oh, ow) = (pool_len(h, k, s, p)?, pool_len(w, k, s, p)?);
                let n = inputs[0].len() / (h * w * c);
                let win = Window { k, s, p };
                let mut dx = vec![0.0f32; inputs[0].len()];
                for ni in 0..n {
                    for oi in 0..oh {
                        for oj in 0..ow {
                            for ci in 0..c {
                                let g = incoming[((ni * oh + oi) * ow + oj) * c + ci];
                                let pick = window2d(
                                    &inputs[0],
                                    ni * h * w * c,
                                    (h, w, c),
                                    (oi, oj),
                                    win,
                                    ci,
                                );
                                dx[pick] += g;
                            }
                        }
                    }
                }
                Ok(vec![dx])
            }
            Self::Avg2d { k, s, p, h, w, c, count_include_pad } => {
                let (oh, ow) = (pool_len(h, k, s, p)?, pool_len(w, k, s, p)?);
                let n = inputs[0].len() / (h * w * c);
                let mut dx = vec![0.0f32; inputs[0].len()];
                for ni in 0..n {
                    for oi in 0..oh {
                        for oj in 0..ow {
                            let (lo_i, hi_i) = bounds(oi, s, p, k, h);
                            let (lo_j, hi_j) = bounds(oj, s, p, k, w);
                            let span = if count_include_pad {
                                k * k
                            } else {
                                (hi_i - lo_i + 1) * (hi_j - lo_j + 1)
                            };
                            for ci in 0..c {
                                let share = incoming[((ni * oh + oi) * ow + oj) * c + ci] / span as f32;
                                for ii in lo_i..=hi_i {
                                    for jj in lo_j..=hi_j {
                                        dx[((ni * h + ii) * w + jj) * c + ci] += share;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![dx])
            }
            Self::Adaptive { oh, ow, h, w, c } => {
                let n = inputs[0].len() / (h * w * c);
                let mut dx = vec![0.0f32; inputs[0].len()];
                for ni in 0..n {
                    for oi in 0..oh {
                        let (h0, h1) = (oi * h / oh, ((oi + 1) * h).div_ceil(oh).min(h));
                        for oj in 0..ow {
                            let (w0, w1) = (oj * w / ow, ((oj + 1) * w).div_ceil(ow).min(w));
                            let area = ((h1 - h0) * (w1 - w0)) as f32;
                            for ci in 0..c {
                                let share = incoming[((ni * oh + oi) * ow + oj) * c + ci] / area;
                                for ii in h0..h1 {
                                    for jj in w0..w1 {
                                        dx[((ni * h + ii) * w + jj) * c + ci] += share;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![dx])
            }
            Self::Interp { oh, ow, h, w, c, linear } => {
                let n = inputs[0].len() / (h * w * c);
                let mut dx = vec![0.0f32; inputs[0].len()];
                for ni in 0..n {
                    for oi in 0..oh {
                        for oj in 0..ow {
                            let g = |ci: usize| incoming[((ni * oh + oi) * ow + oj) * c + ci];
                            if linear {
                                let fy = ((oi as f32 + 0.5) * h as f32 / oh as f32 - 0.5).max(0.0);
                                let y0 = fy.floor().min((h - 1) as f32) as usize;
                                let y1 = (y0 + 1).min(h - 1);
                                let ly = if y0 == y1 { 0.0 } else { fy - y0 as f32 };
                                let fx = ((oj as f32 + 0.5) * w as f32 / ow as f32 - 0.5).max(0.0);
                                let x0 = fx.floor().min((w - 1) as f32) as usize;
                                let x1 = (x0 + 1).min(w - 1);
                                let lx = if x0 == x1 { 0.0 } else { fx - x0 as f32 };
                                for ci in 0..c {
                                    let weight = g(ci);
                                    dx[((ni * h + y0) * w + x0) * c + ci] += weight * (1.0 - lx) * (1.0 - ly);
                                    dx[((ni * h + y0) * w + x1) * c + ci] += weight * lx * (1.0 - ly);
                                    dx[((ni * h + y1) * w + x0) * c + ci] += weight * (1.0 - lx) * ly;
                                    dx[((ni * h + y1) * w + x1) * c + ci] += weight * lx * ly;
                                }
                            } else {
                                let ii = (oi * h / oh).min(h - 1);
                                let jj = (oj * w / ow).min(w - 1);
                                for ci in 0..c {
                                    dx[((ni * h + ii) * w + jj) * c + ci] += g(ci);
                                }
                            }
                        }
                    }
                }
                Ok(vec![dx])
            }
            Self::Norm { n, spatial, channels } => {
                let total = n * spatial * channels;
                let per = n * spatial;
                let (mut dx, mut dgamma, mut dbeta) =
                    (vec![0.0f32; total], vec![0.0f32; channels], vec![0.0f32; channels]);
                for c in 0..channels {
                    let members: Vec<usize> = (0..per).map(|t| t * channels + c).collect();
                    norm_group(
                        &inputs[0],
                        &inputs[1],
                        incoming,
                        &members,
                        |idx| idx % channels,
                        &mut dx,
                        &mut dgamma,
                        &mut dbeta,
                    );
                }
                Ok(vec![dx, dgamma, dbeta])
            }
            Self::Group { groups, n, spatial, channels } => {
                let total = n * spatial * channels;
                let per = channels / groups;
                let (mut dx, mut dgamma, mut dbeta) =
                    (vec![0.0f32; total], vec![0.0f32; channels], vec![0.0f32; channels]);
                for bi in 0..n {
                    for g in 0..groups {
                        let c0 = g * per;
                        let members: Vec<usize> = (0..spatial)
                            .flat_map(|s| (c0..c0 + per).map(move |c| (bi * spatial + s) * channels + c))
                            .collect();
                        norm_group(
                            &inputs[0],
                            &inputs[1],
                            incoming,
                            &members,
                            |idx| idx % channels,
                            &mut dx,
                            &mut dgamma,
                            &mut dbeta,
                        );
                    }
                }
                Ok(vec![dx, dgamma, dbeta])
            }
            Self::Layer { rows, width } => {
                let total = rows * width;
                let (mut dx, mut dgamma, mut dbeta) =
                    (vec![0.0f32; total], vec![0.0f32; width], vec![0.0f32; width]);
                for r in 0..rows {
                    let base = r * width;
                    let members: Vec<usize> = (0..width).map(|j| base + j).collect();
                    norm_group(
                        &inputs[0],
                        &inputs[1],
                        incoming,
                        &members,
                        |idx| idx % width,
                        &mut dx,
                        &mut dgamma,
                        &mut dbeta,
                    );
                }
                Ok(vec![dx, dgamma, dbeta])
            }
        }
    }
}

/// 方形窗口的 `ConvGeom`。
fn square(k: usize, s: usize, p: usize) -> ConvGeom {
    ConvGeom { kh: k, kw: k, sh: s, sw: s, ph: p, pw: p }
}

/// 池化输出长度：`(len + 2p − k)/s + 1`，窗口超过输入即拒绝（不静默出 0）。
fn pool_len(len: usize, k: usize, s: usize, p: usize) -> Result<usize> {
    if k == 0 || s == 0 || len + 2 * p < k {
        return Err(invalid(format!("池化几何不成立：len={len}, k={k}, s={s}, p={p}")));
    }
    Ok((len + 2 * p - k) / s + 1)
}

/// 输出位在第 `axis` 维上的有效窗口（闭区间）。
fn bounds(out: usize, s: usize, p: usize, k: usize, len: usize) -> (usize, usize) {
    let lo = (out * s).saturating_sub(p);
    let hi = (out * s + k - 1 - p).min(len - 1);
    (lo.min(len - 1), hi)
}

/// 方形窗口参数：`k/s/p` 三个数一起走，避免每个调用点再摊成三个实参。
#[derive(Clone, Copy)]
struct Window {
    k: usize,
    s: usize,
    p: usize,
}

/// 一维最大池化窗口里的最大值位置（与前向同一判定：严格更大才替换，平手取先遇到的）。
fn window1d(x: &[f32], base: usize, l: usize, oi: usize, win: Window, ci: usize) -> usize {
    let mut best = f32::MIN;
    let mut pick = base + ci;
    for ki in 0..win.k {
        let ii = oi as i64 * win.s as i64 - win.p as i64 + ki as i64;
        if !(0..l as i64).contains(&ii) {
            continue;
        }
        let at = base + ii as usize + ci;
        if x[at] > best {
            best = x[at];
            pick = at;
        }
    }
    pick
}

/// 二维最大池化窗口里的最大值位置；`plane` 是 `(h, w, c)` 三轴，`base` 是本样本起点。
fn window2d(
    x: &[f32],
    base: usize,
    plane: (usize, usize, usize),
    at: (usize, usize),
    win: Window,
    ci: usize,
) -> usize {
    let (h, w, c) = plane;
    let mut best = f32::MIN;
    let mut pick = base + ci;
    for di in 0..win.k {
        let ii = at.0 as i64 * win.s as i64 - win.p as i64 + di as i64;
        if !(0..h as i64).contains(&ii) {
            continue;
        }
        for dj in 0..win.k {
            let jj = at.1 as i64 * win.s as i64 - win.p as i64 + dj as i64;
            if !(0..w as i64).contains(&jj) {
                continue;
            }
            let slot = base + (ii as usize * w + jj as usize) * c + ci;
            if x[slot] > best {
                best = x[slot];
                pick = slot;
            }
        }
    }
    pick
}

/// 归一化反向的公共骨架：对一组元素按 `(均值, 方差)` 求 `dx`，并把 `dgamma`/`dbeta` 累加进
/// 对应通道。`members` 给出该组元素下标，`channel` 把下标映射回通道（层归一化映射回列）。
#[allow(clippy::too_many_arguments)]
fn norm_group(
    x: &[f32],
    gamma: &[f32],
    incoming: &[f32],
    members: &[usize],
    channel: impl Fn(usize) -> usize,
    dx: &mut [f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) {
    let count = members.len() as f32;
    let (mut sum, mut sq) = (0.0f32, 0.0f32);
    for &idx in members {
        sum += x[idx];
        sq += x[idx] * x[idx];
    }
    let mean = sum / count;
    let var = (sq / count - mean * mean).max(0.0);
    let inv = 1.0 / (var + EPS).sqrt();

    let (mut sum_dxhat, mut sum_dxhat_xhat) = (0.0f32, 0.0f32);
    for &idx in members {
        let col = channel(idx);
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[col];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
        // 对 gamma 的偏导只经 xhat 一条路：dxhat 已乘过 gamma，不能再乘第二次。
        dgamma[col] += incoming[idx] * xhat;
        dbeta[col] += incoming[idx];
    }
    for &idx in members {
        let col = channel(idx);
        let xhat = (x[idx] - mean) * inv;
        let dxhat = incoming[idx] * gamma[col];
        dx[idx] = inv * (dxhat - sum_dxhat / count - xhat * sum_dxhat_xhat / count);
    }
}

/// 几何不符的统一报错。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into())
}

/// 形状核对：不符时把实际值带上，避免只报一句"形状不对"。
fn check(ok: bool, lens: &[usize], out: usize, what: &str, site: &str) -> Result<()> {
    if ok {
        return Ok(());
    }
    Err(invalid(format!(
        "{what} 的形状与实例几何不符：输入 {lens:?}、输出 {out} 个元素"
    ))
    .with(site))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 声明档实例与 `op::classic` 的声明档样本长度逐一对齐：前向缓冲与样本同长。
    #[test]
    fn declared_geometries_match_the_sample_lengths() {
        for (op, sample) in [
            ("max_pool1d_f32", op::max_pool1d_f32::sample(1, 1)),
            ("max_pool2d_f32", op::max_pool2d_f32::sample(1, 1)),
            ("avg_pool2d_f32", op::avg_pool2d_f32::sample(1, 1)),
            ("adaptive_pool_f32", op::adaptive_pool_f32::sample(1, 1)),
            ("interpolate_f32", op::interpolate_f32::sample(1, 1)),
            ("batch_norm_f32", op::batch_norm_f32::sample(1, 1)),
            ("group_norm_f32", op::group_norm_f32::sample(1, 1)),
            ("instance_norm_f32", op::instance_norm_f32::sample(1, 1)),
            ("layer_norm_f32", op::layer_norm_f32::sample(1, 1)),
        ] {
            let g = Geometry::declared(op).expect("声明档实例");
            let x = sample.f32("x").expect("x");
            let y = sample.f32("y").expect("y");
            let lens: Vec<usize> = if g.arity() == 1 {
                vec![x.len()]
            } else {
                vec![
                    x.len(),
                    sample.f32("gamma").expect("gamma").len(),
                    sample.f32("beta").expect("beta").len(),
                ]
            };
            g.expect(&lens, y.len()).expect("声明档形状自洽");
        }
    }

    /// 前向：实例几何的结果必须与 `op::classic` 的 reference 逐位一致（同一份算术的两次执行）。
    #[test]
    fn declared_forward_matches_the_reference() {
        for (op, sample) in [
            ("max_pool1d_f32", op::max_pool1d_f32::sample(1, 1)),
            ("max_pool2d_f32", op::max_pool2d_f32::sample(1, 1)),
            ("avg_pool2d_f32", op::avg_pool2d_f32::sample(1, 1)),
            ("adaptive_pool_f32", op::adaptive_pool_f32::sample(1, 1)),
            ("interpolate_f32", op::interpolate_f32::sample(1, 1)),
            ("batch_norm_f32", op::batch_norm_f32::sample(1, 1)),
            ("group_norm_f32", op::group_norm_f32::sample(1, 1)),
            ("instance_norm_f32", op::instance_norm_f32::sample(1, 1)),
            ("layer_norm_f32", op::layer_norm_f32::sample(1, 1)),
        ] {
            let g = Geometry::declared(op).expect("声明档实例");
            let x = sample.f32("x").expect("x");
            let mut out = vec![0.0f32; g.outputs()];
            if g.arity() == 1 {
                g.forward(&[x], &mut out).expect("前向");
            } else {
                g.forward(
                    &[x, sample.f32("gamma").expect("gamma"), sample.f32("beta").expect("beta")],
                    &mut out,
                )
                .expect("前向");
            }
            let want = (op::model(op).expect("已注册").reference)(&sample).expect("参考");
            assert_eq!(out, want, "{op} 实例前向与声明档参考不一致");
        }
    }

    /// 任意几何也要自洽：非声明档的窗口/分组必须能前向，且形状校验拦住不成立的几何。
    /// 滚动统计按动量递推，同时不改变训练前向的输出（训练语义与数值参考同源）。
    /// 推理用滚动统计归一化：`(x - m) / sqrt(v + eps) * gamma + beta`，与手算逐位一致。
    #[test]
    fn evaluation_normalizes_with_the_given_rolling_statistics() {
        let (n, spatial, channels) = (1usize, 2usize, 2usize);
        let len = n * spatial * channels;
        let geom = Geometry::Norm { n, spatial, channels };
        let x: Vec<f32> = (0..len).map(|i| 0.4 + i as f32 * 0.3).collect();
        let gain = vec![1.5f32, 0.5];
        let shift = vec![0.2f32, -0.1];
        let running = Running { mean: vec![0.3, -0.2], var: vec![2.0, 0.5] };

        let mut out = vec![0.0f32; len];
        geom.forward_eval(&[&x, &gain, &shift], &mut out, &running).expect("推理前向");
        for (index, got) in out.iter().enumerate() {
            let ci = index % channels;
            let want = (x[index] - running.mean[ci]) / (running.var[ci] + EPS).sqrt() * gain[ci]
                + shift[ci];
            assert!((got - want).abs() <= 1e-6, "第 {index} 个输出 {got} vs {want}");
        }
        // 通道数不匹配要在形状层报错，不静默套用错位统计。
        let wrong = Running::fresh(channels + 1);
        let error = geom
            .forward_eval(&[&x, &gain, &shift], &mut out, &wrong)
            .expect_err("通道数不符");
        assert_eq!(error.code(), Code::Invalid);
    }

    #[test]
    fn running_statistics_update_by_momentum_without_changing_the_output() {
        let (n, spatial, channels) = (2usize, 3usize, 2usize);
        let len = n * spatial * channels;
        let geom = Geometry::Norm { n, spatial, channels };
        let gain = vec![1.0f32; channels];
        let shift = vec![0.0f32; channels];
        let first: Vec<f32> = (0..len).map(|i| 0.3 + i as f32 * 0.1).collect();
        let second: Vec<f32> = (0..len).map(|i| 1.4 - i as f32 * 0.06).collect();

        let mut stats = Running::fresh(channels);
        for batch in [&first, &second] {
            let mut out = vec![0.0f32; len];
            geom.forward_with(&[batch, &gain, &shift], &mut out, Some(&mut stats))
                .expect("带统计的前向");
            let mut plain = vec![0.0f32; len];
            geom.forward(&[batch, &gain, &shift], &mut plain).expect("无统计的前向");
            for (index, (a, b)) in out.iter().zip(&plain).enumerate() {
                assert!((a - b).abs() <= 1e-6, "第 {index} 个输出被统计影响：{a} vs {b}");
            }
        }

        // 手动复算两批的滑动口径：`m = 0.9 * m + 0.1 * batch_mean`，方差用无偏值。
        let count = n * spatial;
        let mut mean = vec![0.0f32; channels];
        let mut var = vec![1.0f32; channels];
        for batch in [&first, &second] {
            for ci in 0..channels {
                let values: Vec<f32> = (0..count).map(|p| batch[p * channels + ci]).collect();
                let batch_mean = values.iter().sum::<f32>() / count as f32;
                let batch_var = values
                    .iter()
                    .map(|v| (v - batch_mean).powi(2))
                    .sum::<f32>()
                    / count as f32;
                let unbiased = batch_var * count as f32 / (count - 1) as f32;
                mean[ci] = 0.9 * mean[ci] + 0.1 * batch_mean;
                var[ci] = 0.9 * var[ci] + 0.1 * unbiased;
            }
        }
        for ci in 0..channels {
            assert!((stats.mean[ci] - mean[ci]).abs() <= 1e-5, "通道 {ci} 均值 {} vs {}", stats.mean[ci], mean[ci]);
            assert!((stats.var[ci] - var[ci]).abs() <= 1e-5, "通道 {ci} 方差 {} vs {}", stats.var[ci], var[ci]);
        }
    }

    #[test]
    fn arbitrary_geometries_forward_and_reject_bad_shapes() {
        let pool = Geometry::Pool2d { k: 3, s: 2, p: 0, h: 5, w: 5, c: 2 };
        assert_eq!(pool.outputs(), 2 * 2 * 2);
        let mut out = vec![0.0f32; pool.outputs()];
        let x: Vec<f32> = (0..5 * 5 * 2).map(|i| i as f32 - 12.0).collect();
        pool.forward(&[&x], &mut out).expect("前向");
        assert!(out.iter().all(|v| v.is_finite()));

        // 窗口比输入还大：显式拒绝，不出 0 长度。
        let bad = Geometry::Pool1d { k: 5, s: 1, p: 0, l: 3, c: 1 };
        assert!(bad.expect(&[3], bad.outputs()).is_err());
    }
}
