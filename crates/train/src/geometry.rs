//! 算子实例几何：把窗口、步长、分组、批空间这些**运行时参数**从算子名里拆出来。
//!
//! 三块职责同住本文件：静态规格与声明档表（`EPS`/`MOMENTUM`/[`Running`]/[`Geometry`]）、形状面
//! （核对与窗口几何）、逐变体的前向/反向数值微核，以及把这些路由起来的方法分派。

use diagnostic::{Code, Error, Result};
use nn::conv::{
    AdaptiveAvgPool2d, AvgPool2d, Conv2d, ConvGeom, Interpolate, MaxPool1d, MaxPool2d, Sampler,
};
use nn::kernel::norm::{BatchNorm2d, GroupNorm, LayerNorm};

// ---- 静态规格与声明档表 ------------------------------------------------------

/// 归一化 eps：与 `nn::conv::classic` 四个声明档模块的 `EPS` 同值（算术来源留在 `nn` 侧）。
pub const EPS: f32 = nn::conv::batch_norm::EPS;

/// BatchNorm 的滑动统计动量：只用于更新 running，不参与输出。
pub(crate) const MOMENTUM: f32 = 0.1;

/// BatchNorm 的滚动统计：训练期按 batch 统计与动量递推，推理期直接用这一份。
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
    /// 声明档实例：与 `nn::conv::classic` 里的固定几何逐字段对齐（梯度对拍的基准）。
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
}

/// 十个经典算子的声明档实例：`declared` 的事实来源，一份表、一个数据源。
const DECLARED: &[(&str, Geometry)] = &[
    (
        "max_pool1d_f32",
        Geometry::Pool1d {
            k: 2,
            s: 2,
            p: 0,
            l: nn::conv::max_pool1d::L,
            c: 1,
        },
    ),
    (
        "max_pool2d_f32",
        Geometry::Pool2d {
            k: 3,
            s: 1,
            p: 1,
            h: nn::conv::max_pool2d::H,
            w: nn::conv::max_pool2d::W,
            c: 1,
        },
    ),
    (
        "avg_pool2d_f32",
        Geometry::Avg2d {
            k: 3,
            s: 1,
            p: 1,
            h: nn::conv::avg_pool2d::H,
            w: nn::conv::avg_pool2d::W,
            c: 1,
            count_include_pad: false,
        },
    ),
    (
        "adaptive_pool_f32",
        Geometry::Adaptive {
            oh: nn::conv::adaptive_pool::OH,
            ow: nn::conv::adaptive_pool::OW,
            h: nn::conv::adaptive_pool::H,
            w: nn::conv::adaptive_pool::W,
            c: 1,
        },
    ),
    (
        "interpolate_f32",
        Geometry::Interp {
            oh: nn::conv::interpolate::OH,
            ow: nn::conv::interpolate::OW,
            h: nn::conv::interpolate::H,
            w: nn::conv::interpolate::W,
            c: 1,
            linear: true,
        },
    ),
    (
        "conv2d_f32",
        Geometry::Conv2d {
            k: nn::conv::conv2d::K,
            s: nn::conv::conv2d::S,
            p: nn::conv::conv2d::P,
            h: nn::conv::conv2d::H,
            w: nn::conv::conv2d::W,
            cin: nn::conv::conv2d::CIN,
            cout: nn::conv::conv2d::COUT,
        },
    ),
    (
        "batch_norm_f32",
        Geometry::Norm {
            n: nn::conv::batch_norm::N,
            spatial: nn::conv::batch_norm::S,
            channels: nn::conv::batch_norm::C,
        },
    ),
    (
        "group_norm_f32",
        Geometry::Group {
            groups: nn::conv::group_norm::G,
            n: nn::conv::group_norm::N,
            spatial: nn::conv::group_norm::S,
            channels: nn::conv::group_norm::C,
        },
    ),
    (
        "instance_norm_f32",
        Geometry::Group {
            groups: nn::conv::instance_norm::C,
            n: nn::conv::instance_norm::N,
            spatial: nn::conv::instance_norm::S,
            channels: nn::conv::instance_norm::C,
        },
    ),
    (
        "layer_norm_f32",
        Geometry::Layer {
            rows: nn::conv::layer_norm::R,
            width: nn::conv::layer_norm::W,
        },
    ),
];

// ---- 形状面：核对、窗口几何与共享工具 ----------------------------------------

/// 二维池化/卷积的输出元素数（窗口非法时记 0，与 [`pool_len`] 的拒绝口径一致）。
fn spatial_outputs(g: Geometry) -> usize {
    match g {
        Geometry::Pool2d { k, s, p, h, w, c }
        | Geometry::Avg2d {
            k, s, p, h, w, c, ..
        } => pool_len(h, k, s, p)
            .and_then(|oh| Ok(oh * pool_len(w, k, s, p)? * c))
            .unwrap_or(0),
        Geometry::Conv2d {
            k,
            s,
            p,
            h,
            w,
            cin: _,
            cout,
        } => pool_len(h, k, s, p)
            .and_then(|oh| Ok(oh * pool_len(w, k, s, p)? * cout))
            .unwrap_or(0),
        _ => 0,
    }
}

/// 三种归一化的输出元素数：`n·spatial·channels`（层归一化即 `rows·width`）。
fn norm_outputs(g: Geometry) -> usize {
    match g {
        Geometry::Norm {
            n,
            spatial,
            channels,
        }
        | Geometry::Group {
            n,
            spatial,
            channels,
            ..
        } => n * spatial * channels,
        Geometry::Layer { rows, width } => rows * width,
        _ => 0,
    }
}

/// 一维池化的形状核对：输入 `n·l·c`、输出 `n·ol·c`。
fn check_pool1d(
    lens: &[usize],
    out: usize,
    k: usize,
    s: usize,
    p: usize,
    l: usize,
    c: usize,
) -> Result<()> {
    let per = l
        .checked_mul(c)
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid("池化输入为空"))?;
    let want = lens[0] / per * pool_len(l, k, s, p)? * c;
    ensure_shape(
        lens[0] == lens[0] / per * per && out == want,
        lens,
        out,
        "一维池化",
    )
}

/// 二维池化（最大/平均）的形状核对：输入 `n·h·w·c`、输出 `n·oh·ow·c`。
fn check_spatial_pool(
    lens: &[usize],
    out: usize,
    win: Window,
    plane: (usize, usize, usize),
    what: &str,
) -> Result<()> {
    let (h, w, c) = plane;
    let (oh, ow) = pool2d_out(win, h, w)?;
    let per = h * w * c;
    let n = lens[0] / per;
    ensure_shape(
        lens[0] == n * per && out == n * oh * ow * c,
        lens,
        out,
        what,
    )
}

/// 自适应池化/插值的形状核对：输入 `n·h·w·c`、输出 `n·oh·ow·c`。
fn check_spatial_reshape(
    lens: &[usize],
    out: usize,
    out2: (usize, usize),
    plane: (usize, usize, usize),
    what: &str,
) -> Result<()> {
    let (oh, ow) = out2;
    let (h, w, c) = plane;
    let per = h * w * c;
    let n = lens[0] / per;
    ensure_shape(
        lens[0] == n * per && out == n * oh * ow * c,
        lens,
        out,
        what,
    )
}

/// 批归一化的形状核对：`(n·s·c, c, c)` 的 `(x, gamma, beta)`。
fn check_norm(lens: &[usize], out: usize, n: usize, spatial: usize, channels: usize) -> Result<()> {
    let total = n * spatial * channels;
    ensure_shape(
        lens[0] == total && out == total && lens[1] == channels && lens[2] == channels,
        lens,
        out,
        "批归一化",
    )
}

/// 组归一化（含实例归一化）的形状核对：组数合法、通道被组整除、四长度对齐。
fn check_group(
    lens: &[usize],
    out: usize,
    groups: usize,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<()> {
    let total = n * spatial * channels;
    let channels_ok = [lens[0], lens[1], lens[2], out] == [total, channels, channels, total];
    ensure_shape(
        groups > 0 && channels.is_multiple_of(groups) && channels_ok,
        lens,
        out,
        "组归一化",
    )
}

/// 层归一化的形状核对：`(rows·width, width, width)` 的 `(x, gamma, beta)`。
fn check_layer(lens: &[usize], out: usize, rows: usize, width: usize) -> Result<()> {
    let total = rows * width;
    ensure_shape(
        lens[0] == total && out == total && lens[1] == width && lens[2] == width,
        lens,
        out,
        "层归一化",
    )
}

/// 二维卷积的形状核对：激活/权重/偏置/输出四者长度对齐。
fn check_conv2d_shape(
    lens: &[usize],
    out: usize,
    win: Window,
    shape: (usize, usize, usize, usize),
) -> Result<()> {
    let (h, w, cin, cout) = shape;
    let (oh, ow) = pool2d_out(win, h, w)?;
    let patch = win.k * win.k * cin;
    let io_ok = [lens[0], lens[1], lens[2]] == [h * w * cin, patch * cout, cout];
    ensure_shape(io_ok && out == oh * ow * cout, lens, out, "二维卷积")
}

/// 方形窗口的 `ConvGeom`。
fn square(k: usize, s: usize, p: usize) -> ConvGeom {
    ConvGeom {
        kh: k,
        kw: k,
        sh: s,
        sw: s,
        ph: p,
        pw: p,
    }
}

/// 池化输出长度：`(len + 2p − k)/s + 1`，窗口超过输入即拒绝（不静默出 0）。
fn pool_len(len: usize, k: usize, s: usize, p: usize) -> Result<usize> {
    if k == 0 || s == 0 || len + 2 * p < k {
        return Err(invalid(format!(
            "池化几何不成立：len={len}, k={k}, s={s}, p={p}"
        )));
    }
    Ok((len + 2 * p - k) / s + 1)
}

/// 池化输出边长：`(len + 2p − k)/s + 1`，两轴一起求，供核对与反向复用。
fn pool2d_out(win: Window, h: usize, w: usize) -> Result<(usize, usize)> {
    Ok((
        pool_len(h, win.k, win.s, win.p)?,
        pool_len(w, win.k, win.s, win.p)?,
    ))
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
    for q in 0..win.k * win.k {
        let di = q / win.k;
        let dj = q % win.k;
        let ii = at.0 as i64 * win.s as i64 - win.p as i64 + di as i64;
        if !(0..h as i64).contains(&ii) {
            continue;
        }
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
fn ensure_shape(ok: bool, lens: &[usize], out: usize, what: &str) -> Result<()> {
    if ok {
        return Ok(());
    }
    Err(invalid(format!(
        "{what} 的形状与实例几何不符：输入 {lens:?}、输出 {out} 个元素"
    ))
    .with("train::Geometry::check_shape"))
}

// ---- 数值微核：各变体的前向/反向实现 -----------------------------------------

/// 批归一化前向：有滚动统计就原地递推（训练语义），否则用一次性局部统计。
fn forward_batch_norm(
    inputs: &[&[f32]],
    out: &mut [f32],
    running: Option<&mut Running>,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<()> {
    let mut local = (vec![0.0f32; channels], vec![1.0f32; channels]);
    let (mean, var) = match running {
        Some(stats) if stats.mean.len() == channels && stats.var.len() == channels => {
            (&mut stats.mean, &mut stats.var)
        }
        _ => (&mut local.0, &mut local.1),
    };
    BatchNorm2d {
        channels,
        eps: EPS,
        momentum: MOMENTUM,
    }
    .forward_train(
        inputs[0], inputs[1], inputs[2], n, 1, spatial, mean, var, out,
    )
}

/// 一维最大池化反向：把上游梯度散射回每个窗口的 argmax 位。
fn backward_pool1d(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    k: usize,
    s: usize,
    p: usize,
    l: usize,
    c: usize,
) -> Result<Vec<Vec<f32>>> {
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

/// 池化反向的公共前奏：推输出形状、按样本数分配 `dx`。
fn pool_backward_frame(
    inputs: &[Vec<f32>],
    win: Window,
    plane: (usize, usize, usize),
) -> Result<((usize, usize), usize, Vec<f32>)> {
    let (h, w, c) = plane;
    let n = inputs[0].len() / (h * w * c);
    Ok(((pool2d_out(win, h, w)?), n, vec![0.0f32; inputs[0].len()]))
}

/// 二维最大池化反向：逐样本散射到 argmax 位。
fn backward_pool2d(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    win: Window,
    plane: (usize, usize, usize),
) -> Result<Vec<Vec<f32>>> {
    let ((oh, ow), n, mut dx) = pool_backward_frame(inputs, win, plane)?;
    for ni in 0..n {
        scatter_max2d(&mut dx, &inputs[0], incoming, ni, (oh, ow), plane, win);
    }
    Ok(vec![dx])
}

fn scatter_max2d(
    dx: &mut [f32],
    x: &[f32],
    incoming: &[f32],
    ni: usize,
    out: (usize, usize),
    plane: (usize, usize, usize),
    win: Window,
) {
    let (oh, ow) = out;
    let (h, w, c) = plane;
    for oi in 0..oh {
        for oj in 0..ow {
            for ci in 0..c {
                let g = incoming[((ni * oh + oi) * ow + oj) * c + ci];
                let pick = window2d(x, ni * h * w * c, plane, (oi, oj), win, ci);
                dx[pick] += g;
            }
        }
    }
}

/// 二维平均池化反向：每个输出位均摊回有效窗口，`count_include_pad` 决定分母。
fn backward_avg2d(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    win: Window,
    plane: (usize, usize, usize),
    count_include_pad: bool,
) -> Result<Vec<Vec<f32>>> {
    let ((oh, ow), n, mut dx) = pool_backward_frame(inputs, win, plane)?;
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                scatter_avg2d(
                    &mut dx,
                    incoming,
                    (ni, oi, oj),
                    (oh, ow),
                    plane,
                    win,
                    count_include_pad,
                );
            }
        }
    }
    Ok(vec![dx])
}

fn scatter_avg2d(
    dx: &mut [f32],
    incoming: &[f32],
    at: (usize, usize, usize),
    out: (usize, usize),
    plane: (usize, usize, usize),
    win: Window,
    count_include_pad: bool,
) {
    let (ni, oi, oj) = at;
    let (oh, ow) = out;
    let (h, w, c) = plane;
    let (lo_i, hi_i) = bounds(oi, win.s, win.p, win.k, h);
    let (lo_j, hi_j) = bounds(oj, win.s, win.p, win.k, w);
    let span = if count_include_pad {
        win.k * win.k
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

/// 自适应平均池化反向：按前向划分的 `[h0,h1)×[w0,w1)` 子窗均摊。
fn backward_adaptive(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    oh: usize,
    ow: usize,
    h: usize,
    w: usize,
    c: usize,
) -> Result<Vec<Vec<f32>>> {
    let n = inputs[0].len() / (h * w * c);
    let mut dx = vec![0.0f32; inputs[0].len()];
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                scatter_adaptive(&mut dx, incoming, (ni, oi, oj), (oh, ow), (h, w, c));
            }
        }
    }
    Ok(vec![dx])
}

fn scatter_adaptive(
    dx: &mut [f32],
    incoming: &[f32],
    at: (usize, usize, usize),
    out: (usize, usize),
    plane: (usize, usize, usize),
) {
    let (ni, oi, oj) = at;
    let (oh, ow) = out;
    let (h, w, c) = plane;
    let (h0, h1) = (oi * h / oh, ((oi + 1) * h).div_ceil(oh).min(h));
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

/// 插值反向：双线性按四个角点权重散射，最近邻直接落到源位。
fn backward_interp(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    out: (usize, usize),
    plane: (usize, usize, usize),
    linear: bool,
) -> Result<Vec<Vec<f32>>> {
    let (oh, ow) = out;
    let (h, w, c) = plane;
    let n = inputs[0].len() / (h * w * c);
    let mut dx = vec![0.0f32; inputs[0].len()];
    for ni in 0..n {
        for oi in 0..oh {
            for oj in 0..ow {
                scatter_interp(&mut dx, incoming, (ni, oi, oj), out, plane, linear);
            }
        }
    }
    Ok(vec![dx])
}

/// 反传散射的输入格取值：`(at, out, plane)` 的下标记法只写一处。
fn gain(
    incoming: &[f32],
    at: (usize, usize, usize),
    out: (usize, usize),
    plane: (usize, usize, usize),
) -> impl Fn(usize) -> f32 + '_ {
    let (ni, oi, oj) = at;
    let (oh, ow) = out;
    let c = plane.2;
    move |ci| incoming[((ni * oh + oi) * ow + oj) * c + ci]
}

/// 反传散射：`linear` 走双线性四点分摊，否则按最近邻单点累加。两种口径共享形状分解与
/// 输入格取值，只有采样位置与权重的算法不同。
fn scatter_interp(
    dx: &mut [f32],
    incoming: &[f32],
    at: (usize, usize, usize),
    out: (usize, usize),
    plane: (usize, usize, usize),
    linear: bool,
) {
    let (ni, oi, oj) = at;
    let (oh, ow) = out;
    let (h, w, c) = plane;
    let g = gain(incoming, at, out, plane);
    if linear {
        let fy = ((oi as f32 + 0.5) * h as f32 / oh as f32 - 0.5).max(0.0);
        let y0 = fy.floor().min((h - 1) as f32) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let ly = (y0 != y1) as u8 as f32 * (fy - y0 as f32);
        let fx = ((oj as f32 + 0.5) * w as f32 / ow as f32 - 0.5).max(0.0);
        let x0 = fx.floor().min((w - 1) as f32) as usize;
        let x1 = (x0 + 1).min(w - 1);
        let lx = (x0 != x1) as u8 as f32 * (fx - x0 as f32);
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

/// 批归一化反向：对同通道跨批的整组元素求 `dx/dgamma/dbeta`。
fn backward_norm(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<Vec<Vec<f32>>> {
    let total = n * spatial * channels;
    let per = n * spatial;
    let (mut dx, mut dgamma, mut dbeta) = (
        vec![0.0f32; total],
        vec![0.0f32; channels],
        vec![0.0f32; channels],
    );
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

/// 组归一化反向：每个 `(样本, 组)` 的整组元素分别求导。
fn backward_group(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    groups: usize,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<Vec<Vec<f32>>> {
    let total = n * spatial * channels;
    let per = channels / groups;
    let (mut dx, mut dgamma, mut dbeta) = (
        vec![0.0f32; total],
        vec![0.0f32; channels],
        vec![0.0f32; channels],
    );
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

/// 层归一化反向：逐行求 `dx/dgamma/dbeta`。
fn backward_layer(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<Vec<f32>>> {
    let total = rows * width;
    let (mut dx, mut dgamma, mut dbeta) = (
        vec![0.0f32; total],
        vec![0.0f32; width],
        vec![0.0f32; width],
    );
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

/// 二维卷积反向：复用 `nn::conv::Conv2d::backward_f32`，产出 `dx/dweight/dbias`。
fn backward_conv2d(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    win: Window,
    shape: (usize, usize, usize, usize),
) -> Result<Vec<Vec<f32>>> {
    let (h, w, cin, cout) = shape;
    let conv = Conv2d::new(cin, cout, square(win.k, win.s, win.p));
    let patch = win.k * win.k * cin;
    let (mut dx, mut dweight, mut dbias) = (
        vec![0.0f32; inputs[0].len()],
        vec![0.0f32; patch * cout],
        vec![0.0f32; cout],
    );
    conv.backward_f32(
        &inputs[0],
        &inputs[1],
        incoming,
        1,
        h,
        w,
        &mut dx,
        &mut dweight,
        &mut dbias,
    )?;
    Ok(vec![dx, dweight, dbias])
}

// ---- 方法分派：形状核对 + 逐变体路由 -----------------------------------------
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

#[cfg(test)]
#[path = "../tests/geometry_tests.rs"]
mod tests;
