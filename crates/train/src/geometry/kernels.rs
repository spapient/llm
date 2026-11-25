//! 数值微核：各经典算子变体的前向/反向参考实现（梯度对拍基准）。

use super::shapes::{
    Window, bounds, internal, norm_group, pool_len, pool2d_out, square, window1d, window2d,
};
use crate::geometry::{EPS, Geometry, Geometry::*, MOMENTUM, Running};
use diagnostic::Result;
use nn::conv::{
    AdaptiveAvgPool2d, AvgPool2d, Conv2d as Conv2dKernel, Interpolate, MaxPool1d, MaxPool2d,
    Sampler,
};
use nn::norm::{BatchNorm2d, GroupNorm, LayerNorm};

/// 几何族路由只负责闭集分派，数值和错误路径仍由各微核显式持有。
macro_rules! route {
    ($name:ident($g:ident, $($arg:ident : $ty:ty),*) -> $ret:ty; $message:literal;
     $($pattern:pat => $call:expr),+ $(,)?) => {
        fn $name($g: Geometry, $($arg: $ty),*) -> $ret {
            match $g { $($pattern => $call),+, _ => Err(internal($message)) }
        }
    };
}

/// 前向总路由：归一化、卷积与空间算子各自只有一个入口。
pub(crate) fn forward(
    g: Geometry,
    inputs: &[&[f32]],
    out: &mut [f32],
    running: Option<&mut Running>,
) -> Result<()> {
    match g {
        Norm { .. } | Group { .. } | Layer { .. } => forward_normalized(g, inputs, out, running),
        Geometry::Conv2d { k, s, p, h, w, cin, cout } => {
            forward_conv2d(inputs, out, Window { k, s, p }, (h, w, cin, cout))
        }
        _ => forward_spatial(g, inputs, out),
    }
}

/// 归一化前向路由；滚动统计只被批归一化消费。
fn forward_normalized(
    g: Geometry,
    inputs: &[&[f32]],
    out: &mut [f32],
    running: Option<&mut Running>,
) -> Result<()> {
    match g {
        Norm { n, spatial, channels } => {
            forward_batch_norm(inputs, out, running, n, spatial, channels)
        }
        Group { groups, n, spatial, channels } => GroupNorm {
            groups,
            channels,
            eps: EPS,
        }
        .forward(inputs[0], inputs[1], inputs[2], n, spatial, out),
        Layer { width, .. } => {
            LayerNorm { width, eps: EPS }.forward(inputs[0], inputs[1], inputs[2], out)
        }
        _ => Err(internal("归一化前向收到非归一化几何")),
    }
}

/// 二维卷积前向数值核；窗口与通道几何由路由一次性投影。
fn forward_conv2d(
    inputs: &[&[f32]],
    out: &mut [f32],
    win: Window,
    (h, w, cin, cout): (usize, usize, usize, usize),
) -> Result<()> {
    Conv2dKernel::new(cin, cout, square(win.k, win.s, win.p)).forward_f32(
        inputs[0],
        inputs[1],
        inputs[2],
        1,
        (h, w),
        out,
    )
}

route!(forward_spatial(g, inputs: &[&[f32]], out: &mut [f32]) -> Result<()>;
    "空间前向收到非空间几何";
    Adaptive { .. } | Interp { .. } => forward_reshape(g, inputs, out),
    Pool1d { .. } | Pool2d { .. } | Avg2d { .. } => forward_pool(g, inputs, out)
);

route!(forward_pool(g, inputs: &[&[f32]], out: &mut [f32]) -> Result<()>;
    "池化前向只接收池化几何";
    Pool1d { k, s, p, l, c } => MaxPool1d { k, s, p }
        .forward(inputs[0], inputs[0].len() / (l * c), l, c, out),
    Pool2d { k, s, p, h, w, c } => MaxPool2d { geom: square(k, s, p) }
        .forward(inputs[0], inputs[0].len() / (h * w * c), h, w, c, out),
    Avg2d { k, s, p, h, w, c, count_include_pad } => AvgPool2d {
        geom: square(k, s, p), count_include_pad,
    }.forward(inputs[0], inputs[0].len() / (h * w * c), h, w, c, out)
);

route!(forward_reshape(g, inputs: &[&[f32]], out: &mut [f32]) -> Result<()>;
    "reshape 前向只接收自适应池化或插值几何";
    Adaptive { oh, ow, h, w, c } => AdaptiveAvgPool2d { oh, ow }
        .forward(inputs[0], inputs[0].len() / (h * w * c), h, w, c, out),
    Interp { oh, ow, h, w, c, linear } => Interpolate {
        sampler: if linear { Sampler::Linear } else { Sampler::Nearest }, oh, ow,
    }.forward(inputs[0], inputs[0].len() / (h * w * c), h, w, c, out)
);

/// 选择滚动统计或局部统计；非法滚动维度不能静默降级。
fn running_or_local<'a>(
    running: Option<&'a mut Running>,
    local: &'a mut Running,
    channels: usize,
) -> Result<&'a mut Running> {
    if let Some(stats) = running.as_deref() {
        super::check_running(stats, channels)?;
    }
    Ok(running.unwrap_or(local))
}

/// 批归一化前向：有滚动统计就原地递推，否则用一次性局部统计。
fn forward_batch_norm(
    inputs: &[&[f32]],
    out: &mut [f32],
    running: Option<&mut Running>,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<()> {
    let mut local = Running::fresh(channels);
    let stats = running_or_local(running, &mut local, channels)?;
    BatchNorm2d {
        channels,
        eps: EPS,
        momentum: MOMENTUM,
    }
    .forward_train(
        inputs[0],
        inputs[1],
        inputs[2],
        n,
        1,
        spatial,
        &mut stats.mean,
        &mut stats.var,
        out,
    )
}

/// 批归一化推理前向：只读取滚动统计，不修改训练状态。
pub(crate) fn forward_batch_norm_eval(
    inputs: &[&[f32]],
    out: &mut [f32],
    running: &Running,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<()> {
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

type At = (usize, usize, usize);
type Out = (usize, usize);
type Plane = (usize, usize, usize);
type Grid = (Out, Plane);
type Lane = (At, usize, f32);
type Rect = (usize, usize, usize, usize);

/// 所有空间反向共用输出位置与通道遍历，微核只接收一个通道的上游梯度。
fn backward_grid(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    ((oh, ow), (h, w, c)): Grid,
    mut scatter: impl FnMut(&mut [f32], At, usize, f32),
) -> Result<Vec<Vec<f32>>> {
    let n = inputs[0].len() / (h * w * c);
    let mut dx = vec![0.0f32; inputs[0].len()];
    for pos in 0..n * oh * ow {
        let at = (pos / (oh * ow), (pos / ow) % oh, pos % ow);
        for ci in 0..c {
            let grad = incoming[(pos * c) + ci];
            scatter(&mut dx, at, ci, grad);
        }
    }
    Ok(vec![dx])
}

/// 最大池化只向窗口 argmax 的位置散射；平手规则由窗口函数统一保证。
fn scatter_max(
    dx: &mut [f32],
    x: &[f32],
    ((ni, oi, oj), ci, grad): Lane,
    ((_, _), (h, w, c)): Grid,
    win: Window,
    pick: impl Fn(&[f32], usize, (usize, usize), Window, usize) -> usize,
) {
    let at = pick(x, ni * h * w * c, (oi, oj), win, ci);
    dx[at] += grad;
}

/// 平均类散射共用一个有效矩形和分母，调用方提供当前通道的梯度。
fn scatter_area(
    dx: &mut [f32],
    ((ni, _, _), ci, grad): Lane,
    ((_, _), (h, w, c)): Grid,
    (lo_i, hi_i, lo_j, hi_j): Rect,
    span: usize,
) {
    let share = grad / span as f32;
    for ii in lo_i..=hi_i {
        for jj in lo_j..=hi_j {
            dx[((ni * h + ii) * w + jj) * c + ci] += share;
        }
    }
}

/// 插值反向：双线性按四个角点权重散射，最近邻直接落到源位。
fn scatter_interp(dx: &mut [f32], lane: Lane, grid: Grid, linear: bool) {
    let ((ni, oi, oj), ci, grad) = lane;
    let ((oh, ow), (h, w, c)) = grid;
    if !linear {
        let ii = (oi * h / oh).min(h - 1);
        let jj = (oj * w / ow).min(w - 1);
        scatter_area(dx, lane, grid, (ii, ii, jj, jj), 1);
        return;
    }
    let fy = ((oi as f32 + 0.5) * h as f32 / oh as f32 - 0.5).max(0.0);
    let y0 = fy.floor().min((h - 1) as f32) as usize;
    let y1 = (y0 + 1).min(h - 1);
    let ly = f32::from(u8::from(y0 != y1)) * (fy - y0 as f32);
    let fx = ((oj as f32 + 0.5) * w as f32 / ow as f32 - 0.5).max(0.0);
    let x0 = fx.floor().min((w - 1) as f32) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let lx = f32::from(u8::from(x0 != x1)) * (fx - x0 as f32);
    for (ii, jj, a, b) in [
        (y0, x0, 1.0 - lx, 1.0 - ly),
        (y0, x1, lx, 1.0 - ly),
        (y1, x0, 1.0 - lx, ly),
        (y1, x1, lx, ly),
    ] {
        dx[((ni * h + ii) * w + jj) * c + ci] += grad * a * b;
    }
}

/// 二维平均池化反向：每个输出位均摊回有效窗口。
fn backward_avg2d(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    win: Window,
    plane: Plane,
    include_pad: bool,
) -> Result<Vec<Vec<f32>>> {
    let (h, w, _) = plane;
    let grid = (pool2d_out(win, h, w)?, plane);
    backward_grid(inputs, incoming, grid, |dx, (ni, oi, oj), ci, grad| {
        let (lo_i, hi_i) = bounds(oi, win.s, win.p, win.k, h);
        let (lo_j, hi_j) = bounds(oj, win.s, win.p, win.k, w);
        let span = if include_pad {
            win.k * win.k
        } else {
            (hi_i - lo_i + 1) * (hi_j - lo_j + 1)
        };
        scatter_area(
            dx,
            ((ni, oi, oj), ci, grad),
            grid,
            (lo_i, hi_i, lo_j, hi_j),
            span,
        );
    })
}

/// 自适应平均池化反向：按前向划分的子窗均摊。
fn backward_adaptive(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    out: Out,
    plane: Plane,
) -> Result<Vec<Vec<f32>>> {
    let (h, w, _) = plane;
    let grid = (out, plane);
    backward_grid(inputs, incoming, grid, |dx, (ni, oi, oj), ci, grad| {
        let rect = (
            oi * h / out.0,
            ((oi + 1) * h).div_ceil(out.0).min(h) - 1,
            oj * w / out.1,
            ((oj + 1) * w).div_ceil(out.1).min(w) - 1,
        );
        let span = (rect.1 - rect.0 + 1) * (rect.3 - rect.2 + 1);
        scatter_area(dx, ((ni, oi, oj), ci, grad), grid, rect, span);
    })
}

fn backward_interp(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    out: Out,
    plane: Plane,
    linear: bool,
) -> Result<Vec<Vec<f32>>> {
    let grid = (out, plane);
    backward_grid(inputs, incoming, grid, |dx, at, ci, grad| {
        scatter_interp(dx, (at, ci, grad), grid, linear);
    })
}

/// 三个梯度输出按配置长度一次分配，归一化与卷积共用同一缓冲入口。
fn zero_grads<const N: usize>(lengths: [usize; N]) -> [Vec<f32>; N] {
    lengths.map(|len| vec![0.0; len])
}

/// 归一化反向的泛型骨架：布局给出组数与组宽，闭包给出成员下标。
fn normalized_backward(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    channels: usize,
    (groups, per): (usize, usize),
    idx: impl Fn(usize, usize) -> usize,
) -> Result<Vec<Vec<f32>>> {
    let [mut dx, mut dgamma, mut dbeta] = zero_grads([inputs[0].len(), channels, channels]);
    for group in 0..groups {
        let members = (0..per).map(|at| idx(group, at));
        norm_group(
            &inputs[0],
            &inputs[1],
            incoming,
            members,
            |idx| idx % channels,
            (&mut dx, &mut dgamma, &mut dbeta),
        );
    }
    Ok(vec![dx, dgamma, dbeta])
}

/// 二维卷积反向：产出 `dx/dweight/dbias`。
fn backward_conv2d(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    win: Window,
    (h, w, cin, cout): (usize, usize, usize, usize),
) -> Result<Vec<Vec<f32>>> {
    let conv = Conv2dKernel::new(cin, cout, square(win.k, win.s, win.p));
    let patch = win.k * win.k * cin;
    let [mut dx, mut dweight, mut dbias] = zero_grads([inputs[0].len(), patch * cout, cout]);
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

/// 反向总路由：批归一化与层归一化直接生成成员，其余几何进入二级路由。
pub(crate) fn backward(
    g: Geometry,
    inputs: &[Vec<f32>],
    incoming: &[f32],
) -> Result<Vec<Vec<f32>>> {
    match g {
        Norm {
            n,
            spatial,
            channels,
        } => normalized_backward(
            inputs,
            incoming,
            channels,
            (channels, n * spatial),
            |channel, at| at * channels + channel,
        ),
        Layer { rows, width } => {
            normalized_backward(inputs, incoming, width, (rows, width), |row, col| {
                row * width + col
            })
        }
        _ => backward_other(g, inputs, incoming),
    }
}

/// 组归一化反向：组成员布局只在这里解释，数值复用统一骨架。
fn backward_group(
    inputs: &[Vec<f32>],
    incoming: &[f32],
    groups: usize,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<Vec<Vec<f32>>> {
    let per = channels / groups;
    normalized_backward(
        inputs,
        incoming,
        channels,
        (n * groups, spatial * per),
        |member, at| {
            let (bi, group) = (member / groups, member % groups);
            let (s, channel) = (at / per, at % per);
            (bi * spatial + s) * channels + group * per + channel
        },
    )
}

fn backward_other(g: Geometry, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>> {
    match g {
        Group {
            groups,
            n,
            spatial,
            channels,
        } => backward_group(inputs, incoming, groups, n, spatial, channels),
        Geometry::Conv2d {
            k,
            s,
            p,
            h,
            w,
            cin,
            cout,
        } => backward_conv2d(inputs, incoming, Window { k, s, p }, (h, w, cin, cout)),
        _ => backward_spatial(g, inputs, incoming),
    }
}

route!(backward_spatial(g, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>>;
    "空间反向收到非空间几何";
    Adaptive { oh, ow, h, w, c } => backward_adaptive(inputs, incoming, (oh, ow), (h, w, c)),
    Interp { oh, ow, h, w, c, linear } => backward_interp(inputs, incoming, (oh, ow), (h, w, c), linear),
    Pool1d { .. } | Pool2d { .. } | Avg2d { .. } => backward_pool(g, inputs, incoming)
);

route!(backward_pool(g, inputs: &[Vec<f32>], incoming: &[f32]) -> Result<Vec<Vec<f32>>>;
    "池化反向只接收池化几何";
    Pool1d { k, s, p, l, c } => {
        let plane = (l, 1, c);
        let grid = ((pool_len(l, k, s, p)?, 1), plane);
        let win = Window { k, s, p };
        backward_grid(inputs, incoming, grid, |dx, at, ci, grad| {
            scatter_max(dx, &inputs[0], (at, ci, grad), grid, win,
                |x, base, at, win, ci| window1d(x, base, l, at.0, win, ci));
        })
    },
    Pool2d { k, s, p, h, w, c } => {
        let plane = (h, w, c);
        let grid = (pool2d_out(Window { k, s, p }, h, w)?, plane);
        let win = Window { k, s, p };
        backward_grid(inputs, incoming, grid, |dx, at, ci, grad| {
            scatter_max(dx, &inputs[0], (at, ci, grad), grid, win,
                |x, base, at, win, ci| window2d(x, base, plane, at, win, ci));
        })
    },
    Avg2d { k, s, p, h, w, c, count_include_pad } => backward_avg2d(
        inputs, incoming, Window { k, s, p }, (h, w, c), count_include_pad)
);
