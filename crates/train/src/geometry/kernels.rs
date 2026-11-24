//! 数值微核：各经典算子变体的前向/反向参考实现（梯度对拍基准）。

use super::shapes::{Window, bounds, norm_group, pool_len, pool2d_out, square, window1d, window2d};
use crate::geometry::{EPS, Geometry, MOMENTUM, Running};
use diagnostic::Result;
use nn::conv::{AdaptiveAvgPool2d, AvgPool2d, Conv2d, Interpolate, MaxPool1d, MaxPool2d, Sampler};
use nn::kernel::norm::{BatchNorm2d, GroupNorm, LayerNorm};

// ---- 数值微核：各变体的前向/反向实现 -----------------------------------------

/// 前向总路由：归一化与卷积直接投影配置，五种空间算子进入独立小路由。
pub(crate) fn forward(
    g: Geometry,
    inputs: &[&[f32]],
    out: &mut [f32],
    running: Option<&mut Running>,
) -> Result<()> {
    match g {
        Geometry::Norm {
            n,
            spatial,
            channels,
        } => forward_batch_norm(inputs, out, running, n, spatial, channels),
        Geometry::Group {
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
        Geometry::Layer { width, .. } => {
            LayerNorm { width, eps: EPS }.forward(inputs[0], inputs[1], inputs[2], out)
        }
        Geometry::Conv2d {
            k,
            s,
            p,
            h,
            w,
            cin,
            cout,
        } => Conv2d::new(cin, cout, square(k, s, p)).forward_f32(
            inputs[0],
            inputs[1],
            inputs[2],
            1,
            (h, w),
            out,
        ),
        _ => forward_spatial(g, inputs, out),
    }
}

/// 空间算子二级路由：池化与 reshape 各自只处理自己的闭集。
fn forward_spatial(g: Geometry, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
    match g {
        Geometry::Adaptive { .. } | Geometry::Interp { .. } => forward_reshape(g, inputs, out),
        _ => forward_pool(g, inputs, out),
    }
}

/// 池化前向：由实例配置构造对应 `nn` 算子，批大小只从已校验的输入长度推出。
fn forward_pool(g: Geometry, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
    match g {
        Geometry::Pool1d { k, s, p, l, c } => {
            let n = inputs[0].len() / (l * c);
            MaxPool1d { k, s, p }.forward(inputs[0], n, l, c, out)
        }
        Geometry::Pool2d { k, s, p, h, w, c } => {
            let n = inputs[0].len() / (h * w * c);
            MaxPool2d {
                geom: square(k, s, p),
            }
            .forward(inputs[0], n, h, w, c, out)
        }
        Geometry::Avg2d {
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
        _ => unreachable!("pool 只接收池化几何"),
    }
}

/// 自适应池化与插值前向：二者共享输出平面，采样策略由配置决定。
fn forward_reshape(g: Geometry, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
    match g {
        Geometry::Adaptive { oh, ow, h, w, c } => {
            let n = inputs[0].len() / (h * w * c);
            AdaptiveAvgPool2d { oh, ow }.forward(inputs[0], n, h, w, c, out)
        }
        Geometry::Interp {
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
        _ => unreachable!("reshape 只接收自适应池化或插值几何"),
    }
}

/// 批归一化前向：有滚动统计就原地递推（训练语义），否则用一次性局部统计。
pub(crate) fn forward_batch_norm(
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
pub(crate) fn backward_pool1d(
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
pub(crate) fn backward_pool2d(
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
pub(crate) fn backward_avg2d(
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
pub(crate) fn backward_adaptive(
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
pub(crate) fn backward_interp(
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
        let ly = f32::from(u8::from(y0 != y1)) * (fy - y0 as f32);
        let fx = ((oj as f32 + 0.5) * w as f32 / ow as f32 - 0.5).max(0.0);
        let x0 = fx.floor().min((w - 1) as f32) as usize;
        let x1 = (x0 + 1).min(w - 1);
        let lx = f32::from(u8::from(x0 != x1)) * (fx - x0 as f32);
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
pub(crate) fn backward_norm(
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
pub(crate) fn backward_group(
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
pub(crate) fn backward_layer(
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
pub(crate) fn backward_conv2d(
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
