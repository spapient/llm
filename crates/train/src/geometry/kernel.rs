//! 职责：几何微核——`Geometry` 各变体的形状核对与逐变体前向/反向实现。
//!
//! 这些自由函数只被同模块的 [`super::Geometry`] 方法调用，不对外暴露；每个变体的算术
//! 各自成函数，把 [`Geometry`] 方法里的大 match 拆成圈复杂度 ≤5、可独立阅读的片段。

use diagnostic::{Code, Error, Result};
use op::conv::{Conv2d, ConvGeom};
use op::norm::BatchNorm2d;

use super::{EPS, Geometry, MOMENTUM, Running};

/// 二维池化/卷积的输出元素数（窗口非法时记 0，与 [`pool_len`] 的拒绝口径一致）。
pub(super) fn spatial_outputs(g: Geometry) -> usize {
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
pub(super) fn norm_outputs(g: Geometry) -> usize {
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
pub(super) fn check_pool1d(
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
    check(
        lens[0] == lens[0] / per * per && out == want,
        lens,
        out,
        "一维池化",
    )
}

/// 二维池化（最大/平均）的形状核对：输入 `n·h·w·c`、输出 `n·oh·ow·c`。
pub(super) fn check_spatial_pool(
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
    check(
        lens[0] == n * per && out == n * oh * ow * c,
        lens,
        out,
        what,
    )
}

/// 自适应池化/插值的形状核对：输入 `n·h·w·c`、输出 `n·oh·ow·c`。
pub(super) fn check_spatial_reshape(
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
    check(
        lens[0] == n * per && out == n * oh * ow * c,
        lens,
        out,
        what,
    )
}

/// 批归一化的形状核对：`(n·s·c, c, c)` 的 `(x, gamma, beta)`。
pub(super) fn check_norm(
    lens: &[usize],
    out: usize,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<()> {
    let total = n * spatial * channels;
    check(
        lens[0] == total && out == total && lens[1] == channels && lens[2] == channels,
        lens,
        out,
        "批归一化",
    )
}

/// 组归一化（含实例归一化）的形状核对：组数合法、通道被组整除、四长度对齐。
pub(super) fn check_group(
    lens: &[usize],
    out: usize,
    groups: usize,
    n: usize,
    spatial: usize,
    channels: usize,
) -> Result<()> {
    let total = n * spatial * channels;
    let channels_ok = [lens[0], lens[1], lens[2], out] == [total, channels, channels, total];
    check(
        groups > 0 && channels.is_multiple_of(groups) && channels_ok,
        lens,
        out,
        "组归一化",
    )
}

/// 层归一化的形状核对：`(rows·width, width, width)` 的 `(x, gamma, beta)`。
pub(super) fn check_layer(lens: &[usize], out: usize, rows: usize, width: usize) -> Result<()> {
    let total = rows * width;
    check(
        lens[0] == total && out == total && lens[1] == width && lens[2] == width,
        lens,
        out,
        "层归一化",
    )
}

/// 二维卷积的形状核对：激活/权重/偏置/输出四者长度对齐。
pub(super) fn check_conv2d_shape(
    lens: &[usize],
    out: usize,
    win: Window,
    shape: (usize, usize, usize, usize),
) -> Result<()> {
    let (h, w, cin, cout) = shape;
    let (oh, ow) = pool2d_out(win, h, w)?;
    let patch = win.k * win.k * cin;
    let io_ok = [lens[0], lens[1], lens[2]] == [h * w * cin, patch * cout, cout];
    check(io_ok && out == oh * ow * cout, lens, out, "二维卷积")
}

/// 批归一化前向：有滚动统计就原地递推（训练语义），否则用一次性局部统计。
pub(super) fn forward_batch_norm(
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
pub(super) fn backward_pool1d(
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

/// 池化输出边长：`(len + 2p − k)/s + 1`，两轴一起求，供核对与反向复用。
pub(super) fn pool2d_out(win: Window, h: usize, w: usize) -> Result<(usize, usize)> {
    Ok((
        pool_len(h, win.k, win.s, win.p)?,
        pool_len(w, win.k, win.s, win.p)?,
    ))
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
pub(super) fn backward_pool2d(
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

pub(super) fn scatter_max2d(
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
pub(super) fn backward_avg2d(
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

pub(super) fn scatter_avg2d(
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
pub(super) fn backward_adaptive(
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

pub(super) fn scatter_adaptive(
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
pub(super) fn backward_interp(
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
pub(super) fn scatter_interp(
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
pub(super) fn backward_norm(
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
pub(super) fn backward_group(
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
pub(super) fn backward_layer(
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

/// 二维卷积反向：复用 `op::conv::Conv2d::backward_f32`，产出 `dx/dweight/dbias`。
pub(super) fn backward_conv2d(
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

/// 方形窗口的 `ConvGeom`。
pub(super) fn square(k: usize, s: usize, p: usize) -> ConvGeom {
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
pub(super) fn pool_len(len: usize, k: usize, s: usize, p: usize) -> Result<usize> {
    if k == 0 || s == 0 || len + 2 * p < k {
        return Err(invalid(format!(
            "池化几何不成立：len={len}, k={k}, s={s}, p={p}"
        )));
    }
    Ok((len + 2 * p - k) / s + 1)
}

/// 输出位在第 `axis` 维上的有效窗口（闭区间）。
pub(super) fn bounds(out: usize, s: usize, p: usize, k: usize, len: usize) -> (usize, usize) {
    let lo = (out * s).saturating_sub(p);
    let hi = (out * s + k - 1 - p).min(len - 1);
    (lo.min(len - 1), hi)
}

/// 方形窗口参数：`k/s/p` 三个数一起走，避免每个调用点再摊成三个实参。
#[derive(Clone, Copy)]
pub(super) struct Window {
    pub(super) k: usize,
    pub(super) s: usize,
    pub(super) p: usize,
}

/// 一维最大池化窗口里的最大值位置（与前向同一判定：严格更大才替换，平手取先遇到的）。
pub(super) fn window1d(
    x: &[f32],
    base: usize,
    l: usize,
    oi: usize,
    win: Window,
    ci: usize,
) -> usize {
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
pub(super) fn window2d(
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
pub(super) fn norm_group(
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
pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into())
}

/// 形状核对：不符时把实际值带上，避免只报一句"形状不对"。
pub(super) fn check(ok: bool, lens: &[usize], out: usize, what: &str) -> Result<()> {
    if ok {
        return Ok(());
    }
    Err(invalid(format!(
        "{what} 的形状与实例几何不符：输入 {lens:?}、输出 {out} 个元素"
    ))
    .with("train::Geometry::check_shape"))
}
