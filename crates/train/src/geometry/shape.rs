//! 职责：几何形状面——`Geometry` 各变体的形状核对、窗口几何与共享工具。

use diagnostic::{Code, Error, Result};
use op::conv::ConvGeom;

use super::{EPS, Geometry};

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

/// 池化输出边长：`(len + 2p − k)/s + 1`，两轴一起求，供核对与反向复用。
pub(super) fn pool2d_out(win: Window, h: usize, w: usize) -> Result<(usize, usize)> {
    Ok((
        pool_len(h, win.k, win.s, win.p)?,
        pool_len(w, win.k, win.s, win.p)?,
    ))
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
